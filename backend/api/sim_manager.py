"""Simulation manager — runs the sim in a background thread, broadcasts state via WebSocket."""
from __future__ import annotations
import asyncio
import json
import threading
import time
import numpy as np
from simulation.engine import SimulationEngine
from simulation.species import Species
from simulation.templates import load_planet_config, load_species_template
from ai.provider import AIProvider


class SimManager:
    """Manages a running simulation, decoupled from the web server."""

    def __init__(self):
        self.engine: SimulationEngine | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._speed = 10
        self._ws_clients: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── lifecycle ──────────────────────────────────────────────

    def start(
        self,
        planet: str = "titan",
        producers: int = 2,
        consumers: int = 1,
        grid_size: int = 50,
        ai_provider: AIProvider | None = None,
        ai_interval: int = 60,
    ):
        if self._running:
            self.stop()
            time.sleep(0.2)

        config = load_planet_config(planet)
        self.engine = SimulationEngine(
            config=config,
            grid_size=grid_size,
            ai_provider=ai_provider,
            ai_interval=ai_interval,
        )

        for i in range(producers):
            genome = load_species_template("producer_photo" if i == 0 else "producer_chemo")
            sp = Species.create(f"producer_{i}", f"Producer-{i}", genome, grid_size=grid_size)
            self.engine.add_species(sp)
        for i in range(consumers):
            genome = load_species_template("consumer_herbivore")
            sp = Species.create(f"consumer_{i}", f"Consumer-{i}", genome, grid_size=grid_size, seed_area="random")
            self.engine.add_species(sp)

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def pause(self):
        self._running = False
        # Immediately broadcast paused state
        self._schedule_broadcast(self._build_broadcast())

    def resume(self):
        if self.engine and not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def set_speed(self, tps: int):
        self._speed = max(1, min(200, tps))

    # ── websocket broadcast ────────────────────────────────────

    def register_ws(self, queue: asyncio.Queue):
        self._ws_clients.append(queue)

    def unregister_ws(self, queue: asyncio.Queue):
        if queue in self._ws_clients:
            self._ws_clients.remove(queue)

    def _schedule_broadcast(self, data: dict):
        """Thread-safe broadcast scheduling."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._do_broadcast, data)
        else:
            self._do_broadcast(data)

    def _do_broadcast(self, data: dict):
        payload = json.dumps(data, default=_json_default)
        for q in list(self._ws_clients):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # Drop oldest to make room
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass

    # ── simulation loop ────────────────────────────────────────

    def _run_loop(self):
        while self._running and self.engine:
            t0 = time.time()

            if self.engine.ai_provider:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.engine.step_async())
                loop.close()
            else:
                self.engine.step()

            snapshot = self._build_broadcast()
            self._schedule_broadcast(snapshot)

            elapsed = time.time() - t0
            delay = max(0, 1.0 / self._speed - elapsed)
            time.sleep(delay)

        # Final broadcast with running=False
        if self.engine:
            final = self._build_broadcast()
            final["running"] = False
            self._schedule_broadcast(final)

    def _build_broadcast(self) -> dict:
        eng = self.engine
        if eng is None:
            return {"type": "state", "tick": 0, "running": False, "speed": self._speed,
                    "species": [], "environment": {}, "events": [], "narratives": []}

        species_data = []
        for sp in eng.species_list:
            if not sp.is_alive():
                continue
            species_data.append({
                "id": sp.id,
                "name": sp.name,
                "biomass": round(sp.total_biomass(), 3),
                "metabolic_type": sp.genome.get_enum("metabolic_type"),
                "color": list(sp.color),
                "genes": {
                    k: round(v.value, 3) if hasattr(v, "value") else v
                    for k, v in sp.genome.genes.items()
                },
            })

        # Include grid data in every broadcast (for canvas rendering)
        total_biomass = np.zeros((eng.grid_size, eng.grid_size))
        species_layers = {}
        for sp in eng.species_list:
            if sp.is_alive():
                total_biomass += sp.biomass
                species_layers[sp.id] = {
                    "name": sp.name,
                    "color": list(sp.color),
                    "biomass": sp.biomass.tolist(),
                }

        return {
            "type": "state",
            "tick": eng.tick,
            "running": self._running,
            "speed": self._speed,
            "species": species_data,
            "environment": eng.env.get_snapshot(),
            "events": [
                {"tick": e.tick, "type": e.event_type, "desc": e.description}
                for e in eng.events[-20:]
            ],
            "narratives": eng.ai_narratives[-5:],
            # Grid data for canvas
            "grid_size": eng.grid_size,
            "total_biomass": total_biomass.tolist(),
            "temperature": eng.env.temperature.tolist(),
            "resources": eng.env.resources.tolist(),
            "species_layers": species_layers,
        }

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")
