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
from data.database import (
    init_db, create_run, finish_run, save_snapshot, save_events,
    list_runs, get_run, get_snapshots, get_species_history, get_events,
    export_run_csv, export_run_json,
)
from data.saves import save_simulation, load_simulation, list_saves, delete_save

SNAPSHOT_INTERVAL = 10  # Save to DB every N ticks


class SimManager:
    """Manages a running simulation, decoupled from the web server."""

    def __init__(self):
        self.engine: SimulationEngine | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._speed = 10
        self._ws_clients: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._run_id: int | None = None
        self._last_snapshot_tick: int = 0
        self._planet_name: str = ""
        self._grid_size: int = 50
        self._species_counter: int = 0
        self._auto_save: bool = False
        self._auto_save_interval: int = 500
        self._last_auto_save_tick: int = 0

        # Init database
        init_db()

    # ── lifecycle ──────────────────────────────────────────────

    def start(
        self,
        planet: str = "titan",
        producers: int = 2,
        consumers: int = 1,
        grid_size: int = 50,
        ai_provider: AIProvider | None = None,
        ai_interval: int = 60,
        custom_planet: dict | None = None,
        custom_species: list[dict] | None = None,
    ):
        if self._running:
            self.stop()
            time.sleep(0.2)

        from simulation.environment import PlanetConfig as PC
        if planet == "custom" and custom_planet:
            config = PC(**custom_planet)
        else:
            config = load_planet_config(planet)
        self._planet_name = planet
        self._grid_size = grid_size

        self.engine = SimulationEngine(
            config=config,
            grid_size=grid_size,
            ai_provider=ai_provider,
            ai_interval=ai_interval,
        )

        if custom_species:
            for i, cs in enumerate(custom_species):
                genes = {}
                for k, v in cs.items():
                    if k in ("name", "seed_area", "metabolic_type", "diet_preference"):
                        continue
                    if isinstance(v, (int, float)):
                        from simulation.models import Gene
                        bounds = {
                            "body_size": (0.1, 10.0, 0.03), "temp_optimum": (-250.0, 500.0, 0.02),
                            "temp_tolerance": (5.0, 150.0, 0.02), "reproduction_rate": (0.01, 3.0, 0.03),
                            "reproduction_cost": (0.1, 5.0, 0.02), "defense": (0.0, 1.0, 0.04),
                            "mobility": (0.0, 1.0, 0.03), "sensory_range": (1.0, 15.0, 0.03),
                            "lifespan": (10.0, 2000.0, 0.02), "adaptability": (0.0, 1.0, 0.04),
                        }
                        mn, mx, mr = bounds.get(k, (0, 1, 0.05))
                        genes[k] = Gene(value=float(v), min_value=mn, max_value=mx, mutation_rate=mr)
                genes["metabolic_type"] = cs.get("metabolic_type", "photosynthesis")
                genes["diet_preference"] = cs.get("diet_preference", "producer")
                from simulation.models import Genome
                genome = Genome(genes=genes)
                sp = Species.create(
                    species_id=self._next_species_id(),
                    name=cs.get("name", f"Species-{i}"),
                    genome=genome, grid_size=grid_size,
                    seed_area=cs.get("seed_area", "center"),
                )
                self.engine.add_species(sp)
        else:
            for i in range(producers):
                genome = load_species_template("producer_photo" if i == 0 else "producer_chemo")
                sp = Species.create(f"producer_{i}", f"Producer-{i}", genome, grid_size=grid_size)
                self.engine.add_species(sp)
            for i in range(consumers):
                genome = load_species_template("consumer_herbivore")
                sp = Species.create(f"consumer_{i}", f"Consumer-{i}", genome, grid_size=grid_size, seed_area="random")
                self.engine.add_species(sp)

        # Create DB run record
        ai_name = type(ai_provider).__name__ if ai_provider else None
        ai_model = getattr(ai_provider, 'model', None) if ai_provider else None
        self._run_id = create_run(
            planet=planet, grid_size=grid_size, producers=producers, consumers=consumers,
            ai_provider=ai_name, ai_model=ai_model, ai_interval=ai_interval if ai_provider else None,
        )
        self._last_snapshot_tick = 0

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._run_id and self.engine:
            finish_run(self._run_id, self.engine.tick)

    def pause(self):
        self._running = False
        if self._run_id and self.engine:
            # Save final snapshot on pause
            self._save_current_snapshot()
            finish_run(self._run_id, self.engine.tick)
        self._schedule_broadcast(self._build_broadcast())

    def resume(self):
        if self.engine and not self._running:
            # Create new run record for resumed session
            self._run_id = create_run(
                planet=self._planet_name, grid_size=self._grid_size,
                producers=0, consumers=0, config_json='{"resumed": true}',
            )
            self._last_snapshot_tick = self.engine.tick
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def set_speed(self, tps: int):
        self._speed = max(1, min(200, tps))

    def _next_species_id(self) -> str:
        self._species_counter = getattr(self, '_species_counter', 0) + 1
        return f"sp_{self._species_counter:03d}"

    # ── websocket broadcast ────────────────────────────────────

    def register_ws(self, queue: asyncio.Queue):
        self._ws_clients.append(queue)

    def unregister_ws(self, queue: asyncio.Queue):
        if queue in self._ws_clients:
            self._ws_clients.remove(queue)

    def _schedule_broadcast(self, data: dict):
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

            # Save snapshot to DB periodically
            if self._run_id and (self.engine.tick - self._last_snapshot_tick >= SNAPSHOT_INTERVAL):
                self._save_current_snapshot()

            # Auto-save
            if self._auto_save and self.engine.tick - self._last_auto_save_tick >= self._auto_save_interval:
                self._do_auto_save()

            snapshot = self._build_broadcast()
            self._schedule_broadcast(snapshot)

            elapsed = time.time() - t0
            delay = max(0, 1.0 / self._speed - elapsed)
            time.sleep(delay)

        # Final broadcast with running=False
        if self.engine:
            self._save_current_snapshot()
            final = self._build_broadcast()
            final["running"] = False
            self._schedule_broadcast(final)

    def _save_current_snapshot(self):
        """Save current engine state to database."""
        eng = self.engine
        if not eng or not self._run_id:
            return

        species_data = []
        for sp in eng.species_list:
            if not sp.is_alive():
                continue
            species_data.append({
                "id": sp.id,
                "name": sp.name,
                "biomass": round(sp.total_biomass(), 3),
                "metabolic_type": sp.genome.get_enum("metabolic_type"),
                "genes": {
                    k: round(v.value, 3) if hasattr(v, "value") else v
                    for k, v in sp.genome.genes.items()
                },
            })

        events_data = [
            {"tick": e.tick, "type": e.event_type, "desc": e.description}
            for e in eng.events[-20:]
        ]

        try:
            save_snapshot(self._run_id, eng.tick, eng.env.get_snapshot(), species_data, events_data)
            save_events(self._run_id, events_data)
            self._last_snapshot_tick = eng.tick
        except Exception as e:
            print(f"[DB] Snapshot save error: {e}")

    def _build_broadcast(self) -> dict:
        eng = self.engine
        if eng is None:
            return {"type": "state", "tick": 0, "running": False, "speed": self._speed,
                    "species": [], "environment": {}, "events": [], "narratives": [],
                    "grid_size": self._grid_size, "total_biomass": [], "temperature": [],
                    "resources": [], "species_layers": {}, "run_id": None}

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
            "grid_size": eng.grid_size,
            "total_biomass": total_biomass.tolist(),
            "temperature": eng.env.temperature.tolist(),
            "resources": eng.env.resources.tolist(),
            "species_layers": species_layers,
            "run_id": self._run_id,
        }

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def get_state(self) -> dict:
        return self._build_broadcast()

    # ── data access ────────────────────────────────────────────

    def list_runs(self) -> list[dict]:
        return list_runs()

    def get_run(self, run_id: int) -> dict | None:
        return get_run(run_id)

    def get_snapshots(self, run_id: int) -> list[dict]:
        return get_snapshots(run_id)

    def get_species_history(self, run_id: int, species_id: str | None = None) -> list[dict]:
        return get_species_history(run_id, species_id)

    def get_events(self, run_id: int) -> list[dict]:
        return get_events(run_id)

    def export_csv(self, run_id: int, output_dir: str) -> dict[str, str]:
        return export_run_csv(run_id, output_dir)

    def export_json(self, run_id: int, output_path: str) -> str:
        return export_run_json(run_id, output_path)

    # ── save/load ───────────────────────────────────────────────

    def set_auto_save(self, enabled: bool, interval: int = 500):
        self._auto_save = enabled
        self._auto_save_interval = max(50, interval)

    def _do_auto_save(self):
        if not self.engine:
            return
        try:
            save_id = save_simulation(self.engine, name=f"自动存档 tick {self.engine.tick}", auto=True)
            self._last_auto_save_tick = self.engine.tick
            print(f"[Auto-Save] {save_id}")
        except Exception as e:
            print(f"[Auto-Save Error] {e}")

    def save(self, name: str = "") -> str | None:
        """Manual save. Returns save_id."""
        if not self.engine:
            return None
        return save_simulation(self.engine, name=name, auto=False)

    def load(self, save_id: str, ai_provider: AIProvider | None = None) -> bool:
        """Load a saved simulation."""
        try:
            data = load_simulation(save_id)
            self.stop()
            time.sleep(0.2)
            self.engine = SimulationEngine.from_save(data, ai_provider=ai_provider)
            self._planet_name = data.get("planet", "")
            self._grid_size = data.get("grid_size", 50)
            self._last_auto_save_tick = self.engine.tick
            # Broadcast loaded state
            self._schedule_broadcast(self._build_broadcast())
            return True
        except Exception as e:
            print(f"[Load Error] {e}")
            return False

    def list_saves(self) -> list[dict]:
        return list_saves()

    def delete_save(self, save_id: str) -> bool:
        return delete_save(save_id)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")
