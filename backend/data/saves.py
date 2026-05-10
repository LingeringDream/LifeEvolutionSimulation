"""Save/Load system for simulation state."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
import numpy as np

SAVES_DIR = Path(__file__).parent.parent / "saves"


def _ensure_dir():
    SAVES_DIR.mkdir(parents=True, exist_ok=True)


def list_saves() -> list[dict]:
    """List all saved simulations."""
    _ensure_dir()
    saves = []
    for f in sorted(SAVES_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            saves.append({
                "id": f.stem,
                "name": meta.get("name", f.stem),
                "planet": meta.get("planet", "?"),
                "tick": meta.get("tick", 0),
                "n_species": meta.get("n_species", 0),
                "saved_at": meta.get("saved_at", 0),
                "auto": meta.get("auto", False),
                "file": str(f),
            })
        except Exception:
            continue
    return saves


def save_simulation(engine, name: str = "", auto: bool = False) -> str:
    """Save the full simulation state to a JSON file.

    Args:
        engine: SimulationEngine instance.
        name: User-given name for the save.
        auto: Whether this is an auto-save.

    Returns:
        Save ID (filename stem).
    """
    _ensure_dir()

    save_id = f"save_{int(time.time())}_{engine.tick}"
    if name:
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()
        save_id = f"{safe_name}_{engine.tick}"

    # Serialize species
    species_data = []
    for sp in engine.species_list:
        genes = {}
        for k, v in sp.genome.genes.items():
            if hasattr(v, "value"):
                genes[k] = {
                    "type": "float",
                    "value": v.value,
                    "min": v.min_value,
                    "max": v.max_value,
                    "mut_rate": v.mutation_rate,
                    "dom": v.dominance,
                }
            else:
                genes[k] = {"type": "enum", "value": v}

        species_data.append({
            "id": sp.id,
            "name": sp.name,
            "genes": genes,
            "generation": sp.genome.generation,
            "parent_ids": sp.genome.parent_ids,
            "biomass": sp.biomass.tolist(),
            "ancestor_id": sp.ancestor_id,
            "color": list(sp.color),
        })

    # Serialize environment
    env = engine.env
    env_data = {
        "temperature": env.temperature.tolist(),
        "resources": env.resources.tolist(),
        "light": env.light.tolist(),
        "water": env.water.tolist(),
        "volcanic_heat": env.volcanic_heat.tolist(),
        "atmosphere": dict(env.atmosphere),
        "tick_count": env.tick_count,
    }

    # Serialize events
    events_data = [
        {"tick": e.tick, "type": e.event_type, "desc": e.description, "details": e.details}
        for e in engine.events
    ]

    data = {
        "version": 1,
        "name": name or f"Tick {engine.tick}",
        "planet": engine.config.name,
        "tick": engine.tick,
        "n_species": len(engine.species_list),
        "saved_at": time.time(),
        "auto": auto,
        "grid_size": engine.grid_size,
        "config": {
            "name": engine.config.name,
            "gravity": engine.config.gravity,
            "surface_temp": engine.config.surface_temp,
            "albedo": engine.config.albedo,
            "axial_tilt": engine.config.axial_tilt,
            "orbital_distance": engine.config.orbital_distance,
            "atmospheric_pressure": engine.config.atmospheric_pressure,
            "co2_ratio": engine.config.co2_ratio,
            "ch4_ratio": engine.config.ch4_ratio,
            "magnetic_field": engine.config.magnetic_field,
            "season_period": engine.config.season_period,
        },
        "species": species_data,
        "environment": env_data,
        "events": events_data,
        "narratives": engine.ai_narratives,
        "species_counter": engine._species_counter,
    }

    path = SAVES_DIR / f"{save_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return save_id


def load_simulation(save_id: str) -> dict:
    """Load a saved simulation state.

    Args:
        save_id: The save ID (filename stem).

    Returns:
        Dict with all data needed to reconstruct the engine.

    Raises:
        FileNotFoundError: If save doesn't exist.
    """
    path = SAVES_DIR / f"{save_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Save not found: {save_id}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_save(save_id: str) -> bool:
    """Delete a save file."""
    path = SAVES_DIR / f"{save_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False
