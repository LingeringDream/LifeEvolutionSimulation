from __future__ import annotations
import json
import os
from pathlib import Path
from simulation.environment import PlanetConfig
from simulation.models import Genome, Gene, MetabolicType, DietPreference


CONFIG_DIR = Path(__file__).parent.parent / "config"


def list_planets() -> list[str]:
    """List available planet template names."""
    planets_dir = CONFIG_DIR / "planets"
    if not planets_dir.exists():
        return []
    return [f.stem for f in planets_dir.glob("*.json")]


def list_species_templates() -> list[str]:
    """List available species template names."""
    species_dir = CONFIG_DIR / "species"
    if not species_dir.exists():
        return []
    return [f.stem for f in species_dir.glob("*.json")]


def load_planet_config(name: str) -> PlanetConfig:
    """Load a planet configuration from JSON template.

    Args:
        name: Planet name (e.g., "titan", "mars").

    Returns:
        PlanetConfig instance.

    Raises:
        FileNotFoundError: If the template file doesn't exist.
    """
    path = CONFIG_DIR / "planets" / f"{name.lower()}.json"
    if not path.exists():
        raise FileNotFoundError(f"Planet template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PlanetConfig(**data)


def load_species_template(name: str) -> Genome:
    """Load a species genome from JSON template.

    Args:
        name: Species template name (e.g., "producer_photo").

    Returns:
        Genome instance.

    Raises:
        FileNotFoundError: If the template file doesn't exist.
    """
    path = CONFIG_DIR / "species" / f"{name.lower()}.json"
    if not path.exists():
        raise FileNotFoundError(f"Species template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract name (not a gene)
    data.pop("name", None)

    # Build genes dict
    genes = {}
    enum_genes = {"metabolic_type", "diet_preference"}

    for key, val in data.items():
        if key in enum_genes:
            genes[key] = val  # Store as string for enum genes
        elif isinstance(val, (int, float)):
            # Determine bounds based on gene name
            min_val, max_val, mut_rate = _get_gene_bounds(key)
            genes[key] = Gene(
                value=float(val),
                min_value=min_val,
                max_value=max_val,
                mutation_rate=mut_rate,
            )

    return Genome(genes=genes)


def _get_gene_bounds(name: str) -> tuple[float, float, float]:
    """Return (min, max, mutation_rate) for a gene name."""
    bounds = {
        "body_size": (0.1, 10.0, 0.03),
        "temp_optimum": (-250.0, 500.0, 0.02),
        "temp_tolerance": (5.0, 150.0, 0.02),
        "reproduction_rate": (0.01, 3.0, 0.03),
        "reproduction_cost": (0.1, 5.0, 0.02),
        "defense": (0.0, 1.0, 0.04),
        "mobility": (0.0, 1.0, 0.03),
        "sensory_range": (1.0, 15.0, 0.03),
        "lifespan": (10.0, 2000.0, 0.02),
        "adaptability": (0.0, 1.0, 0.04),
    }
    return bounds.get(name, (0.0, 1.0, 0.05))
