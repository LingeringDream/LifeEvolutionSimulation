from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from simulation.models import Genome, MetabolicType
from simulation.gpu_backend import to_numpy


@dataclass
class EvolutionEvent:
    tick: int
    event_type: str
    description: str
    details: dict = field(default_factory=dict)


@dataclass
class Species:
    id: str
    name: str
    genome: Genome
    biomass: np.ndarray
    ancestor_id: str | None = None
    extinction_tick: int | None = None
    color: tuple[int, int, int] = (100, 200, 100)
    history: list[EvolutionEvent] = field(default_factory=list)

    @staticmethod
    def create(
        species_id: str,
        name: str,
        genome: Genome,
        grid_size: int = 50,
        initial_biomass: float = 0.5,
        seed_area: str = "center",
        rng: np.random.RandomState | None = None,
    ) -> Species:
        if rng is None:
            rng = np.random.RandomState()

        biomass = np.zeros((grid_size, grid_size))

        if seed_area == "center":
            mid = grid_size // 2
            radius = max(2, grid_size // 8)
            y, x = np.ogrid[:grid_size, :grid_size]
            dist = np.sqrt((x - mid) ** 2 + (y - mid) ** 2)
            mask = dist < radius
            biomass[mask] = initial_biomass
        elif seed_area == "random":
            n_patches = rng.randint(3, 8)
            for _ in range(n_patches):
                py, px = rng.randint(0, grid_size, 2)
                radius = rng.randint(1, max(2, grid_size // 6))
                y, x = np.ogrid[:grid_size, :grid_size]
                dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                mask = dist < radius
                biomass[mask] = initial_biomass * rng.uniform(0.3, 1.0)
        elif seed_area == "everywhere":
            biomass[:] = initial_biomass * 0.1
            biomass += rng.uniform(0, 0.05, (grid_size, grid_size))

        # Generate visually distinct color using HSL
        # Hue range depends on metabolic type for semantic grouping
        import colorsys
        meta_type = genome.get_enum("metabolic_type")
        # Use species_id hash for deterministic but diverse hue
        hue_seed = hash(species_id) % 1000 / 1000.0

        if meta_type == MetabolicType.PHOTOSYNTHESIS.value:
            # Greens: hue 80-160
            h = 0.22 + hue_seed * 0.22  # 80°-160°
            s, l = 0.7 + hue_seed * 0.2, 0.35 + hue_seed * 0.15
        elif meta_type == MetabolicType.CHEMOSYNTHESIS.value:
            # Blues/purples: hue 200-280
            h = 0.55 + hue_seed * 0.22  # 200°-280°
            s, l = 0.65 + hue_seed * 0.2, 0.4 + hue_seed * 0.15
        else:
            # Reds/oranges/yellows: hue 0-60
            h = hue_seed * 0.17  # 0°-60°
            s, l = 0.7 + hue_seed * 0.2, 0.4 + hue_seed * 0.15

        r, g, b = colorsys.hls_to_rgb(h, l, s)
        color = (int(r * 255), int(g * 255), int(b * 255))

        return Species(
            id=species_id,
            name=name,
            genome=genome,
            biomass=biomass,
            color=color,
        )

    def total_biomass(self) -> float:
        return float(to_numpy(self.biomass).sum())

    def is_alive(self) -> bool:
        try:
            s = self.biomass.sum()
            return float(s) > 0.001
        except Exception:
            return float(to_numpy(self.biomass).sum()) > 0.001

    def mean_fitness(self) -> float:
        return 0.5
