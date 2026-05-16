"""Planet environment with GPU-accelerated grid operations."""
from __future__ import annotations
import numpy as np  # Keep for random/scalar ops
from pydantic import BaseModel
from simulation.gpu_backend import xp, to_numpy, GPU_AVAILABLE


class PlanetConfig(BaseModel):
    """Physical parameters of a planet."""
    name: str = "Unknown"
    gravity: float = 9.8
    surface_temp: float = 15.0
    albedo: float = 0.3
    axial_tilt: float = 23.5
    orbital_distance: float = 1.0
    atmospheric_pressure: float = 1.0
    co2_ratio: float = 0.0004
    ch4_ratio: float = 0.0
    magnetic_field: float = 1.0
    season_period: int = 365

    @staticmethod
    def titan() -> PlanetConfig:
        return PlanetConfig(name="Titan", gravity=1.35, surface_temp=-179.0, albedo=0.22, axial_tilt=26.7, orbital_distance=9.5, atmospheric_pressure=1.45, co2_ratio=0.0, ch4_ratio=0.05, magnetic_field=0.0, season_period=10759)

    @staticmethod
    def mars() -> PlanetConfig:
        return PlanetConfig(name="Mars", gravity=3.72, surface_temp=-60.0, albedo=0.25, axial_tilt=25.2, orbital_distance=1.52, atmospheric_pressure=0.006, co2_ratio=0.95, ch4_ratio=0.0, magnetic_field=0.0, season_period=687)

    @staticmethod
    def europa() -> PlanetConfig:
        return PlanetConfig(name="Europa", gravity=1.315, surface_temp=-160.0, albedo=0.67, axial_tilt=0.1, orbital_distance=5.2, atmospheric_pressure=0.0000001, co2_ratio=0.0, ch4_ratio=0.0, magnetic_field=0.0, season_period=int(3.55 * 365))

    @staticmethod
    def kepler442b() -> PlanetConfig:
        return PlanetConfig(name="Kepler-442b", gravity=12.16, surface_temp=-2.0, albedo=0.3, axial_tilt=15.0, orbital_distance=0.409, atmospheric_pressure=1.5, co2_ratio=0.01, ch4_ratio=0.001, magnetic_field=1.2, season_period=112)


class Environment:
    """Planet environment with GPU-accelerated grid computations."""

    def __init__(self, config: PlanetConfig, size: int = 50, rng: np.random.RandomState | None = None):
        self.config = config
        self.size = size
        self.rng = rng if rng is not None else np.random.RandomState()
        self.tick_count = 0

        self.latitudes = np.linspace(90, -90, size)

        # Grids on GPU (or CPU if no GPU)
        self.temperature = self._init_temperature()
        self.resources = self._init_resources()
        self.light = self._init_light()
        self.water = self._init_water()

        self.atmosphere = {
            "O2": 0.0,
            "CO2": config.co2_ratio,
            "CH4": config.ch4_ratio,
            "N2": 1.0 - config.co2_ratio - config.ch4_ratio,
            "Pressure": config.atmospheric_pressure,
        }

        self.volcanic_heat = xp.zeros((size, size))
        self._init_volcanoes(n_volcanoes=max(1, size // 10))

    def _to_xp(self, arr):
        """Move numpy array to GPU if available."""
        return xp.asarray(arr) if GPU_AVAILABLE else arr

    def _init_temperature(self):
        base = self.config.surface_temp
        gradient_strength = 30.0 / max(self.config.orbital_distance, 0.1)
        lat_rad = np.deg2rad(self.latitudes)
        lat_factor = np.cos(lat_rad)
        gradient = np.outer(lat_factor, np.ones(self.size)) * gradient_strength
        greenhouse = self._greenhouse_effect()
        temp = base + gradient + greenhouse
        temp += self.rng.normal(0, 1.0, (self.size, self.size))
        return self._to_xp(temp)

    def _greenhouse_effect(self):
        co2_warming = 3.0 * np.log(1 + self.config.co2_ratio / 0.0004) if self.config.co2_ratio > 0 else 0.0
        ch4_warming = 8.0 * np.log(1 + self.config.ch4_ratio / 0.0001) if self.config.ch4_ratio > 0 else 0.0
        return co2_warming + ch4_warming

    def _init_resources(self):
        resources = np.ones((self.size, self.size)) * 0.5
        center = self.size // 2
        y, x = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        resources += 0.5 * np.exp(-dist ** 2 / (2 * (self.size / 3) ** 2))
        resources += self.rng.normal(0, 0.1, (self.size, self.size))
        return self._to_xp(np.clip(resources, 0.1, 2.0))

    def _init_light(self):
        base_light = 1.0 / (self.config.orbital_distance ** 2)
        lat_rad = np.deg2rad(self.latitudes)
        lat_factor = np.cos(lat_rad)
        light = base_light * np.outer(lat_factor, np.ones(self.size))
        light *= min(1.0, self.config.atmospheric_pressure)
        return self._to_xp(np.clip(light, 0.0, 2.0))

    def _init_water(self):
        lat_rad = np.deg2rad(self.latitudes)
        lat_factor = np.cos(lat_rad)
        water = 0.3 + 0.4 * np.outer(lat_factor, np.ones(self.size))
        water += self.rng.normal(0, 0.05, (self.size, self.size))
        return self._to_xp(np.clip(water, 0.0, 1.0))

    def _init_volcanoes(self, n_volcanoes):
        for _ in range(n_volcanoes):
            vy = self.rng.randint(0, self.size)
            vx = self.rng.randint(0, self.size)
            radius = self.rng.randint(2, max(3, self.size // 5))
            y, x = np.ogrid[:self.size, :self.size]
            dist = np.sqrt((x - vx) ** 2 + (y - vy) ** 2)
            heat = 20.0 * np.exp(-dist ** 2 / (2 * radius ** 2))
            self.volcanic_heat += self._to_xp(heat)

    def advance_season(self, ticks: int = 1):
        self.update(ticks=ticks)

    def update(self, ticks: int = 1, biomass_heat=None):
        self.tick_count += ticks
        t = self.tick_count

        # Seasonal oscillation
        season_phase = 2 * np.pi * t / max(self.config.season_period, 1)
        seasonal_amplitude = self.config.axial_tilt * 0.5
        lat_rad = np.deg2rad(self.latitudes)
        seasonal_offset = seasonal_amplitude * np.sin(season_phase) * np.sin(lat_rad)
        seasonal_grid = self._to_xp(np.outer(seasonal_offset, np.ones(self.size)))

        greenhouse = self._greenhouse_effect()
        self.volcanic_heat *= 0.999
        bio_heat = biomass_heat if biomass_heat is not None else xp.zeros((self.size, self.size))

        # All grid math on GPU
        base = self.config.surface_temp
        lat_gradient = 30.0 / max(self.config.orbital_distance, 0.1)
        lat_factor = self._to_xp(np.cos(lat_rad))
        gradient = xp.outer(lat_factor, xp.ones(self.size)) * lat_gradient

        noise = self._to_xp(self.rng.normal(0, 0.3, (self.size, self.size)))
        self.temperature = base + gradient + greenhouse + seasonal_grid + self.volcanic_heat + bio_heat + noise

        # Resource regeneration
        regen_rate = 0.01 * self.light
        self.resources = xp.clip(self.resources + regen_rate, 0.0, 2.0)

    def get_snapshot(self):
        return {
            "tick": self.tick_count,
            "temperature_mean": float(to_numpy(self.temperature).mean()),
            "temperature_min": float(to_numpy(self.temperature).min()),
            "temperature_max": float(to_numpy(self.temperature).max()),
            "resources_mean": float(to_numpy(self.resources).mean()),
            "light_mean": float(to_numpy(self.light).mean()),
            "atmosphere": dict(self.atmosphere),
        }
