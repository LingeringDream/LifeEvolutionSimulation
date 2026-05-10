import numpy as np
from simulation.environment import Environment, PlanetConfig


class TestPlanetConfig:
    def test_default_titan_config(self):
        config = PlanetConfig.titan()
        assert config.name == "Titan"
        assert config.gravity < 10.0
        assert config.surface_temp < -100.0

    def test_custom_config(self):
        config = PlanetConfig(
            name="TestPlanet",
            gravity=9.8,
            surface_temp=20.0,
            albedo=0.3,
            axial_tilt=23.5,
            orbital_distance=1.0,
            atmospheric_pressure=1.0,
            co2_ratio=0.0004,
            ch4_ratio=0.0,
        )
        assert config.name == "TestPlanet"


class TestEnvironment:
    def test_create_environment(self, small_grid_size):
        config = PlanetConfig.titan()
        env = Environment(config, size=small_grid_size)
        assert env.temperature.shape == (small_grid_size, small_grid_size)
        assert env.resources.shape == (small_grid_size, small_grid_size)
        assert env.light.shape == (small_grid_size, small_grid_size)

    def test_temperature_has_latitude_gradient(self, small_grid_size):
        config = PlanetConfig(surface_temp=20.0, axial_tilt=0.0, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0, co2_ratio=0.0004, ch4_ratio=0.0)
        env = Environment(config, size=small_grid_size)
        equator_temp = np.mean(env.temperature[small_grid_size // 2 - 1 : small_grid_size // 2 + 1, :])
        pole_temp = np.mean(env.temperature[0:2, :])
        assert equator_temp > pole_temp

    def test_greenhouse_effect_warms_planet(self, small_grid_size):
        config_cold = PlanetConfig(surface_temp=20.0, co2_ratio=0.0004, ch4_ratio=0.0, axial_tilt=0.0, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0)
        config_warm = PlanetConfig(surface_temp=20.0, co2_ratio=0.15, ch4_ratio=0.05, axial_tilt=0.0, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0)
        env_cold = Environment(config_cold, size=small_grid_size)
        env_warm = Environment(config_warm, size=small_grid_size)
        assert np.mean(env_warm.temperature) > np.mean(env_cold.temperature)

    def test_seasonal_cycle_changes_temperature(self, small_grid_size):
        config = PlanetConfig(surface_temp=20.0, axial_tilt=23.5, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0, co2_ratio=0.0004, ch4_ratio=0.0)
        env = Environment(config, size=small_grid_size)
        temp_tick_0 = env.temperature.copy()
        env.advance_season(ticks=100)
        temp_tick_100 = env.temperature.copy()
        assert not np.allclose(temp_tick_0, temp_tick_100, atol=0.01)

    def test_resources_have_initial_distribution(self, small_grid_size):
        config = PlanetConfig.titan()
        env = Environment(config, size=small_grid_size)
        assert np.all(env.resources >= 0)
        assert np.mean(env.resources) > 0
