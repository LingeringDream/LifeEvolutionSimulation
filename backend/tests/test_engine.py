import numpy as np
from simulation.engine import SimulationEngine
from simulation.models import Genome
from simulation.species import Species
from simulation.environment import PlanetConfig


class TestSimulationEngine:
    def test_create_engine(self, small_grid_size):
        config = PlanetConfig.titan()
        engine = SimulationEngine(config=config, grid_size=small_grid_size)
        assert engine.tick == 0
        assert len(engine.species_list) == 0

    def test_add_species(self, small_grid_size):
        config = PlanetConfig(surface_temp=20.0, axial_tilt=0.0, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0, co2_ratio=0.0004, ch4_ratio=0.0)
        engine = SimulationEngine(config=config, grid_size=small_grid_size)
        producer = Species.create("p1", "Plant", Genome.create_default(), grid_size=small_grid_size)
        engine.add_species(producer)
        assert len(engine.species_list) == 1

    def test_engine_step_advances_tick(self, small_grid_size):
        config = PlanetConfig(surface_temp=20.0, axial_tilt=0.0, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0, co2_ratio=0.0004, ch4_ratio=0.0)
        engine = SimulationEngine(config=config, grid_size=small_grid_size)
        producer = Species.create("p1", "Plant", Genome.create_default(), grid_size=small_grid_size)
        engine.add_species(producer)
        engine.step()
        assert engine.tick == 1

    def test_producer_grows_with_resources(self, small_grid_size):
        config = PlanetConfig(surface_temp=20.0, axial_tilt=0.0, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0, co2_ratio=0.0004, ch4_ratio=0.0)
        engine = SimulationEngine(config=config, grid_size=small_grid_size)
        producer = Species.create("p1", "Plant", Genome.create_default(), grid_size=small_grid_size, initial_biomass=0.1)
        initial_total = producer.total_biomass()
        engine.add_species(producer)
        for _ in range(10):
            engine.step()
        assert producer.total_biomass() > initial_total * 0.5

    def test_species_extinction_removes_from_list(self, small_grid_size):
        config = PlanetConfig(surface_temp=20.0, axial_tilt=0.0, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0, co2_ratio=0.0004, ch4_ratio=0.0)
        engine = SimulationEngine(config=config, grid_size=small_grid_size)
        producer = Species.create("p1", "Plant", Genome.create_default(), grid_size=small_grid_size)
        engine.add_species(producer)
        producer.biomass = np.zeros((small_grid_size, small_grid_size))
        engine.step()
        alive = [s for s in engine.species_list if s.is_alive()]
        assert len(alive) == 0

    def test_get_snapshot(self, small_grid_size):
        config = PlanetConfig(surface_temp=20.0, axial_tilt=0.0, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0, co2_ratio=0.0004, ch4_ratio=0.0)
        engine = SimulationEngine(config=config, grid_size=small_grid_size)
        producer = Species.create("p1", "Plant", Genome.create_default(), grid_size=small_grid_size)
        engine.add_species(producer)
        engine.step()
        snapshot = engine.get_snapshot()
        assert "tick" in snapshot
        assert "species" in snapshot
        assert "environment" in snapshot
        assert len(snapshot["species"]) == 1
