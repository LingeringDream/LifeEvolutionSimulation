import numpy as np
from simulation.models import Genome
from simulation.species import Species


class TestSpecies:
    def test_create_species(self, small_grid_size):
        genome = Genome.create_default()
        sp = Species.create("sp_001", "TestPlant", genome, grid_size=small_grid_size)
        assert sp.id == "sp_001"
        assert sp.name == "TestPlant"
        assert sp.biomass.shape == (small_grid_size, small_grid_size)
        assert np.sum(sp.biomass) > 0

    def test_species_total_biomass(self, small_grid_size):
        genome = Genome.create_default()
        sp = Species.create("sp_001", "TestPlant", genome, grid_size=small_grid_size)
        assert sp.total_biomass() > 0

    def test_species_is_alive(self, small_grid_size):
        genome = Genome.create_default()
        sp = Species.create("sp_001", "TestPlant", genome, grid_size=small_grid_size)
        assert sp.is_alive()

    def test_species_dies_when_biomass_zero(self, small_grid_size):
        genome = Genome.create_default()
        sp = Species.create("sp_001", "TestPlant", genome, grid_size=small_grid_size)
        sp.biomass = np.zeros((small_grid_size, small_grid_size))
        assert not sp.is_alive()

    def test_species_center_seed(self, small_grid_size):
        genome = Genome.create_default()
        sp = Species.create("sp_001", "TestPlant", genome, grid_size=small_grid_size, seed_area="center")
        mid = small_grid_size // 2
        assert sp.biomass[mid, mid] > 0
        assert sp.biomass[0, 0] == 0.0

    def test_species_random_seed(self, small_grid_size):
        genome = Genome.create_default()
        rng = np.random.RandomState(42)
        sp = Species.create("sp_001", "TestPlant", genome, grid_size=small_grid_size, seed_area="random", rng=rng)
        assert sp.total_biomass() > 0
