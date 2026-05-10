from simulation.templates import load_planet_config, load_species_template, list_planets


class TestTemplates:
    def test_list_planets(self):
        planets = list_planets()
        assert "titan" in planets
        assert "mars" in planets
        assert "europa" in planets

    def test_load_titan(self):
        config = load_planet_config("titan")
        assert config.name == "Titan"
        assert config.surface_temp < -100

    def test_load_species_template(self):
        genome = load_species_template("producer_photo")
        assert genome.get_enum("metabolic_type") == "photosynthesis"
