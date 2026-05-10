import numpy as np
from simulation.models import Genome, Gene, MetabolicType, DietPreference
from simulation.species import Species
from simulation.food_web import FoodWeb


class TestFoodWeb:
    def test_producer_is_not_prey(self, small_grid_size):
        web = FoodWeb()
        producer = Species.create("p1", "Plant", Genome.create_default(), grid_size=small_grid_size)
        consumer = Species.create("c1", "Herbivore", Genome.create_consumer(), grid_size=small_grid_size)
        web.register(producer)
        web.register(consumer)
        assert producer.id not in web.get_prey(producer.id)

    def test_consumer_eats_producer(self, small_grid_size):
        web = FoodWeb()
        producer = Species.create("p1", "Plant", Genome.create_default(), grid_size=small_grid_size)
        consumer = Species.create("c1", "Herbivore", Genome.create_consumer(), grid_size=small_grid_size)
        web.register(producer)
        web.register(consumer)
        prey_list = web.get_prey(consumer.id)
        assert producer.id in prey_list

    def test_omni_eats_both(self, small_grid_size):
        web = FoodWeb()
        producer = Species.create("p1", "Plant", Genome.create_default(), grid_size=small_grid_size)
        consumer = Species.create("c1", "Herbivore", Genome.create_consumer(), grid_size=small_grid_size)
        omni_genome = Genome.create_consumer()
        omni_genome.genes["diet_preference"] = DietPreference.OMNI.value
        omni = Species.create("o1", "Omnivore", omni_genome, grid_size=small_grid_size)
        web.register(producer)
        web.register(consumer)
        web.register(omni)
        prey_list = web.get_prey(omni.id)
        assert producer.id in prey_list
        assert consumer.id in prey_list

    def test_attack_power_formula(self):
        web = FoodWeb()
        genome = Genome.create_consumer()
        genome.genes["body_size"] = Gene(value=3.0, min_value=0.1, max_value=10.0)
        genome.genes["mobility"] = Gene(value=0.5, min_value=0.0, max_value=1.0)
        genome.genes["sensory_range"] = Gene(value=4.0, min_value=1.0, max_value=15.0)
        attack = web._calc_attack_power(genome)
        assert abs(attack - 3.0 * 0.5 * 4.0) < 0.001

    def test_predation_efficiency(self):
        web = FoodWeb()
        eff = web.predation_efficiency(attack_power=5.0, prey_defense=5.0)
        assert abs(eff - 0.5) < 0.001

    def test_predation_efficiency_no_defense(self):
        web = FoodWeb()
        eff = web.predation_efficiency(attack_power=5.0, prey_defense=0.0)
        assert abs(eff - 1.0) < 0.001
