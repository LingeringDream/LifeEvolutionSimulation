import numpy as np
from simulation.models import Genome, Gene, MetabolicType
from simulation.genetics import mutate_genome, crossover_genomes, calculate_genetic_distance


class TestMutation:
    def test_mutation_changes_value(self):
        rng = np.random.RandomState(42)
        original = Genome.create_default()
        original_val = original.get_float("body_size")

        mutated_any = False
        for _ in range(100):
            mutated = mutate_genome(original, rng=rng)
            if mutated.get_float("body_size") != original_val:
                mutated_any = True
                break
        assert mutated_any, "Mutation never changed body_size in 100 attempts"

    def test_mutation_respects_bounds(self):
        rng = np.random.RandomState(42)
        genome = Genome.create_default()
        for _ in range(1000):
            mutated = mutate_genome(genome, rng=rng)
            val = mutated.get_float("body_size")
            assert 0.1 <= val <= 10.0

    def test_mutation_increments_generation(self):
        rng = np.random.RandomState(42)
        original = Genome.create_default()
        mutated = mutate_genome(original, rng=rng)
        assert mutated.generation == original.generation + 1

    def test_mutation_preserves_parent_id(self):
        rng = np.random.RandomState(42)
        original = Genome.create_default()
        mutated = mutate_genome(original, parent_id="sp_001", rng=rng)
        assert "sp_001" in mutated.parent_ids


class TestCrossover:
    def test_crossover_produces_valid_genome(self):
        rng = np.random.RandomState(42)
        parent_a = Genome.create_default()
        parent_b = Genome.create_consumer()
        child = crossover_genomes(parent_a, parent_b, rng=rng)
        assert "body_size" in child.genes
        assert "metabolic_type" in child.genes
        assert child.generation == max(parent_a.generation, parent_b.generation) + 1

    def test_crossover_child_is_different_from_parents(self):
        rng = np.random.RandomState(42)
        parent_a = Genome.create_default()
        parent_b = Genome.create_consumer()
        child = crossover_genomes(parent_a, parent_b, rng=rng)
        differs = False
        for name in ["body_size", "temp_optimum", "defense"]:
            child_val = child.get_float(name)
            a_val = parent_a.get_float(name)
            b_val = parent_b.get_float(name)
            if child_val != a_val or child_val != b_val:
                differs = True
                break
        if child.get_enum("metabolic_type") != parent_a.get_enum("metabolic_type"):
            differs = True
        assert differs


class TestGeneticDistance:
    def test_identical_genomes_have_zero_distance(self):
        g = Genome.create_default()
        assert calculate_genetic_distance(g, g) == 0.0

    def test_different_genomes_have_positive_distance(self):
        a = Genome.create_default()
        b = Genome.create_consumer()
        dist = calculate_genetic_distance(a, b)
        assert dist > 0.0
