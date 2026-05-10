from simulation.models import Gene, Genome, MetabolicType, DietPreference


class TestGene:
    def test_create_float_gene(self):
        gene = Gene(value=0.5, min_value=0.0, max_value=1.0, mutation_rate=0.05)
        assert gene.value == 0.5
        assert gene.min_value == 0.0
        assert gene.max_value == 1.0

    def test_gene_clamps_value(self):
        gene = Gene(value=1.5, min_value=0.0, max_value=1.0)
        assert gene.value == 1.0

    def test_gene_clamps_negative(self):
        gene = Gene(value=-0.5, min_value=0.0, max_value=1.0)
        assert gene.value == 0.0


class TestGenome:
    def test_create_genome_with_defaults(self):
        genome = Genome.create_default()
        assert "body_size" in genome.genes
        assert "temp_optimum" in genome.genes
        assert genome.generation == 0

    def test_genome_has_all_required_genes(self):
        genome = Genome.create_default()
        required = [
            "body_size", "metabolic_type", "temp_optimum", "temp_tolerance",
            "reproduction_rate", "reproduction_cost", "defense", "mobility",
            "sensory_range", "diet_preference", "lifespan", "adaptability"
        ]
        for gene_name in required:
            assert gene_name in genome.genes, f"Missing gene: {gene_name}"

    def test_metabolic_type_is_enum(self):
        assert MetabolicType.PHOTOSYNTHESIS.value == "photosynthesis"
        assert MetabolicType.CHEMOSYNTHESIS.value == "chemosynthesis"
        assert MetabolicType.HETEROTROPHY.value == "heterotrophy"

    def test_diet_preference_is_enum(self):
        assert DietPreference.PRODUCER.value == "producer"
        assert DietPreference.CONSUMER.value == "consumer"
        assert DietPreference.OMNI.value == "omni"
