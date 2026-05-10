from __future__ import annotations
import copy
import numpy as np
from simulation.models import Genome, Gene, MetabolicType, DietPreference


def mutate_genome(genome: Genome, mutation_strength: float = 0.05, parent_id: str | None = None, rng: np.random.RandomState | None = None) -> Genome:
    """Create a mutated copy of a genome.

    Args:
        genome: The parent genome to mutate.
        mutation_strength: Fraction of gene range for mutation magnitude.
        parent_id: ID of the parent species to record in lineage.
        rng: Random state for reproducibility.

    Returns:
        A new Genome with mutations applied.
    """
    if rng is None:
        rng = np.random.RandomState()

    new_genes: dict[str, Gene | str] = {}
    for name, gene in genome.genes.items():
        if isinstance(gene, Gene):
            if rng.random() < gene.mutation_rate:
                delta_range = (gene.max_value - gene.min_value) * mutation_strength
                delta = rng.normal(0, delta_range)
                new_val = gene.value + delta
                new_val = max(gene.min_value, min(gene.max_value, new_val))
                new_genes[name] = Gene(
                    value=new_val,
                    min_value=gene.min_value,
                    max_value=gene.max_value,
                    mutation_rate=gene.mutation_rate,
                    dominance=gene.dominance,
                )
            else:
                new_genes[name] = gene.model_copy(deep=True)
        else:
            # Enum gene — small chance to flip
            if rng.random() < 0.01:
                if name == "metabolic_type":
                    options = [e.value for e in MetabolicType]
                    new_genes[name] = rng.choice(options)
                elif name == "diet_preference":
                    options = [e.value for e in DietPreference]
                    new_genes[name] = rng.choice(options)
                else:
                    new_genes[name] = gene
            else:
                new_genes[name] = gene

    new_parent_ids = list(genome.parent_ids)
    if parent_id is not None:
        new_parent_ids.append(parent_id)

    return Genome(
        genes=new_genes,
        generation=genome.generation + 1,
        parent_ids=new_parent_ids,
    )


def crossover_genomes(parent_a: Genome, parent_b: Genome, rng: np.random.RandomState | None = None) -> Genome:
    """Create a child genome by crossing two parents.

    Each gene is randomly selected from one parent, with a chance of blending for float genes.
    """
    if rng is None:
        rng = np.random.RandomState()

    all_keys = set(parent_a.genes.keys()) | set(parent_b.genes.keys())
    child_genes: dict[str, Gene | str] = {}

    for name in all_keys:
        if name not in parent_a.genes:
            child_genes[name] = copy.deepcopy(parent_b.genes[name])
        elif name not in parent_b.genes:
            child_genes[name] = copy.deepcopy(parent_a.genes[name])
        else:
            gene_a = parent_a.genes[name]
            gene_b = parent_b.genes[name]

            if isinstance(gene_a, Gene) and isinstance(gene_b, Gene):
                alpha = rng.uniform(0.3, 0.7)
                blended_val = gene_a.value * alpha + gene_b.value * (1 - alpha)
                blended_val += rng.normal(0, (gene_a.max_value - gene_a.min_value) * 0.02)
                blended_val = max(gene_a.min_value, min(gene_a.max_value, blended_val))
                child_genes[name] = Gene(
                    value=blended_val,
                    min_value=gene_a.min_value,
                    max_value=gene_a.max_value,
                    mutation_rate=(gene_a.mutation_rate + gene_b.mutation_rate) / 2,
                    dominance=(gene_a.dominance + gene_b.dominance) / 2,
                )
            else:
                child_genes[name] = gene_a if rng.random() < 0.5 else gene_b

    return Genome(
        genes=child_genes,
        generation=max(parent_a.generation, parent_b.generation) + 1,
        parent_ids=[],
    )


def calculate_genetic_distance(genome_a: Genome, genome_b: Genome) -> float:
    """Calculate normalized genetic distance between two genomes.

    Returns a value between 0.0 (identical) and ~1.0 (maximally different).
    """
    total_distance = 0.0
    count = 0

    for name in genome_a.genes:
        if name not in genome_b.genes:
            continue
        a = genome_a.genes[name]
        b = genome_b.genes[name]

        if isinstance(a, Gene) and isinstance(b, Gene):
            gene_range = a.max_value - a.min_value
            if gene_range > 0:
                total_distance += abs(a.value - b.value) / gene_range
            count += 1
        else:
            if a != b:
                total_distance += 1.0
            count += 1

    return total_distance / max(count, 1)
