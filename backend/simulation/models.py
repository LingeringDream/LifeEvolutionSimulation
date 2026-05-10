from __future__ import annotations
from enum import Enum
from typing import Optional
import numpy as np
from pydantic import BaseModel, Field, model_validator


class MetabolicType(str, Enum):
    PHOTOSYNTHESIS = "photosynthesis"
    CHEMOSYNTHESIS = "chemosynthesis"
    HETEROTROPHY = "heterotrophy"


class DietPreference(str, Enum):
    PRODUCER = "producer"
    CONSUMER = "consumer"
    OMNI = "omni"


class Gene(BaseModel):
    value: float
    min_value: float
    max_value: float
    mutation_rate: float = 0.05
    dominance: float = 0.5

    @model_validator(mode="after")
    def clamp_value(self) -> Gene:
        self.value = max(self.min_value, min(self.max_value, self.value))
        return self


class Genome(BaseModel):
    genes: dict[str, Gene | str]
    generation: int = 0
    parent_ids: list[str] = Field(default_factory=list)

    def get_float(self, name: str) -> float:
        gene = self.genes[name]
        if isinstance(gene, Gene):
            return gene.value
        raise TypeError(f"Gene '{name}' is not a float gene")

    def get_enum(self, name: str) -> str:
        gene = self.genes[name]
        if isinstance(gene, str):
            return gene
        raise TypeError(f"Gene '{name}' is not an enum gene")

    @staticmethod
    def create_default() -> Genome:
        return Genome(
            genes={
                "body_size": Gene(value=1.0, min_value=0.1, max_value=10.0, mutation_rate=0.03),
                "metabolic_type": MetabolicType.PHOTOSYNTHESIS.value,
                "temp_optimum": Gene(value=20.0, min_value=-250.0, max_value=500.0, mutation_rate=0.02),
                "temp_tolerance": Gene(value=30.0, min_value=5.0, max_value=150.0, mutation_rate=0.02),
                "reproduction_rate": Gene(value=0.5, min_value=0.01, max_value=3.0, mutation_rate=0.03),
                "reproduction_cost": Gene(value=1.0, min_value=0.1, max_value=5.0, mutation_rate=0.02),
                "defense": Gene(value=0.1, min_value=0.0, max_value=1.0, mutation_rate=0.04),
                "mobility": Gene(value=0.3, min_value=0.0, max_value=1.0, mutation_rate=0.03),
                "sensory_range": Gene(value=3.0, min_value=1.0, max_value=15.0, mutation_rate=0.03),
                "diet_preference": DietPreference.PRODUCER.value,
                "lifespan": Gene(value=500.0, min_value=10.0, max_value=2000.0, mutation_rate=0.02),
                "adaptability": Gene(value=0.3, min_value=0.0, max_value=1.0, mutation_rate=0.04),
            }
        )

    @staticmethod
    def create_consumer() -> Genome:
        genome = Genome.create_default()
        genome.genes["metabolic_type"] = MetabolicType.HETEROTROPHY.value
        genome.genes["diet_preference"] = DietPreference.CONSUMER.value
        genome.genes["body_size"] = Gene(value=2.0, min_value=0.1, max_value=10.0, mutation_rate=0.03)
        genome.genes["mobility"] = Gene(value=0.6, min_value=0.0, max_value=1.0, mutation_rate=0.03)
        genome.genes["defense"] = Gene(value=0.2, min_value=0.0, max_value=1.0, mutation_rate=0.04)
        return genome

    @staticmethod
    def create_decomposer() -> Genome:
        genome = Genome.create_default()
        genome.genes["metabolic_type"] = MetabolicType.HETEROTROPHY.value
        genome.genes["diet_preference"] = DietPreference.CONSUMER.value
        genome.genes["body_size"] = Gene(value=0.2, min_value=0.1, max_value=10.0, mutation_rate=0.03)
        genome.genes["mobility"] = Gene(value=0.1, min_value=0.0, max_value=1.0, mutation_rate=0.03)
        genome.genes["reproduction_rate"] = Gene(value=1.5, min_value=0.01, max_value=3.0, mutation_rate=0.03)
        return genome
