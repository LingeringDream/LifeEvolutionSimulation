"""Abstract AI provider interface for evolution decisions."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class EcosystemSnapshot:
    """Serializable snapshot of the ecosystem state, sent to the AI."""
    tick: int
    planet_name: str
    planet_config: dict
    environment: dict  # temperature stats, atmosphere, resources
    species: list[dict]  # each species' id, name, biomass, fitness, genes
    recent_events: list[dict]  # recent mutations, extinctions, disasters


@dataclass
class MutationDecision:
    """A single gene mutation recommended by the AI."""
    species_id: str
    gene: str
    delta: float
    reason: str


@dataclass
class SpeciationDecision:
    """A new species to branch off from an existing one."""
    parent_species_id: str
    new_species_name: str
    genome_overrides: dict[str, float | str]
    seed_area: str = "random"
    reason: str = ""


@dataclass
class EvolutionDecision:
    """Complete AI evolution decision for one cycle."""
    mutations: list[MutationDecision] = field(default_factory=list)
    speciations: list[SpeciationDecision] = field(default_factory=list)
    narrative: str = ""
    raw_response: str = ""


class AIProvider(ABC):
    """Abstract base class for AI evolution providers.

    Implementations call an LLM to analyze the ecosystem and return
    structured evolution decisions.
    """

    @abstractmethod
    async def analyze_ecosystem(self, snapshot: EcosystemSnapshot) -> EvolutionDecision:
        """Analyze ecosystem state and return evolution decisions.

        Args:
            snapshot: Current state of the ecosystem.

        Returns:
            EvolutionDecision with mutations, speciations, and narrative.
        """
        ...

    @abstractmethod
    async def generate_narrative(self, event: str, context: dict) -> str:
        """Generate an educational narrative for an evolution event.

        Args:
            event: Event type (e.g., "mass_extinction", "speciation").
            context: Additional context about the event.

        Returns:
            Human-readable narrative string.
        """
        ...

    @abstractmethod
    async def name_species(self, traits: dict) -> str:
        """Generate a name for a new species based on its traits.

        Args:
            traits: Dictionary of species traits (metabolic_type, temp_optimum, etc.)

        Returns:
            A species name string.
        """
        ...
