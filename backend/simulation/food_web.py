from __future__ import annotations
import numpy as np
from simulation.models import MetabolicType, DietPreference, Genome
from simulation.species import Species


class FoodWeb:
    """Emergent food web based on species genetics."""

    def __init__(self):
        self.species_map: dict[str, Species] = {}

    def register(self, species: Species):
        self.species_map[species.id] = species

    def unregister(self, species_id: str):
        self.species_map.pop(species_id, None)

    def get_prey(self, predator_id: str) -> list[str]:
        predator = self.species_map.get(predator_id)
        if predator is None:
            return []

        meta_type = predator.genome.get_enum("metabolic_type")
        diet = predator.genome.get_enum("diet_preference")

        if meta_type in (MetabolicType.PHOTOSYNTHESIS.value, MetabolicType.CHEMOSYNTHESIS.value):
            return []

        prey_ids = []
        for sid, sp in self.species_map.items():
            if sid == predator_id:
                continue
            if not sp.is_alive():
                continue

            sp_meta = sp.genome.get_enum("metabolic_type")

            if diet == DietPreference.CONSUMER.value:
                if sp_meta in (MetabolicType.PHOTOSYNTHESIS.value, MetabolicType.CHEMOSYNTHESIS.value):
                    prey_ids.append(sid)
            elif diet == DietPreference.OMNI.value:
                prey_ids.append(sid)

        return prey_ids

    def _calc_attack_power(self, genome: Genome) -> float:
        body = genome.get_float("body_size")
        mob = genome.get_float("mobility")
        sense = genome.get_float("sensory_range")
        return body * mob * sense

    @staticmethod
    def predation_efficiency(attack_power: float, prey_defense: float) -> float:
        if attack_power + prey_defense == 0:
            return 0.0
        return attack_power / (attack_power + prey_defense)

    def compute_predation_rates(self) -> dict[tuple[str, str], float]:
        rates = {}
        for pred_id, predator in self.species_map.items():
            if not predator.is_alive():
                continue
            prey_ids = self.get_prey(pred_id)
            attack = self._calc_attack_power(predator.genome)
            for prey_id in prey_ids:
                prey = self.species_map[prey_id]
                defense = prey.genome.get_float("defense")
                eff = self.predation_efficiency(attack, defense)
                rates[(pred_id, prey_id)] = eff
        return rates

    def get_producers(self) -> list[str]:
        return [
            sid for sid, sp in self.species_map.items()
            if sp.genome.get_enum("metabolic_type") in (MetabolicType.PHOTOSYNTHESIS.value, MetabolicType.CHEMOSYNTHESIS.value)
            and sp.is_alive()
        ]

    def get_consumers(self) -> list[str]:
        return [
            sid for sid, sp in self.species_map.items()
            if sp.genome.get_enum("metabolic_type") == MetabolicType.HETEROTROPHY.value
            and sp.is_alive()
        ]

    def get_trophic_level(self, species_id: str) -> float:
        sp = self.species_map.get(species_id)
        if sp is None:
            return 0.0
        meta = sp.genome.get_enum("metabolic_type")
        if meta in (MetabolicType.PHOTOSYNTHESIS.value, MetabolicType.CHEMOSYNTHESIS.value):
            return 1.0
        diet = sp.genome.get_enum("diet_preference")
        if diet == DietPreference.CONSUMER.value:
            return 2.0
        if diet == DietPreference.OMNI.value:
            return 2.5
        return 2.0
