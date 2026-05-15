"""Core simulation engine with rich species-environment interactions."""
from __future__ import annotations
import asyncio
import numpy as np
from scipy.signal import convolve2d
from simulation.environment import Environment, PlanetConfig
from simulation.species import Species, EvolutionEvent
from simulation.food_web import FoodWeb
from simulation.models import Gene, MetabolicType
from simulation.genetics import mutate_genome
from ai.provider import AIProvider, EcosystemSnapshot


class SimulationEngine:
    """Core simulation loop with 7 interaction mechanisms:

    1. Decomposition: dead biomass → resource pool
    2. Atmosphere feedback: O2/CO2 levels affect fitness
    3. Thermal biology: dense biomass generates local heat
    4. Toxin accumulation: heterotroph waste harms others
    5. Carrying capacity: logistic growth limits
    6. Mutualism: producer-consumer nutrient cycling
    7. Light competition: large organisms shade small ones
    """

    def __init__(
        self,
        config: PlanetConfig,
        grid_size: int = 50,
        rng: np.random.RandomState | None = None,
        ai_provider: AIProvider | None = None,
        ai_interval: int = 60,
    ):
        self.config = config
        self.grid_size = grid_size
        self.rng = rng if rng is not None else np.random.RandomState()
        self.env = Environment(config, size=grid_size, rng=self.rng)
        self.food_web = FoodWeb()
        self.species_list: list[Species] = []
        self.tick = 0
        self.events: list[EvolutionEvent] = []
        self.ai_narratives: list[str] = []

        # AI integration
        self.ai_provider = ai_provider
        self.ai_interval = ai_interval
        self._species_counter = 0

        # New interaction layers
        self.dead_biomass = np.zeros((grid_size, grid_size))  # Decomposition pool
        self.toxins = np.zeros((grid_size, grid_size))        # Toxin accumulation
        self.nutrients = np.ones((grid_size, grid_size)) * 0.5  # Nutrient cycling

        self.diffusion_kernel = np.array([
            [0.05, 0.1, 0.05],
            [0.1, 0.4, 0.1],
            [0.05, 0.1, 0.05],
        ])

    def _next_species_id(self) -> str:
        self._species_counter += 1
        return f"sp_{self._species_counter:03d}"

    def add_species(self, species: Species):
        self.species_list.append(species)
        self.food_web.register(species)

    def step(self):
        self.tick += 1

        # ── 1. Decomposition: dead_biomass → resources ─────────
        decay_rate = 0.05
        decomposed = self.dead_biomass * decay_rate
        self.env.resources = np.clip(self.env.resources + decomposed * 0.8, 0, 3)
        self.nutrients = np.clip(self.nutrients + decomposed * 0.2, 0, 2)
        self.dead_biomass -= decomposed
        self.dead_biomass = np.clip(self.dead_biomass, 0, 100)

        # ── 2. Toxin decay ─────────────────────────────────────
        self.toxins *= 0.98  # Slow natural decay

        # ── 3. Compute total biomass for interactions ──────────
        total_biomass = np.zeros((self.grid_size, self.grid_size))
        producer_biomass = np.zeros((self.grid_size, self.grid_size))
        consumer_biomass = np.zeros((self.grid_size, self.grid_size))
        max_body_size = np.zeros((self.grid_size, self.grid_size))

        for sp in self.species_list:
            if not sp.is_alive():
                continue
            total_biomass += sp.biomass
            meta = sp.genome.get_enum("metabolic_type")
            bs = sp.genome.get_float("body_size")
            if meta in ("photosynthesis", "chemosynthesis"):
                producer_biomass += sp.biomass
            else:
                consumer_biomass += sp.biomass
            max_body_size = np.maximum(max_body_size, sp.biomass * bs)

        # ── 4. Thermal biology: biomass generates heat ─────────
        biomass_heat = total_biomass * 0.02 + consumer_biomass * 0.01

        # ── 5. Environment update ──────────────────────────────
        self.env.update(ticks=1, biomass_heat=biomass_heat)

        # ── 6. Process each species ────────────────────────────
        dead_species = []
        for sp in self.species_list:
            if not sp.is_alive():
                dead_species.append(sp)
                continue
            self._process_species(sp, total_biomass, producer_biomass, consumer_biomass, max_body_size)

        # ── 7. Handle extinctions → dead_biomass pool ──────────
        for sp in dead_species:
            sp.extinction_tick = self.tick
            sp.history.append(EvolutionEvent(
                tick=self.tick, event_type="extinction",
                description=f"{sp.name} 在第 {self.tick} 周期灭绝",
            ))
            self.events.append(sp.history[-1])
            # Dead biomass goes to decomposition pool
            self.dead_biomass += sp.biomass * sp.genome.get_float("body_size")
            self.food_web.unregister(sp.id)

        self.species_list = [s for s in self.species_list if s.is_alive()]

        # ── 8. Nutrient & toxin diffusion ──────────────────────
        self.nutrients = convolve2d(self.nutrients, self.diffusion_kernel, mode='same')
        self.toxins = convolve2d(self.toxins, self.diffusion_kernel, mode='same') * 0.95

    def _process_species(self, sp: Species, total_biomass, producer_biomass, consumer_biomass, max_body_size):
        meta_type = sp.genome.get_enum("metabolic_type")
        body_size = sp.genome.get_float("body_size")
        rep_rate = sp.genome.get_float("reproduction_rate")
        rep_cost = sp.genome.get_float("reproduction_cost")
        mobility = sp.genome.get_float("mobility")
        lifespan = sp.genome.get_float("lifespan")
        temp_opt = sp.genome.get_float("temp_optimum")
        temp_tol = sp.genome.get_float("temp_tolerance")
        defense = sp.genome.get_float("defense")
        adaptability = sp.genome.get_float("adaptability")

        G = self.grid_size

        # ── A. Temperature fitness ─────────────────────────────
        temp_dist = np.abs(self.env.temperature - temp_opt)
        temp_fitness = np.exp(-(temp_dist ** 2) / (2 * temp_tol ** 2))

        # ── B. Resource & light fitness ────────────────────────
        if meta_type == "photosynthesis":
            # Light competition: large organisms block light
            light_blocked = np.clip(max_body_size / (max_body_size + body_size + 0.1), 0, 0.8)
            effective_light = self.env.light * (1 - light_blocked * 0.5)
            resource_fitness = np.clip(effective_light * 2.0, 0, 2) * np.clip(self.env.resources, 0, 2)
            # Nutrient boost from consumers (mutualism)
            nutrient_boost = 1.0 + self.nutrients * 0.3
            resource_fitness *= nutrient_boost
        elif meta_type == "chemosynthesis":
            resource_fitness = np.clip(self.env.resources * 1.5, 0, 2)
        else:
            resource_fitness = np.clip(self.env.resources, 0, 2)

        # ── C. Atmosphere fitness ──────────────────────────────
        o2 = self.env.atmosphere.get("O2", 0)
        co2 = self.env.atmosphere.get("CO2", 0)
        if meta_type == "heterotrophy":
            # Heterotrophs need O2, stressed by high CO2
            o2_fitness = np.clip(o2 * 10, 0.1, 1.5)  # More O2 = better
            co2_stress = np.clip(co2 * 5, 0, 0.5)     # High CO2 = stress
            atm_fitness = o2_fitness - co2_stress
        elif meta_type == "photosynthesis":
            # Producers use CO2, benefit from higher CO2
            atm_fitness = 1.0 + np.clip(co2 * 3, 0, 0.5)
        else:
            atm_fitness = 1.0

        atm_fitness = np.clip(atm_fitness, 0.2, 2.0)

        # ── D. Toxin stress ────────────────────────────────────
        toxin_resistance = adaptability * 0.5 + defense * 0.5
        toxin_stress = np.clip(self.toxins * (1 - toxin_resistance), 0, 0.8)

        # ── E. Competition & carrying capacity ─────────────────
        competition = 1.0 / (1.0 + total_biomass * 0.08)
        # Carrying capacity: logistic limit based on resources
        carrying = np.clip(1.0 - sp.biomass * 0.02, 0, 1)

        # ── F. Combined fitness ────────────────────────────────
        fitness = temp_fitness * resource_fitness * atm_fitness * competition * carrying
        fitness = fitness * (1 - toxin_stress)
        fitness = np.clip(fitness, 0, 3)

        # ── G. Births ─────────────────────────────────────────
        births = sp.biomass * rep_rate * fitness * 0.1
        resource_cost = births * rep_cost * body_size * 0.08
        self.env.resources = np.clip(self.env.resources - resource_cost, 0, 3)

        # ── H. Deaths ─────────────────────────────────────────
        base_mortality = 1.0 / max(lifespan, 1)
        env_stress = np.clip(1.0 - temp_fitness, 0, 1) * 0.04
        starvation = np.clip(1.0 - resource_fitness, 0, 1) * 0.02
        atm_stress = np.clip(1.0 - atm_fitness, 0, 1) * 0.02
        deaths = sp.biomass * (base_mortality + env_stress + starvation + atm_stress)

        # ── I. Predation ───────────────────────────────────────
        predation_rates = self.food_web.compute_predation_rates()
        for (pred_id, prey_id), rate in predation_rates.items():
            if prey_id == sp.id:
                predator = self._find_species(pred_id)
                if predator is not None:
                    consumed = sp.biomass * rate * predator.biomass * 0.05
                    consumed = np.minimum(consumed, sp.biomass)
                    sp.biomass -= consumed
                    predator.biomass += consumed * 0.5

        # ── J. Update biomass ──────────────────────────────────
        sp.biomass = sp.biomass + births - deaths
        sp.biomass = np.clip(sp.biomass, 0, 50)

        # ── K. Dead biomass from natural deaths ────────────────
        self.dead_biomass += deaths * body_size * 0.5

        # ── L. Diffusion ───────────────────────────────────────
        if mobility > 0.01:
            diffusion_strength = mobility * 0.3
            diffused = convolve2d(sp.biomass, self.diffusion_kernel, mode='same')
            sp.biomass = sp.biomass * (1 - diffusion_strength) + diffused * diffusion_strength

        # ── M. Atmosphere feedback ─────────────────────────────
        biomass_sum = np.sum(sp.biomass)
        if meta_type == "photosynthesis":
            rate = biomass_sum * 0.00015
            self.env.atmosphere["O2"] = min(0.5, self.env.atmosphere.get("O2", 0) + rate)
            self.env.atmosphere["CO2"] = max(0, self.env.atmosphere.get("CO2", 0) - rate)
        elif meta_type == "chemosynthesis":
            rate = biomass_sum * 0.00008
            self.env.atmosphere["CH4"] = min(0.3, self.env.atmosphere.get("CH4", 0) + rate)
        elif meta_type == "heterotrophy":
            o2_use = biomass_sum * 0.00008
            co2_out = biomass_sum * 0.00008
            self.env.atmosphere["O2"] = max(0, self.env.atmosphere.get("O2", 0) - o2_use)
            self.env.atmosphere["CO2"] = min(1.0, self.env.atmosphere.get("CO2", 0) + co2_out)
            # Toxin production from heterotroph waste
            self.toxins += sp.biomass * 0.001 * body_size

    def get_snapshot(self) -> dict:
        return {
            "tick": self.tick,
            "environment": self.env.get_snapshot(),
            "species": [
                {
                    "id": sp.id, "name": sp.name,
                    "total_biomass": sp.total_biomass(),
                    "is_alive": sp.is_alive(),
                    "metabolic_type": sp.genome.get_enum("metabolic_type"),
                    "color": sp.color,
                }
                for sp in self.species_list
            ],
            "events": [
                {"tick": e.tick, "type": e.event_type, "description": e.description}
                for e in self.events[-10:]
            ],
            "ai_narratives": self.ai_narratives[-5:],
        }

    def _find_species(self, species_id: str) -> Species | None:
        for sp in self.species_list:
            if sp.id == species_id:
                return sp
        return None

    # ── AI integration ─────────────────────────────────────────

    async def step_async(self):
        self.step()
        if self.ai_provider and self.ai_interval > 0 and self.tick % self.ai_interval == 0:
            await self._run_ai_evolution()

    async def _run_ai_evolution(self):
        snapshot = self._build_snapshot()
        try:
            decision = await self.ai_provider.analyze_ecosystem(snapshot)
        except Exception as e:
            self.events.append(EvolutionEvent(
                tick=self.tick, event_type="ai_error", description=f"AI error: {e}",
            ))
            return
        for mut in decision.mutations:
            target = self._find_species(mut.species_id)
            if target:
                self._apply_mutation(target, mut.gene, mut.delta, mut.reason)
        for spec in decision.speciations:
            parent = self._find_species(spec.parent_species_id)
            if parent:
                self._apply_speciation(parent, spec)
        if decision.narrative:
            self.ai_narratives.append(f"[tick {self.tick}] {decision.narrative}")
            self.events.append(EvolutionEvent(
                tick=self.tick, event_type="ai_narrative", description=decision.narrative,
            ))

    def _build_snapshot(self) -> EcosystemSnapshot:
        species_data = []
        for sp in self.species_list:
            if not sp.is_alive():
                continue
            temp_opt = sp.genome.get_float("temp_optimum")
            temp_tol = sp.genome.get_float("temp_tolerance")
            temp_fitness = np.exp(-((self.env.temperature - temp_opt) ** 2) / (2 * temp_tol ** 2))
            genes = {k: round(v.value, 3) if hasattr(v, "value") else v for k, v in sp.genome.genes.items()}
            species_data.append({
                "id": sp.id, "name": sp.name,
                "total_biomass": round(sp.total_biomass(), 3),
                "avg_fitness": round(float(np.mean(temp_fitness)), 4),
                "genes": genes, "ancestor_id": sp.ancestor_id,
            })
        return EcosystemSnapshot(
            tick=self.tick, planet_name=self.config.name,
            planet_config={"gravity": self.config.gravity, "surface_temp": self.config.surface_temp},
            environment=self.env.get_snapshot(), species=species_data,
            recent_events=[{"tick": e.tick, "type": e.event_type, "description": e.description} for e in self.events[-10:]],
        )

    def _apply_mutation(self, species: Species, gene_name: str, delta: float, reason: str):
        if gene_name not in species.genome.genes:
            return
        gene = species.genome.genes[gene_name]
        if not isinstance(gene, Gene):
            return
        old_val = gene.value
        new_val = max(gene.min_value, min(gene.max_value, old_val + delta))
        gene.value = new_val
        event = EvolutionEvent(
            tick=self.tick, event_type="mutation",
            description=f"{species.name}: {gene_name} {old_val:.2f} → {new_val:.2f} ({reason})",
            details={"gene": gene_name, "old": old_val, "new": new_val, "delta": delta},
        )
        species.history.append(event)
        self.events.append(event)

    def _apply_speciation(self, parent: Species, spec):
        new_genome = mutate_genome(parent.genome, mutation_strength=0.1, parent_id=parent.id, rng=self.rng)
        new_genome.generation = parent.genome.generation + 1
        for gene_name, value in spec.genome_overrides.items():
            if gene_name in new_genome.genes:
                existing = new_genome.genes[gene_name]
                if isinstance(existing, Gene):
                    new_genome.genes[gene_name] = Gene(
                        value=float(value), min_value=existing.min_value, max_value=existing.max_value,
                        mutation_rate=existing.mutation_rate, dominance=existing.dominance,
                    )
                else:
                    new_genome.genes[gene_name] = value
        new_id = self._next_species_id()
        new_species = Species.create(
            species_id=new_id, name=spec.new_species_name, genome=new_genome,
            grid_size=self.grid_size, initial_biomass=0.2, seed_area=spec.seed_area, rng=self.rng,
        )
        new_species.ancestor_id = parent.id
        self.add_species(new_species)
        event = EvolutionEvent(
            tick=self.tick, event_type="speciation",
            description=f"{spec.new_species_name} 从 {parent.name} 分化 ({spec.reason})",
            details={"parent_id": parent.id, "new_id": new_id},
        )
        parent.history.append(event)
        new_species.history.append(event)
        self.events.append(event)

    def get_snapshot_legacy(self) -> dict:
        """Legacy snapshot for backwards compat."""
        return self.get_snapshot()
