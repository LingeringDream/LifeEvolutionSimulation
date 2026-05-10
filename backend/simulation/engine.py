"""Core simulation engine with optional AI-driven evolution."""
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
    """Core simulation loop: environment <-> species interactions.

    Supports optional AI-driven evolution via an AIProvider.
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
        """Execute one simulation tick (synchronous)."""
        self.tick += 1

        # 1. Environment update
        biomass_heat = np.zeros((self.grid_size, self.grid_size))
        for sp in self.species_list:
            if sp.is_alive():
                biomass_heat += sp.biomass * 0.01
        self.env.update(ticks=1, biomass_heat=biomass_heat)

        # 2. Process species dynamics
        dead_species = []
        for sp in self.species_list:
            if not sp.is_alive():
                dead_species.append(sp)
                continue
            self._process_species(sp)

        # 3. Handle extinctions
        for sp in dead_species:
            sp.extinction_tick = self.tick
            sp.history.append(EvolutionEvent(
                tick=self.tick,
                event_type="extinction",
                description=f"{sp.name} went extinct at tick {self.tick}",
            ))
            self.events.append(sp.history[-1])
            self.food_web.unregister(sp.id)

        self.species_list = [s for s in self.species_list if s.is_alive()]

    async def step_async(self):
        """Execute one tick, with AI evolution decisions at configured intervals."""
        self.step()

        # AI evolution check
        if self.ai_provider and self.ai_interval > 0 and self.tick % self.ai_interval == 0:
            await self._run_ai_evolution()

    async def _run_ai_evolution(self):
        """Call the AI provider to analyze the ecosystem and apply decisions."""
        snapshot = self._build_snapshot()

        try:
            decision = await self.ai_provider.analyze_ecosystem(snapshot)
        except Exception as e:
            self.events.append(EvolutionEvent(
                tick=self.tick,
                event_type="ai_error",
                description=f"AI call failed: {e}",
            ))
            return

        # Apply mutations
        for mut in decision.mutations:
            target = self._find_species(mut.species_id)
            if target is None:
                continue
            self._apply_mutation(target, mut.gene, mut.delta, mut.reason)

        # Apply speciations
        for spec in decision.speciations:
            parent = self._find_species(spec.parent_species_id)
            if parent is None:
                continue
            self._apply_speciation(parent, spec)

        # Record narrative
        if decision.narrative:
            self.ai_narratives.append(f"[tick {self.tick}] {decision.narrative}")
            self.events.append(EvolutionEvent(
                tick=self.tick,
                event_type="ai_narrative",
                description=decision.narrative,
            ))

    def _build_snapshot(self) -> EcosystemSnapshot:
        """Build a serializable snapshot for the AI provider."""
        species_data = []
        for sp in self.species_list:
            if not sp.is_alive():
                continue
            # Calculate average fitness
            temp_opt = sp.genome.get_float("temp_optimum")
            temp_tol = sp.genome.get_float("temp_tolerance")
            temp_dist = np.abs(self.env.temperature - temp_opt)
            temp_fitness = np.exp(-(temp_dist ** 2) / (2 * temp_tol ** 2))
            avg_fitness = float(np.mean(temp_fitness))

            genes = {}
            for name, gene in sp.genome.genes.items():
                if isinstance(gene, Gene):
                    genes[name] = round(gene.value, 3)
                else:
                    genes[name] = gene

            species_data.append({
                "id": sp.id,
                "name": sp.name,
                "total_biomass": round(sp.total_biomass(), 3),
                "avg_fitness": round(avg_fitness, 4),
                "genes": genes,
                "ancestor_id": sp.ancestor_id,
            })

        return EcosystemSnapshot(
            tick=self.tick,
            planet_name=self.config.name,
            planet_config={
                "gravity": self.config.gravity,
                "surface_temp": self.config.surface_temp,
                "albedo": self.config.albedo,
                "axial_tilt": self.config.axial_tilt,
                "orbital_distance": self.config.orbital_distance,
            },
            environment=self.env.get_snapshot(),
            species=species_data,
            recent_events=[
                {"tick": e.tick, "type": e.event_type, "description": e.description}
                for e in self.events[-10:]
            ],
        )

    def _find_species(self, species_id: str) -> Species | None:
        for sp in self.species_list:
            if sp.id == species_id:
                return sp
        return None

    def _apply_mutation(self, species: Species, gene_name: str, delta: float, reason: str):
        """Apply a single gene mutation to a species."""
        if gene_name not in species.genome.genes:
            return
        gene = species.genome.genes[gene_name]
        if not isinstance(gene, Gene):
            # Enum gene — skip for now (could handle metabolic_type changes)
            return

        old_val = gene.value
        new_val = max(gene.min_value, min(gene.max_value, old_val + delta))
        gene.value = new_val

        event = EvolutionEvent(
            tick=self.tick,
            event_type="mutation",
            description=f"{species.name}: {gene_name} {old_val:.2f} → {new_val:.2f} ({reason})",
            details={"gene": gene_name, "old": old_val, "new": new_val, "delta": delta},
        )
        species.history.append(event)
        self.events.append(event)

    def _apply_speciation(self, parent: Species, spec):
        """Create a new species branching from a parent."""
        new_genome = mutate_genome(parent.genome, mutation_strength=0.1, parent_id=parent.id, rng=self.rng)
        new_genome.generation = parent.genome.generation + 1

        # Apply genome overrides from AI
        for gene_name, value in spec.genome_overrides.items():
            if gene_name in new_genome.genes:
                existing = new_genome.genes[gene_name]
                if isinstance(existing, Gene):
                    new_genome.genes[gene_name] = Gene(
                        value=float(value),
                        min_value=existing.min_value,
                        max_value=existing.max_value,
                        mutation_rate=existing.mutation_rate,
                        dominance=existing.dominance,
                    )
                else:
                    new_genome.genes[gene_name] = value

        new_id = self._next_species_id()
        new_name = spec.new_species_name

        new_species = Species.create(
            species_id=new_id,
            name=new_name,
            genome=new_genome,
            grid_size=self.grid_size,
            initial_biomass=0.2,
            seed_area=spec.seed_area,
            rng=self.rng,
        )
        new_species.ancestor_id = parent.id

        self.add_species(new_species)

        event = EvolutionEvent(
            tick=self.tick,
            event_type="speciation",
            description=f"{new_name} branched from {parent.name} ({spec.reason})",
            details={"parent_id": parent.id, "new_id": new_id},
        )
        parent.history.append(event)
        new_species.history.append(event)
        self.events.append(event)

    def _process_species(self, sp: Species):
        meta_type = sp.genome.get_enum("metabolic_type")
        body_size = sp.genome.get_float("body_size")
        rep_rate = sp.genome.get_float("reproduction_rate")
        rep_cost = sp.genome.get_float("reproduction_cost")
        mobility = sp.genome.get_float("mobility")
        lifespan = sp.genome.get_float("lifespan")
        temp_opt = sp.genome.get_float("temp_optimum")
        temp_tol = sp.genome.get_float("temp_tolerance")

        # A. Temperature fitness
        temp_dist = np.abs(self.env.temperature - temp_opt)
        temp_fitness = np.exp(-(temp_dist ** 2) / (2 * temp_tol ** 2))

        # B. Resource fitness
        if meta_type in ("photosynthesis", "chemosynthesis"):
            resource_fitness = np.clip(self.env.light * 2.0, 0, 2) * np.clip(self.env.resources, 0, 2)
        else:
            resource_fitness = np.clip(self.env.resources, 0, 2)

        # C. Competition
        total_biomass_here = np.zeros((self.grid_size, self.grid_size))
        for other in self.species_list:
            if other.id != sp.id and other.is_alive():
                total_biomass_here += other.biomass
        competition = 1.0 / (1.0 + total_biomass_here * 0.1)

        # D. Combined fitness
        fitness = temp_fitness * resource_fitness * competition
        fitness = np.clip(fitness, 0, 3)

        # E. Births
        births = sp.biomass * rep_rate * fitness * 0.1
        resource_cost = births * rep_cost * body_size * 0.1
        self.env.resources = np.clip(self.env.resources - resource_cost, 0, 2)

        # F. Deaths
        base_mortality = 1.0 / max(lifespan, 1)
        env_stress = np.clip(1.0 - temp_fitness, 0, 1) * 0.05
        starvation = np.clip(1.0 - resource_fitness, 0, 1) * 0.02
        deaths = sp.biomass * (base_mortality + env_stress + starvation)

        # G. Predation
        predation_rates = self.food_web.compute_predation_rates()
        for (pred_id, prey_id), rate in predation_rates.items():
            if prey_id == sp.id:
                predator = self._find_species(pred_id)
                if predator is not None:
                    consumed = sp.biomass * rate * predator.biomass * 0.05
                    consumed = np.minimum(consumed, sp.biomass)
                    sp.biomass -= consumed
                    predator.biomass += consumed * 0.5

        # H. Update biomass
        sp.biomass = sp.biomass + births - deaths
        sp.biomass = np.clip(sp.biomass, 0, 50)

        # I. Diffusion
        if mobility > 0.01:
            diffusion_strength = mobility * 0.3
            diffused = convolve2d(sp.biomass, self.diffusion_kernel, mode='same')
            sp.biomass = sp.biomass * (1 - diffusion_strength) + diffused * diffusion_strength

        # J. Atmosphere feedback
        if meta_type == "photosynthesis":
            o2_production = np.sum(sp.biomass) * 0.0001
            co2_consumption = np.sum(sp.biomass) * 0.0001
            self.env.atmosphere["O2"] = min(0.5, self.env.atmosphere.get("O2", 0) + o2_production)
            self.env.atmosphere["CO2"] = max(0, self.env.atmosphere.get("CO2", 0) - co2_consumption)

        if meta_type == "heterotrophy":
            o2_consumption = np.sum(sp.biomass) * 0.00005
            co2_production = np.sum(sp.biomass) * 0.00005
            self.env.atmosphere["O2"] = max(0, self.env.atmosphere.get("O2", 0) - o2_consumption)
            self.env.atmosphere["CO2"] = min(1.0, self.env.atmosphere.get("CO2", 0) + co2_production)

    def get_snapshot(self) -> dict:
        return {
            "tick": self.tick,
            "environment": self.env.get_snapshot(),
            "species": [
                {
                    "id": sp.id,
                    "name": sp.name,
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
