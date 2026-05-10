"""Matplotlib-based real-time visualization for the evolution simulation."""
from __future__ import annotations
import asyncio
import matplotlib.pyplot as plt
import numpy as np
from simulation.engine import SimulationEngine
from simulation.species import Species
from simulation.environment import PlanetConfig
from simulation.models import Genome
from simulation.templates import load_planet_config, load_species_template
from ai.provider import AIProvider


def run_visual_simulation(
    planet_name: str = "titan",
    n_producers: int = 2,
    n_consumers: int = 1,
    steps: int = 2000,
    grid_size: int = 50,
    ai_provider: AIProvider | None = None,
    ai_interval: int = 60,
):
    """Run simulation with matplotlib 4-panel visualization.

    Args:
        planet_name: Name of the planet template to load.
        n_producers: Number of producer species to seed.
        n_consumers: Number of consumer species to seed.
        steps: Total simulation ticks to run.
        grid_size: Size of the spatial grid.
        ai_provider: Optional AI provider for evolution decisions.
        ai_interval: Ticks between AI evolution calls.
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    config = load_planet_config(planet_name)
    engine = SimulationEngine(
        config=config,
        grid_size=grid_size,
        ai_provider=ai_provider,
        ai_interval=ai_interval,
    )

    # Seed producer species
    for i in range(n_producers):
        genome = load_species_template("producer_photo")
        if i > 0:
            genome = load_species_template("producer_chemo")
        sp = Species.create(f"producer_{i}", f"Producer-{i}", genome, grid_size=grid_size)
        engine.add_species(sp)

    # Seed consumer species
    for i in range(n_consumers):
        genome = load_species_template("consumer_herbivore")
        sp = Species.create(f"consumer_{i}", f"Consumer-{i}", genome, grid_size=grid_size, seed_area="random")
        engine.add_species(sp)

    # Run with asyncio if AI is enabled, otherwise synchronous
    if ai_provider:
        asyncio.run(_run_async(engine, steps, grid_size))
    else:
        _run_sync(engine, steps, grid_size)


def _run_sync(engine: SimulationEngine, steps: int, grid_size: int):
    """Synchronous simulation loop (no AI)."""
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Alien Evolution Simulator — {engine.config.name}", fontsize=14)

    pop_history: dict[str, list[float]] = {sp.id: [] for sp in engine.species_list}
    pop_names: dict[str, str] = {sp.id: sp.name for sp in engine.species_list}
    tick_history: list[int] = []
    diversity_history: list[float] = []

    try:
        for step_i in range(steps):
            engine.step()
            _record_and_draw(engine, step_i, steps, grid_size,
                             pop_history, pop_names, tick_history, diversity_history, fig, axes)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        plt.ioff()
        plt.show(block=True)


async def _run_async(engine: SimulationEngine, steps: int, grid_size: int):
    """Async simulation loop (with AI)."""
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Alien Evolution Simulator — {engine.config.name} [AI]", fontsize=14)

    pop_history: dict[str, list[float]] = {sp.id: [] for sp in engine.species_list}
    pop_names: dict[str, str] = {sp.id: sp.name for sp in engine.species_list}
    tick_history: list[int] = []
    diversity_history: list[float] = []

    try:
        for step_i in range(steps):
            await engine.step_async()

            # Register new species from AI speciation
            for sp in engine.species_list:
                if sp.id not in pop_history:
                    pop_history[sp.id] = []
                    pop_names[sp.id] = sp.name

            _record_and_draw(engine, step_i, steps, grid_size,
                             pop_history, pop_names, tick_history, diversity_history, fig, axes)

            # Print AI narratives
            if engine.ai_narratives and step_i % engine.ai_interval == 0:
                print(f"  AI: {engine.ai_narratives[-1]}")

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        plt.ioff()
        plt.show(block=True)


def _record_and_draw(
    engine: SimulationEngine,
    step_i: int,
    steps: int,
    grid_size: int,
    pop_history: dict,
    pop_names: dict,
    tick_history: list,
    diversity_history: list,
    fig,
    axes,
):
    """Record history and update matplotlib display."""
    tick_history.append(engine.tick)

    for sp in engine.species_list:
        if sp.id in pop_history:
            pop_history[sp.id].append(sp.total_biomass())

    # Shannon diversity
    alive_biomasses = [sp.total_biomass() for sp in engine.species_list if sp.is_alive()]
    if len(alive_biomasses) > 1:
        total = sum(alive_biomasses)
        proportions = [b / total for b in alive_biomasses if b > 0]
        diversity = -sum(p * np.log(p) for p in proportions if p > 0)
    else:
        diversity = 0.0
    diversity_history.append(diversity)

    # Draw every 10 steps
    if step_i % 10 == 0:
        # Panel 1: Combined biomass
        axes[0, 0].clear()
        total = np.zeros((grid_size, grid_size))
        for sp in engine.species_list:
            total += sp.biomass
        axes[0, 0].imshow(total, cmap='YlGn', vmin=0, vmax=5)
        axes[0, 0].set_title(f"Biomass (tick {engine.tick})")

        # Panel 2: Temperature
        axes[0, 1].clear()
        axes[0, 1].imshow(engine.env.temperature, cmap='RdYlBu_r')
        axes[0, 1].set_title(f"Temperature (avg {np.mean(engine.env.temperature):.1f}°C)")

        # Panel 3: Resources
        axes[1, 0].clear()
        axes[1, 0].imshow(engine.env.resources, cmap='Greens', vmin=0, vmax=2)
        axes[1, 0].set_title("Resources")

        # Panel 4: Population
        axes[1, 1].clear()
        for sp_id, history in pop_history.items():
            if len(history) > 0:
                label = pop_names.get(sp_id, sp_id)
                axes[1, 1].plot(tick_history[:len(history)], history, label=label)
        if len(diversity_history) > 0:
            axes[1, 1].plot(tick_history[:len(diversity_history)], diversity_history,
                            'k--', alpha=0.4, label='Diversity')
        axes[1, 1].set_title("Population & Diversity")
        axes[1, 1].set_xlabel("Tick")
        axes[1, 1].set_ylabel("Biomass / Diversity")
        axes[1, 1].legend(fontsize=7, loc='upper left')

        plt.tight_layout()
        plt.pause(0.01)

    # Console output every 100 ticks
    if step_i % 100 == 0:
        alive = [s for s in engine.species_list if s.is_alive()]
        ai_status = f", AI decisions: {len(engine.events)}" if engine.ai_provider else ""
        print(f"Tick {engine.tick}: {len(alive)} species alive, "
              f"env temp={np.mean(engine.env.temperature):.1f}°C, "
              f"O2={engine.env.atmosphere.get('O2', 0):.4f}, "
              f"diversity={diversity:.2f}{ai_status}")
