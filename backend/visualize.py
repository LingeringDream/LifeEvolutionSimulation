"""Matplotlib-based real-time visualization for the evolution simulation."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from simulation.engine import SimulationEngine
from simulation.species import Species
from simulation.environment import PlanetConfig
from simulation.models import Genome
from simulation.templates import load_planet_config, load_species_template


def run_visual_simulation(
    planet_name: str = "titan",
    n_producers: int = 2,
    n_consumers: int = 1,
    steps: int = 2000,
    grid_size: int = 50,
):
    """Run simulation with matplotlib 4-panel visualization.

    Args:
        planet_name: Name of the planet template to load.
        n_producers: Number of producer species to seed.
        n_consumers: Number of consumer species to seed.
        steps: Total simulation ticks to run.
        grid_size: Size of the spatial grid.
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    config = load_planet_config(planet_name)
    engine = SimulationEngine(config=config, grid_size=grid_size)

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

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Alien Evolution Simulator — {config.name}", fontsize=14)

    # History tracking
    pop_history: dict[str, list[float]] = {sp.id: [] for sp in engine.species_list}
    pop_names: dict[str, str] = {sp.id: sp.name for sp in engine.species_list}
    tick_history: list[int] = []
    diversity_history: list[float] = []

    try:
        for step_i in range(steps):
            engine.step()

            # Record history
            tick_history.append(engine.tick)
            for sp in engine.species_list:
                if sp.id in pop_history:
                    pop_history[sp.id].append(sp.total_biomass())

            # Shannon diversity index
            alive_biomasses = [sp.total_biomass() for sp in engine.species_list if sp.is_alive()]
            if len(alive_biomasses) > 1:
                total = sum(alive_biomasses)
                proportions = [b / total for b in alive_biomasses if b > 0]
                diversity = -sum(p * np.log(p) for p in proportions if p > 0)
            else:
                diversity = 0.0
            diversity_history.append(diversity)

            # Update display every 10 steps
            if step_i % 10 == 0:
                # Panel 1: Combined biomass heatmap
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

                # Panel 4: Population over time
                axes[1, 1].clear()
                for sp_id, history in pop_history.items():
                    if len(history) > 0:
                        label = pop_names.get(sp_id, sp_id)
                        axes[1, 1].plot(tick_history[:len(history)], history, label=label)
                # Add diversity as secondary axis
                ax2 = axes[1, 1].twinx()
                ax2.plot(tick_history[:len(diversity_history)], diversity_history,
                         'k--', alpha=0.3, label='Diversity')
                ax2.set_ylabel('Diversity', alpha=0.5)
                axes[1, 1].set_title("Population")
                axes[1, 1].set_xlabel("Tick")
                axes[1, 1].set_ylabel("Total Biomass")
                axes[1, 1].legend(fontsize=7, loc='upper left')

                plt.tight_layout()
                plt.pause(0.01)

            # Console output every 100 ticks
            if step_i % 100 == 0:
                alive = [s for s in engine.species_list if s.is_alive()]
                print(f"Tick {engine.tick}: {len(alive)} species alive, "
                      f"env temp={np.mean(engine.env.temperature):.1f}°C, "
                      f"O2={engine.env.atmosphere.get('O2', 0):.4f}, "
                      f"diversity={diversity:.2f}")

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        plt.ioff()
        plt.show(block=True)
