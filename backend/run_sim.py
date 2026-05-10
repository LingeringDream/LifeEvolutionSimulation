"""Entry point for running the evolution simulation from command line.

Usage:
    python run_sim.py                    # Run with defaults (Titan, 2 producers, 1 consumer)
    python run_sim.py --planet mars      # Run on Mars
    python run_sim.py --producers 3 --consumers 2 --steps 5000
"""
import argparse
from visualize import run_visual_simulation


def main():
    parser = argparse.ArgumentParser(description="Alien Planet Evolution Simulator")
    parser.add_argument("--planet", type=str, default="titan",
                        choices=["titan", "mars", "europa", "kepler442b"],
                        help="Planet to simulate (default: titan)")
    parser.add_argument("--producers", type=int, default=2,
                        help="Number of producer species (default: 2)")
    parser.add_argument("--consumers", type=int, default=1,
                        help="Number of consumer species (default: 1)")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Simulation duration in ticks (default: 2000)")
    parser.add_argument("--grid", type=int, default=50,
                        help="Grid size (default: 50)")
    args = parser.parse_args()

    print(f"Starting simulation: planet={args.planet}, "
          f"producers={args.producers}, consumers={args.consumers}, "
          f"steps={args.steps}, grid={args.grid}x{args.grid}")

    run_visual_simulation(
        planet_name=args.planet,
        n_producers=args.producers,
        n_consumers=args.consumers,
        steps=args.steps,
        grid_size=args.grid,
    )


if __name__ == "__main__":
    main()
