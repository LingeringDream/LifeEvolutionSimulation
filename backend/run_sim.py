"""Entry point for running the evolution simulation from command line.

Usage:
    python run_sim.py                                      # No AI (default)
    python run_sim.py --ai openai                          # With OpenAI
    python run_sim.py --ai claude                          # With Claude
    python run_sim.py --ai custom --ai-base-url http://localhost:11434/v1 --ai-model llama3
    python run_sim.py --ai custom --ai-base-url https://api.deepseek.com/v1 --ai-model deepseek-chat
    python run_sim.py --ai openai --ai-interval 30 --planet mars --steps 5000
"""
import argparse
from visualize import run_visual_simulation
from ai.factory import create_ai_provider


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
    parser.add_argument("--ai", type=str, default=None,
                        choices=["openai", "claude", "custom"],
                        help="AI provider: openai, claude, or custom (OpenAI-compatible)")
    parser.add_argument("--ai-model", type=str, default=None,
                        help="AI model name (e.g. gpt-4o-mini, deepseek-chat, llama3)")
    parser.add_argument("--ai-key", type=str, default=None,
                        help="AI API key (or set in .env)")
    parser.add_argument("--ai-base-url", type=str, default=None,
                        help="Custom API base URL (required for --ai custom)")
    parser.add_argument("--ai-interval", type=int, default=60,
                        help="Ticks between AI evolution calls (default: 60)")
    args = parser.parse_args()

    # Create AI provider if requested
    ai_provider = None
    if args.ai:
        ai_provider = create_ai_provider(
            provider=args.ai,
            api_key=args.ai_key,
            model=args.ai_model,
            base_url=args.ai_base_url,
        )
        model_info = args.ai_model or "default"
        url_info = f", url={args.ai_base_url}" if args.ai_base_url else ""
        print(f"AI enabled: {args.ai} (model: {model_info}{url_info}, interval: {args.ai_interval} ticks)")

    print(f"Starting simulation: planet={args.planet}, "
          f"producers={args.producers}, consumers={args.consumers}, "
          f"steps={args.steps}, grid={args.grid}x{args.grid}")

    run_visual_simulation(
        planet_name=args.planet,
        n_producers=args.producers,
        n_consumers=args.consumers,
        steps=args.steps,
        grid_size=args.grid,
        ai_provider=ai_provider,
        ai_interval=args.ai_interval,
    )


if __name__ == "__main__":
    main()
