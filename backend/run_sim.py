"""Entry point for running the evolution simulation.

Usage:
    python run_sim.py                              # Web mode (default), auto-opens browser
    python run_sim.py --cli                        # Matplotlib CLI mode
    python run_sim.py --port 3000                  # Web on custom port
    python run_sim.py --planet mars --steps 5000   # Custom params (web mode)
"""
import sys
import os
import argparse
import webbrowser
import threading

# Ensure backend/ is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_web(args):
    """Start FastAPI web server and auto-open browser."""
    import uvicorn
    from api.app import app, sim

    url = f"http://{args.host}:{args.port}"

    # Use uvicorn startup event to open browser
    @app.on_event("startup")
    async def _open_browser():
        webbrowser.open_new_tab(url)

    print(f"Web 服务器启动: {url}")
    print("浏览器将自动打开...")
    print("按 Ctrl+C 停止\n")

    uvicorn.run(app, host=args.host, port=args.port)


def run_cli(args):
    """Run matplotlib CLI visualization."""
    from visualize import run_visual_simulation
    from ai.factory import create_ai_provider

    ai_provider = None
    if args.ai:
        ai_provider = create_ai_provider(
            provider=args.ai, api_key=args.ai_key,
            model=args.ai_model, base_url=args.ai_base_url,
        )

    print(f"CLI 模式: planet={args.planet}, producers={args.producers}, "
          f"consumers={args.consumers}, steps={args.steps}, grid={args.grid}")

    run_visual_simulation(
        planet_name=args.planet, n_producers=args.producers,
        n_consumers=args.consumers, steps=args.steps,
        grid_size=args.grid, ai_provider=ai_provider,
        ai_interval=args.ai_interval,
    )


def main():
    parser = argparse.ArgumentParser(description="外星球演化模拟器")
    parser.add_argument("--cli", action="store_true", help="使用 matplotlib 命令行模式（默认为 Web 模式）")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web 服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="Web 服务器端口")
    parser.add_argument("--planet", type=str, default="titan")
    parser.add_argument("--producers", type=int, default=2)
    parser.add_argument("--consumers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--grid", type=int, default=50)
    parser.add_argument("--ai", type=str, default=None)
    parser.add_argument("--ai-model", type=str, default=None)
    parser.add_argument("--ai-key", type=str, default=None)
    parser.add_argument("--ai-base-url", type=str, default=None)
    parser.add_argument("--ai-interval", type=int, default=60)
    args = parser.parse_args()

    if args.cli:
        run_cli(args)
    else:
        run_web(args)


if __name__ == "__main__":
    main()
