"""Web server entry point.

Usage:
    python server.py                      # Start on http://localhost:8000
    python server.py --port 3000          # Custom port
    python server.py --host 0.0.0.0       # Allow external access
"""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Alien Evolution Simulator Web Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    print(f"Starting web server at http://{args.host}:{args.port}")
    print("Open in browser to begin.")

    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
