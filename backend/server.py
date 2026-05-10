"""Web server entry point.

Usage:
    cd backend
    python server.py                      # Start on http://localhost:8000
    python server.py --port 3000          # Custom port
    python server.py --host 0.0.0.0       # Allow external access
"""
import sys
import os
import argparse

# Ensure backend/ is in Python path so api/, ai/, simulation/ are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Alien Evolution Simulator Web Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    print(f"Starting web server at http://{args.host}:{args.port}")
    print("Open in browser to begin.")

    from api.app import app
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
