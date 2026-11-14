"""Run the API: python -m src.api [--host 127.0.0.1] [--port 8000].

On a hosting service set HOST and PORT (Render sets PORT; render.yaml sets
HOST=0.0.0.0). Locally the defaults keep the server on 127.0.0.1:8000.
"""

from __future__ import annotations

import argparse
import os

import uvicorn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the FireLens API.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    uvicorn.run("src.api.app:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
