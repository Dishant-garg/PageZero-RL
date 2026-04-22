# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Pagezero Environment.

This module creates an HTTP server that exposes the PagezeroEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

# Absolute imports from the project root
try:
    from models import PageZeroAction, PageZeroObservation
    from server.PageZero_environment import PageZeroEnvironment
except ImportError:
    import sys
    import os
    # Add project root to sys.path if not already there
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import PageZeroAction, PageZeroObservation
    from server.PageZero_environment import PageZeroEnvironment


app = create_app(
    PageZeroEnvironment,
    PageZeroAction,
    PageZeroObservation,
    env_name="PageZero",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()

    uvicorn.run(app, host=host, port=args.port)

if __name__ == "__main__":
    main()
