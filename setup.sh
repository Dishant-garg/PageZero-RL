#!/bin/bash
# setup.sh — One-time setup for PageZero Docker permissions
# Run once as yourself (sudo will prompt for password):
#
#   bash setup.sh

set -e

echo "╔══════════════════════════════════════════════════╗"
echo "║         PageZero — One-Time Setup Script         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Add user to docker group so docker commands work without sudo
if ! groups | grep -q docker; then
    echo "Adding $USER to the 'docker' group..."
    sudo usermod -aG docker "$USER"
    echo "✓ Added to docker group."
    echo ""
    echo "⚠  You must LOG OUT and LOG BACK IN (or run: newgrp docker)"
    echo "   for the group change to take effect, then re-run:"
    echo ""
    echo "     bash setup.sh"
    echo ""
    exit 0
else
    echo "✓ $USER is already in the docker group."
fi

# Install Python dependencies for the OpenEnv server
echo ""
echo "Installing Python dependencies..."
if command -v uv &>/dev/null; then
    uv sync
else
    pip install "openenv-core[core]>=0.2.1" fastapi uvicorn --quiet
fi
echo "✓ Python deps installed."

# Bring up the Docker stack
echo ""
echo "Starting Docker stack (Postgres + Redis + App)..."
cd "$(dirname "$0")"
docker compose up -d --build

echo ""
echo "Waiting for healthchecks to pass..."
sleep 8

docker compose ps

echo ""
echo "Running verification..."
python verify.py

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Setup complete! Next steps:                    ║"
echo "║                                                   ║"
echo "║  1. Start the OpenEnv server:                    ║"
echo "║     make server                                   ║"
echo "║     (or: uvicorn server.app:app --reload)        ║"
echo "║                                                   ║"
echo "║  2. Run training (requires GPU + TRL/vLLM):      ║"
echo "║     make train                                    ║"
echo "║                                                   ║"
echo "║  3. Re-verify at any time:                       ║"
echo "║     python verify.py --verbose                   ║"
echo "╚══════════════════════════════════════════════════╝"
