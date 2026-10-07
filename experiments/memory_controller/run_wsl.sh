#!/usr/bin/env bash
# Run E1 experiment from WSL Ubuntu.
# Usage: bash experiments/memory_controller/run_wsl.sh [--mock] [--num-tasks N]

set -e

THESIS_ROOT="/mnt/c/Users/ayane/OneDrive/Documents/Masters/Thesis"
VENV="$THESIS_ROOT/.venv_wsl"

# Create virtualenv on first run
if [ ! -d "$VENV" ]; then
    echo "[setup] Creating WSL virtualenv..."
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

# Install dependencies
echo "[setup] Installing dependencies..."
pip install --quiet networkx pyyaml requests matplotlib numpy
pip install --quiet alfworld 2>&1 | tail -3

# Download ALFWorld data if not already present
if [ -z "$ALFWORLD_DATA" ]; then
    DATA_DIR="$THESIS_ROOT/.alfworld_data"
    mkdir -p "$DATA_DIR"
    export ALFWORLD_DATA="$DATA_DIR"
fi

if [ ! -d "$ALFWORLD_DATA/json_2.1.1" ]; then
    echo "[setup] Downloading ALFWorld data (this takes a few minutes)..."
    python3 -m alfworld.scripts.alfworld_download --data-path "$ALFWORLD_DATA"
fi

echo "[setup] ALFWORLD_DATA=$ALFWORLD_DATA"
echo "[run] Starting experiment..."

cd "$THESIS_ROOT"
python3 experiments/memory_controller/main.py "$@"
