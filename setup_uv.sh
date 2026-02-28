#!/bin/bash
# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv could not be found. Please install it first (e.g. 'curl -LsSf https://astral.sh/uv/install.sh | sh')"
    exit 1
fi

echo "Creating virtual environment with uv..."
uv venv .venv --python 3.10

echo "Activating environment..."
source .venv/bin/activate

echo "Installing dependencies..."
# gsplat-mps or diff-gaussian-rasterization usually needs manual compile or specific wheel.
# For this script we install standard requirements.
uv pip install -r requirements.txt

echo "Setup complete. Activate with 'source .venv/bin/activate'."
