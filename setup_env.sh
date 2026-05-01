#!/usr/bin/env bash
# Set up a Python 3.13 venv with all dependencies needed to replicate
# the VPD paper's transcoder and pareto experiments.
#
# Requires: Python 3.13, CUDA 12.4-compatible GPU, git.
#
# Usage:
#   bash setup_env.sh
#   source .venv/bin/activate

set -euo pipefail

NN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPD_DIR="${SPD_DIR:-$NN_DIR/external/spd}"
SPD_BRANCH="${SPD_BRANCH:-snapshot/launch-20260225_151714}"  # branch used to train the VPD baseline (s-55ea3f9b)

if ! command -v python3.13 &>/dev/null; then
    echo "Error: python3.13 is required but not found." >&2
    echo "Install it via deadsnakes (Ubuntu) or pyenv, then re-run." >&2
    exit 1
fi

echo "Creating venv at $NN_DIR/.venv ..."
python3.13 -m venv "$NN_DIR/.venv"
# shellcheck source=/dev/null
source "$NN_DIR/.venv/bin/activate"

pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "Cloning SPD repo into $SPD_DIR (branch $SPD_BRANCH)..."
mkdir -p "$(dirname "$SPD_DIR")"
if [ ! -d "$SPD_DIR" ]; then
    git clone --branch "$SPD_BRANCH" https://github.com/goodfire-ai/spd.git "$SPD_DIR"
else
    (cd "$SPD_DIR" && git fetch origin "$SPD_BRANCH" && git checkout "$SPD_BRANCH")
fi
pip install -e "$SPD_DIR"

echo "Installing nn_decompositions (editable)..."
pip install -e "$NN_DIR"

# Update sys.path hint for the experiment scripts so they can find SPD locally.
echo "Note: experiment scripts add /workspace/spd to sys.path. If you cloned"
echo "      SPD elsewhere, edit the sys.path.insert lines or symlink:"
echo "        ln -s '$SPD_DIR' /workspace/spd"

python -c "
import torch
print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
import spd; print('spd OK')
from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.clt import CrossLayerTranscoder
print('nn_decompositions OK')
"

echo ""
echo "Setup complete. Activate with:  source $NN_DIR/.venv/bin/activate"
