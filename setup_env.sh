#!/usr/bin/env bash
# One-shot setup for the VPD paper-replication repo.
#
# Uses `uv` (https://astral.sh/uv) to:
#   1. install itself if missing (single static binary, no root needed),
#   2. fetch CPython 3.13 if it isn't already on the system,
#   3. create a .venv pinned to that interpreter,
#   4. install the upstream `spd` package (editable, from a pinned branch)
#      and this repo (editable) into the venv.
#
# Requires: a CUDA-12.4-compatible GPU and git. Python 3.13 does NOT need to
# be pre-installed — uv will fetch it.
#
# Usage:
#   bash setup_env.sh
#   source .venv/bin/activate

set -euo pipefail

NN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPD_DIR="${SPD_DIR:-$NN_DIR/external/spd}"
SPD_BRANCH="${SPD_BRANCH:-snapshot/launch-20260225_151714}"  # branch used to train the VPD baseline (s-55ea3f9b)

# --------------------------------------------------------------------------
# 1. Ensure `uv` is on PATH (~25 MB static binary, no root required).
# --------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer drops uv into ~/.local/bin (or $XDG_BIN_HOME). Source the
    # generated env file so this shell picks it up immediately.
    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck source=/dev/null
        source "$HOME/.local/bin/env"
    else
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
echo "Using uv: $(uv --version)"

# Some images (RunPod, Colab) preset `UV_SYSTEM_PYTHON=1`, which makes
# `uv pip` ignore the venv and dump packages into the host Python.
# Clear it so every `uv pip install` below targets our venv.
unset UV_SYSTEM_PYTHON
unset UV_PYTHON

# --------------------------------------------------------------------------
# 2-3. Install Python 3.13 if needed and create the venv.
# --------------------------------------------------------------------------
uv python install 3.13
echo "Creating venv at $NN_DIR/.venv (Python 3.13)..."
uv venv --python 3.13 "$NN_DIR/.venv"

# Pass --python explicitly to every `uv pip install` so it targets the venv
# regardless of any host Python on PATH.
VENV_PY="$NN_DIR/.venv/bin/python"
export VIRTUAL_ENV="$NN_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# --------------------------------------------------------------------------
# 4. Install PyTorch (CUDA 12.4) + the upstream `spd` package + this repo.
# --------------------------------------------------------------------------
uv pip install --python "$VENV_PY" torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "Cloning upstream spd into $SPD_DIR (branch $SPD_BRANCH)..."
mkdir -p "$(dirname "$SPD_DIR")"
if [ ! -d "$SPD_DIR" ]; then
    git clone --branch "$SPD_BRANCH" https://github.com/goodfire-ai/spd.git "$SPD_DIR"
else
    (cd "$SPD_DIR" && git fetch origin "$SPD_BRANCH" && git checkout "$SPD_BRANCH")
fi
uv pip install --python "$VENV_PY" -e "$SPD_DIR"

echo "Installing nn_decompositions (editable)..."
uv pip install --python "$VENV_PY" -e "$NN_DIR"

# --------------------------------------------------------------------------
# 5. Smoke test (use the venv interpreter explicitly so the result is
#    independent of whatever Python happens to be first on PATH).
# --------------------------------------------------------------------------
"$VENV_PY" -c "
import sys; print(f'python {sys.version.split()[0]}')
import torch
print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
import spd; print('spd OK')
from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.clt import CrossLayerTranscoder
print('nn_decompositions OK')
"

echo ""
echo "Setup complete. Activate with:  source $NN_DIR/.venv/bin/activate"
