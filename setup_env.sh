#!/bin/bash
# Setup script for nn_decompositions on RunPod (Ubuntu 22.04, CUDA 12.4).
#
# Installs the venv on the local machine disk (fast) and symlinks it
# from the workspace so `source .venv/bin/activate` still works.
#
# Usage:
#   bash setup_env.sh          # full install
#   source .venv/bin/activate  # activate after install
#
set -euo pipefail

LOCAL_VENV="/root/nn_decompositions_venv"
SYMLINK="/workspace/nn_decompositions/.venv"
NN_DIR="/workspace/nn_decompositions"
SPD_DIR="/workspace/bartbussmann_spd"

# --------------------------------------------------------------------------
# 1. Ensure Python 3.13 is available
# --------------------------------------------------------------------------
if command -v python3.13 &>/dev/null; then
    PY=python3.13
    echo "Found $($PY --version) at $(which $PY)"
else
    echo "Python 3.13 not found — installing via deadsnakes PPA..."
    apt-get update -qq
    apt-get install -y -qq software-properties-common

    if ! add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null; then
        echo "add-apt-repository failed (likely broken apt_pkg) — trying to fix..."
        # Reinstall python3-apt and symlink the .so for the current python3
        apt-get install -y -qq --reinstall python3-apt 2>/dev/null || true
        SO_FILE=$(find /usr/lib/python3/dist-packages -name 'apt_pkg.cpython-*.so' 2>/dev/null | head -1)
        if [ -n "$SO_FILE" ]; then
            ln -sf "$SO_FILE" /usr/lib/python3/dist-packages/apt_pkg.so
        fi

        if ! add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null; then
            echo "Still failing — adding deadsnakes PPA manually..."
            CODENAME=$(. /etc/os-release && echo "${VERSION_CODENAME:-jammy}")
            echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu ${CODENAME} main" \
                > /etc/apt/sources.list.d/deadsnakes-ppa.list
            apt-get install -y -qq gpg
            apt-key adv --keyserver keyserver.ubuntu.com \
                --recv-keys F23C5A6CF475977595C89F51BA6932366A755776
        fi
    fi

    apt-get update -qq
    apt-get install -y -qq python3.13 python3.13-venv python3.13-dev
    PY=python3.13
    echo "Installed $($PY --version)"
fi

# --------------------------------------------------------------------------
# 2. Create venv on LOCAL disk (fast I/O)
# --------------------------------------------------------------------------
if [ -d "$LOCAL_VENV" ]; then
    echo "Removing existing venv at $LOCAL_VENV..."
    rm -rf "$LOCAL_VENV"
fi
# Also clean up any old symlink or directory at the workspace path
if [ -L "$SYMLINK" ] || [ -d "$SYMLINK" ]; then
    rm -rf "$SYMLINK"
fi

echo "Creating venv at $LOCAL_VENV (local disk)..."
$PY -m venv "$LOCAL_VENV"

# Symlink so `source .venv/bin/activate` works from the project dir
ln -s "$LOCAL_VENV" "$SYMLINK"
echo "Symlinked $SYMLINK -> $LOCAL_VENV"

source "$LOCAL_VENV/bin/activate"
pip install --upgrade pip setuptools wheel

# --------------------------------------------------------------------------
# 3. Install PyTorch + CUDA 12.4
# --------------------------------------------------------------------------
echo "Installing PyTorch with CUDA 12.4 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# --------------------------------------------------------------------------
# 4. Install SPD (editable, with all its deps)
# --------------------------------------------------------------------------
echo "Installing SPD (editable)..."
pip install -e "$SPD_DIR"

# --------------------------------------------------------------------------
# 5. Install nn_decompositions (editable, with all extras)
# --------------------------------------------------------------------------
echo "Installing nn-decompositions (editable, all extras)..."
pip install -e "$NN_DIR[dev,analysis,simplestories]"

# --------------------------------------------------------------------------
# 6. Extra packages used by experiment scripts but not in pyproject.toml
# --------------------------------------------------------------------------
echo "Installing extra experiment dependencies..."
pip install openai tabulate pyyaml

# --------------------------------------------------------------------------
# 7. Verify
# --------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Verifying installation..."
echo "============================================================"
python -c "
import torch
print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda}')
import wandb, datasets, tqdm, matplotlib, einops, jaxtyping
print('Core packages OK')
import spd
print('SPD OK')
from nn_decompositions.transcoder import BatchTopKTranscoder
from nn_decompositions.config import EncoderConfig, CLTConfig
from nn_decompositions.clt import CrossLayerTranscoder
print('nn_decompositions core imports OK')
from spd.models.components import make_mask_infos
print('SPD imports OK')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "  Activate with:  source $SYMLINK/bin/activate"
echo "  Venv location:  $LOCAL_VENV (local disk)"
echo ""
echo "  NOTE: The venv lives on the machine's local disk for speed."
echo "  It will NOT persist across pod restarts — rerun this script."
echo "============================================================"
