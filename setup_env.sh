#!/bin/bash
# Create a virtual environment with the packages in requirements.txt.
# Edit the settings below for your machine, then run:  ./setup_env.sh
set -euo pipefail

# ---- settings --------------------------------------------------------------------------------
PYTHON=python3        # a Python 3.10 interpreter
VENV=.venv            # where to create the environment (the torch wheel is ~3 GB)
CUDA=cu130            # torch build: cu130, cu126, ... or cpu (see https://pytorch.org/get-started)
# ----------------------------------------------------------------------------------------------

HERE=$(cd "$(dirname "$0")" && pwd)
"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install torch==2.14.1 --index-url "https://download.pytorch.org/whl/$CUDA"
"$VENV/bin/pip" install -r "$HERE/requirements.txt"
"$VENV/bin/python" -c "import torch, sklearn; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| scikit-learn', sklearn.__version__)"
echo "activate with: source $VENV/bin/activate"
