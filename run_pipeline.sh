#!/bin/bash
# Run all five steps for one seed, then make the paper's figures and Table 8.
# Edit the settings below for your machine, then run:
#   ./run_pipeline.sh
# To submit it to a cluster scheduler, add your scheduler's header lines (resources, time limit)
# at the top and submit this file.
set -euo pipefail

# ---- settings --------------------------------------------------------------------------------
SEED=42                   # data and initialisation seed
VENV=.venv                # environment created by setup_env.sh
RUN=runs/s${SEED}         # run directory: checkpoints (~18 GB) and all outputs go here
JOBS=8                    # CPU cores for clustering (step 4)
# ----------------------------------------------------------------------------------------------

cd "$(dirname "$0")"
source "$VENV/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # clustering parallelises over vectors

echo "run directory: $RUN"
python -u train.py "$RUN" --seed "$SEED"      # 1. train, save checkpoints          (GPU)
python -u basis.py "$RUN"                     # 2. Hessian basis, paper Algorithm 1  (GPU)
python -u trajectories.py "$RUN"              # 3. POLCA trajectories and filters    (GPU)
python -u cluster.py "$RUN" --jobs "$JOBS"    # 4. HDBSCAN clusters per basis vector (CPU)
python -u stats.py "$RUN"                     # 5. homogeneity, recall, hidden breakthroughs
python -u paper_figures.py "$RUN"             # paper Figures 3, 4 and 7, and Table 8
