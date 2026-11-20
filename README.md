# POLCA on 3-digit addition

This repository runs the arithmetic experiment of Kangaslahti, Rosenfeld and Saphra, *Hidden
Breakthroughs in Language Model Training* (ICLR 2026, [arXiv:2506.15872](https://arxiv.org/abs/2506.15872)).

It trains a small Transformer on 3-digit addition and builds the POLCA basis from Hessian
eigenvectors. Each test token's loss change is then tracked along every basis direction, giving
one trajectory per token and direction. The trajectories are clustered, and the clusters are
scored by how purely they collect tokens that need a carry.

The basis is built with the paper's method (Algorithm 1) only.

## Setup

Python 3.10 and a CUDA GPU. Steps 1, 4 and 5 also run on a CPU; steps 2 and 3 are slow there.

`setup_env.sh` creates a virtual environment with the packages in `requirements.txt`. Edit the
settings at its top first:
- `PYTHON`: a Python 3.10 interpreter.
- `VENV`: where to put the environment. The torch wheel is about 3 GB.
- `CUDA`: which torch build to install, for example `cu130`, `cu126` or `cpu`.

```bash
./setup_env.sh
source .venv/bin/activate
```

## Running

`run_pipeline.sh` runs all five steps for one seed. Edit the settings at its top first:
- `SEED`
- `VENV`: the environment from `setup_env.sh`.
- `RUN`: the run directory. It needs about 21 GB.
- `JOBS`: CPU cores for clustering.

```bash
./run_pipeline.sh
```

To run it on a cluster, add your scheduler's header lines (resources, time limit) at the top of
the script and submit it.

Each step can also be run on its own. Every step reads what the previous step saved in the run
directory:

```bash
RUN=runs/s42
python train.py $RUN --seed 42        # 1. train, save checkpoints
python basis.py $RUN                  # 2. Hessian basis (paper Algorithm 1)
python trajectories.py $RUN           # 3. POLCA trajectories + filters
python cluster.py $RUN                # 4. HDBSCAN clusters per basis vector
python stats.py $RUN                  # 5. homogeneity, recall, hidden breakthroughs
```

Every script has `--help` with its options and definitions.

| Step | Script | Device | Time (seed 42) | Output size |
|---|---|---|---|---|
| 1 | `train.py` | GPU | ~2 min (A100) | 18 GB (501 checkpoints) |
| 2 | `basis.py` | GPU | ~8 min (RTX PRO 6000) | 1.8 GB |
| 3 | `trajectories.py` | GPU | ~21 min (L40S) | 1.5 GB |
| 4 | `cluster.py` | CPU, 8 cores | < 1 min | small |
| 5 | `stats.py` | CPU | < 1 min | small |

## What each step does

### 1. Train (`train.py`)

- **Model:** the paper's 3-layer decoder-only Transformer, 9,475,594 parameters.
  - d_model 512, 4 heads, MLP 2048 with ReLU.
  - Pre-LayerNorm, learned position embeddings.
  - TransformerLens's default initialisation.
- **Data:** problems `AAA+BBB=CCCC`. The loss is cross-entropy on the 4 answer digits.
- **Training:** AdamW with learning rate 1e-5, betas (0.9, 0.95) and weight decay 0.1, a 100-step
  warmup, batch size 64, 10,000 steps.
- **Test set:** the first 1,250 problems (5,000 answer tokens) drawn with the same seed.
- **Saved:**
  - checkpoints every 20 steps (`model_step_{e}.pt` holds the weights after loop iteration e)
  - the test problems, with each answer token's place (1000s, 100s, 10s, 1s) and whether a carry
    flows into it

### 2. Basis (`basis.py`)

The basis is built one row at a time, at 50 checkpoints: steps 200, 400, …, 9,800 and the last
step. At each one:

- **H:** the Hessian of the mean answer-token loss, estimated on a fresh sample of 1,250
  problems. As in the released code, each sample is the next draw from the generator that
  produced the test set. They are new problems, not the test set itself.
- **Π:** projects out the rows already in the basis.
- **New row:** the top eigenvector of ΠH, the most curved direction the basis does not yet
  contain. Its eigenvalue is saved too, because step 3 uses it.

The eigenvector is found by power iteration with the released library's settings (at most 20
steps). The library is `hessian_eigenthings`; its random number sequence is kept, so the same
checkpoints give the same basis.

Output: `basis/basis.npy` with shape (50, 9,475,594), plus `eigenvalues.npy` and
`basis_steps.npy`.

### 3. Trajectories (`trajectories.py`)

**Projections.** At every 20th checkpoint (t = 0, 20, …, 9,980), for each test token x and basis
vector b, two quantities are computed:
- the gradient of the token's loss along b, ⟨b, ∇L(x)⟩;
- the weight step along b, ⟨b, θ_t − θ_{t−1}⟩.

**Trajectory.** Each token's trajectory along b is the cumulative sum, over the intervals so
far, of the estimated loss change along b:
- **first order** (Eq. 2): the gradient times the step;
- **second order** (Eq. 6, the default and the paper's POLCA): adds the term
  ½ λ_b ⟨b, Δθ⟩² Δg(x) / ΔG, where λ_b is the basis eigenvalue and G sums the tokens' gradients.

**Filters** (Section 4.1):
- a basis vector is kept only if the mean trajectory ends below zero, meaning the loss along that
  direction decreased overall;
- within a kept vector, a token is kept only if its own trajectory ends below zero.

Outputs: `trajectories/projections.npz` (the raw projections, reused if present) and
`trajectories/trajectories.npz` (trajectories and filter masks).

### 4. Cluster (`cluster.py`)

Each kept basis vector is clustered on its own, as in the released clustering script:
1. z-score every checkpoint across tokens;
2. centre each trajectory;
3. compute cosine distances between trajectories;
4. run HDBSCAN with leaf selection. Settings from Appendix Table 4: `min_cluster_size` 150,
   `min_samples` = kept tokens // 15.

Output: `clusters/labels.npz`, with labels of shape (tokens, vectors):
- 0, 1, …: cluster id within that vector
- −1: HDBSCAN outlier
- −2: filtered out in step 3

### 5. Statistics (`stats.py`)

Per cluster (Appendix G):
- **Homogeneity:** fraction of the cluster's tokens that receive a carry.
- **Recall and F1:** recall is the cluster's share of that vector's clustered carry tokens; F1
  combines homogeneity and recall.
- **Place counts:** tokens per answer place.
- **Central-carry fraction:** share of the cluster's carry tokens that are in the 10s or 100s
  place.
- **Breakthrough step** (Eq. 9): the step of the sharpest bend in the trajectory, averaged over
  the cluster's tokens. A breakthrough is **hidden** if it comes after step 1,000 (Eq. 10).

Summary, next to the paper's Table 8 and 9 values:
- **Top 3 homogeneities with the Appendix G filter:** only clusters whose carry tokens are more
  than 85% in the 10s/100s place. This is the paper's stated criterion.
- **Top 3 without the filter:** what the released `compute_label_homogeneity` computes.
- **Fraction of clusters with a hidden breakthrough.**

Outputs: `stats/clusters.csv`, `stats/summary.json`, `stats/summary.md`.


## Files

```
common.py         model, data generator, loss, token labels, checkpoint helpers
train.py          step 1
basis.py          step 2
trajectories.py   step 3
cluster.py        step 4
stats.py          step 5
requirements.txt  pinned package versions
setup_env.sh      creates the virtual environment
run_pipeline.sh   runs all five steps
```
