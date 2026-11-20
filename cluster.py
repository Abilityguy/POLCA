"""Step 4: cluster the tokens' trajectories, separately for each kept basis vector.

As in the released clustering script: z-score each checkpoint across tokens, centre each
token's trajectory, cosine distance, HDBSCAN with leaf cluster selection. Paper settings
(Appendix Table 4): min_cluster_size 150, min_samples = kept tokens // 15.

Each trajectory type from step 3 (first and second order) is clustered separately.

Outputs in RUN_DIR/clusters/:
  labels_{order}.npz  labels (tokens, K) int: cluster id within that vector, -1 = HDBSCAN outlier,
              -2 = not clustered (token or vector filtered out in step 3)
  info.json   settings and the number of clusters per vector

Usage:
  python cluster.py RUN_DIR [--orders first second] [--min_cluster_size 150] [--min_samples 0] [--jobs 8]
"""

import argparse
import json
import os

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances


def cluster(traj, min_cluster_size, min_samples):
    """HDBSCAN labels for trajectories (tokens, T)."""
    z = (traj - traj.mean(0, keepdims=True)) / np.maximum(traj.std(0, keepdims=True), 1e-8)
    z = z - z.mean(1, keepdims=True)
    dist = pairwise_distances(z, metric="cosine")
    np.fill_diagonal(dist, 0.0)
    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="leaf",
        metric="precomputed",
    ).fit_predict(dist)


def cluster_vector(traj, keep, min_cluster_size, min_samples):
    labels = np.full(len(keep), -2)
    n = int(keep.sum())
    ms = min_samples or max(1, n // 15)
    if n >= min_cluster_size:
        labels[keep] = cluster(traj[keep], min_cluster_size, ms)
    return labels, ms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--min_cluster_size", type=int, default=150)
    ap.add_argument("--min_samples", type=int, default=0, help="0 = kept tokens // 15 (paper)")
    ap.add_argument("--jobs", type=int, default=8, help="vectors clustered in parallel")
    ap.add_argument("--orders", nargs="+", choices=["first", "second"], default=["first", "second"])
    a = ap.parse_args()

    out = os.path.join(a.run_dir, "clusters")
    os.makedirs(out, exist_ok=True)
    info = {"args": vars(a)}
    for order in a.orders:
        T = np.load(os.path.join(a.run_dir, "trajectories", f"trajectories_{order}.npz"))
        traj, keep, vector_kept = T["traj"], T["keep"], T["vector_kept"]
        kept = np.flatnonzero(vector_kept)
        print(f"{order} order: clustering {len(kept)} kept vectors of {len(vector_kept)}", flush=True)
        results = Parallel(n_jobs=a.jobs)(
            delayed(cluster_vector)(traj[:, :, k], keep[:, k], a.min_cluster_size, a.min_samples) for k in kept
        )
        labels = np.full(keep.shape, -2)
        min_samples = {}
        for k, (lab, ms) in zip(kept, results, strict=True):
            labels[:, k] = lab
            min_samples[int(k)] = ms
        np.savez(os.path.join(out, f"labels_{order}.npz"), labels=labels)
        n_clusters = {int(k): int(len(set(labels[:, k]) - {-1, -2})) for k in kept}
        info[order] = {"min_samples": min_samples, "clusters_per_vector": n_clusters}
        print(f"{order} order: {sum(n_clusters.values())} clusters over {len(kept)} vectors", flush=True)
        del traj
    with open(os.path.join(out, "info.json"), "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
