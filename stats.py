"""Step 5: statistics of the clusters (paper Section 4 and Appendix G).

For every cluster of every basis vector:
  homogeneity             fraction of the cluster's tokens that receive a carry
  recall                  carry tokens in the cluster / carry tokens clustered for that vector
                          (outliers excluded); F1 of homogeneity and recall
  central_carry_fraction  fraction of the cluster's carry tokens in the 10s or 100s place
  places                  token counts per answer place [1000s, 100s, 10s, 1s]
  mean_breakthrough       mean over the cluster's tokens of the breakthrough step (Eq. 9): the step
                          t of the largest f(t+D) - 2 f(t) + f(t-D), D = 100 steps, on the token's
                          trajectory f
  hidden                  mean breakthrough after tau = 1,000 steps (Eq. 10)

Summary:
  top 3 homogeneity, Appendix G   among clusters whose carry tokens are > 85% in the 10s/100s place
  top 3 homogeneity, no filter    what the released compute_label_homogeneity computes
  hidden fraction                 fraction of all clusters with a hidden breakthrough

Outputs in RUN_DIR/stats/: clusters.csv (one row per cluster), summary.json, summary.md

Usage:
  python stats.py RUN_DIR [--delta 100] [--tau 1000]
"""

import argparse
import csv
import json
import os

import numpy as np

# Paper Table 8 (top 3 homogeneities with recall) and Table 9 (hidden fraction)
PAPER = {
    "second": ("0.973 (.74), 0.946 (.77), 0.903 (.76)", 0.355),
    "first": ("0.948 (.77), 0.928 (.77), 0.887 (.75)", 0.307),
}


def breakthrough_steps(traj, steps, delta):
    """Eq. 9 per token: the step of the largest f(t + D) - 2 f(t) + f(t - D)."""
    lag = max(1, int(round(delta / np.median(np.diff(steps)))))
    second_diff = traj[:, 2 * lag :] - 2 * traj[:, lag:-lag] + traj[:, : -2 * lag]
    return steps[np.argmax(second_diff, axis=1) + lag]


def cluster_stats(labels, carry, place, breakthrough, tau):
    clustered = labels >= 0
    total_carry = max(int(carry[clustered].sum()), 1)
    rows = []
    for c in sorted(set(labels[clustered])):
        m = labels == c
        n_carry = int(carry[m].sum())
        hom, rec = n_carry / int(m.sum()), n_carry / total_carry
        carry_places = place[m & (carry == 1)]
        rows.append({
            "cluster": int(c), "size": int(m.sum()), "homogeneity": hom, "recall": rec,
            "f1": 2 * hom * rec / (hom + rec) if hom + rec else 0.0,
            "central_carry_fraction": float(np.isin(carry_places, [1, 2]).mean()) if carry_places.size else 0.0,
            "places": [int(np.sum(m & (place == p))) for p in range(4)],
            "mean_breakthrough": float(breakthrough[m].mean()), "hidden": bool(breakthrough[m].mean() > tau),
        })  # fmt: skip
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--delta", type=int, default=100, help="breakthrough window D in steps")
    ap.add_argument("--tau", type=int, default=1000, help="hidden-breakthrough threshold in steps")
    a = ap.parse_args()

    T = np.load(os.path.join(a.run_dir, "trajectories", "trajectories.npz"))
    traj, keep, steps, place, carry = T["traj"], T["keep"], T["steps"], T["place"], T["carry"]
    order = str(T["order"])
    labels = np.load(os.path.join(a.run_dir, "clusters", "labels.npz"))["labels"]

    rows = []
    for k in np.flatnonzero((labels >= 0).any(0)):
        bt = np.full(len(labels), np.nan)
        bt[keep[:, k]] = breakthrough_steps(traj[keep[:, k], :, k], steps, a.delta)
        for r in cluster_stats(labels[:, k], carry, place, bt, a.tau):
            rows.append({"vector": int(k), **r})

    by_hom = sorted(rows, key=lambda r: -r["homogeneity"])
    central = [r for r in by_hom if r["central_carry_fraction"] > 0.85]
    summary = {
        "order": order, "vectors_kept": int(T["vector_kept"].sum()), "vectors": int(len(T["vector_kept"])),
        "clusters": len(rows), "top3_appendix_g": central[:3], "top3_unfiltered": by_hom[:3],
        "hidden_fraction": float(np.mean([r["hidden"] for r in rows])) if rows else float("nan"),
        "paper": PAPER.get(order), "delta": a.delta, "tau": a.tau,
    }  # fmt: skip

    out = os.path.join(a.run_dir, "stats")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    fields = [
        "vector",
        "cluster",
        "size",
        "homogeneity",
        "recall",
        "f1",
        "central_carry_fraction",
        "places",
        "mean_breakthrough",
        "hidden",
    ]
    with open(os.path.join(out, "clusters.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(r):
        return f"{r['homogeneity']:.3f} (recall {r['recall']:.2f}, vector {r['vector']}, places {r['places']})"

    paper = PAPER.get(order, ("-", float("nan")))
    lines = [
        f"# Cluster statistics ({order}-order POLCA)", "",
        f"- Vectors kept by the filter: {summary['vectors_kept']} / {summary['vectors']}",
        f"- Clusters: {summary['clusters']}", "",
        "| Score | This run | Paper |", "|---|---|---|",
        f"| Top 3 homogeneity, Appendix G filter | {'<br>'.join(map(fmt, central[:3])) or '-'} | {paper[0]} |",
        f"| Top 3 homogeneity, no filter | {'<br>'.join(map(fmt, by_hom[:3])) or '-'} | |",
        f"| Fraction of clusters with hidden breakthroughs | {summary['hidden_fraction']:.3f} | {paper[1]} |",
    ]  # fmt: skip
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
