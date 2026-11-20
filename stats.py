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

Computed for each trajectory type (first and second order).

Outputs in RUN_DIR/stats/: clusters_{order}.csv (one row per cluster), summary_{order}.json, and
summary.md for all types

Usage:
  python stats.py RUN_DIR [--orders first second] [--delta 100] [--tau 1000]
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


def summarise(run_dir, order, delta, tau):
    """Score every cluster of one trajectory type; returns (rows, summary)."""
    T = np.load(os.path.join(run_dir, "trajectories", f"trajectories_{order}.npz"))
    traj, keep, steps, place, carry = T["traj"], T["keep"], T["steps"], T["place"], T["carry"]
    labels = np.load(os.path.join(run_dir, "clusters", f"labels_{order}.npz"))["labels"]
    rows = []
    for k in np.flatnonzero((labels >= 0).any(0)):
        bt = np.full(len(labels), np.nan)
        bt[keep[:, k]] = breakthrough_steps(traj[keep[:, k], :, k], steps, delta)
        for r in cluster_stats(labels[:, k], carry, place, bt, tau):
            rows.append({"vector": int(k), **r})
    by_hom = sorted(rows, key=lambda r: -r["homogeneity"])
    central = [r for r in by_hom if r["central_carry_fraction"] > 0.85]
    summary = {
        "order": order, "vectors_kept": int(T["vector_kept"].sum()), "vectors": int(len(T["vector_kept"])),
        "clusters": len(rows), "top3_appendix_g": central[:3], "top3_unfiltered": by_hom[:3],
        "hidden_fraction": float(np.mean([r["hidden"] for r in rows])) if rows else float("nan"),
        "paper": PAPER.get(order), "delta": delta, "tau": tau,
    }  # fmt: skip
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--orders", nargs="+", choices=["first", "second"], default=["first", "second"])
    ap.add_argument("--delta", type=int, default=100, help="breakthrough window D in steps")
    ap.add_argument("--tau", type=int, default=1000, help="hidden-breakthrough threshold in steps")
    a = ap.parse_args()

    out = os.path.join(a.run_dir, "stats")
    os.makedirs(out, exist_ok=True)
    fields = ["vector", "cluster", "size", "homogeneity", "recall", "f1", "central_carry_fraction", "places",
              "mean_breakthrough", "hidden"]  # fmt: skip

    def fmt(r):
        return f"{r['homogeneity']:.3f} (recall {r['recall']:.2f}, vector {r['vector']}, places {r['places']})"

    lines = ["# Cluster statistics", ""]
    for order in a.orders:
        rows, summary = summarise(a.run_dir, order, a.delta, a.tau)
        with open(os.path.join(out, f"summary_{order}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(out, f"clusters_{order}.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        paper = PAPER.get(order, ("-", float("nan")))
        lines += [
            f"## {order.capitalize()}-order POLCA", "",
            f"- Vectors kept by the filter: {summary['vectors_kept']} / {summary['vectors']}",
            f"- Clusters: {summary['clusters']}", "",
            "| Score | This run | Paper |", "|---|---|---|",
            f"| Top 3 homogeneity, Appendix G filter | {'<br>'.join(map(fmt, summary['top3_appendix_g'])) or '-'} "
            f"| {paper[0]} |",
            f"| Top 3 homogeneity, no filter | {'<br>'.join(map(fmt, summary['top3_unfiltered'])) or '-'} | |",
            f"| Fraction of clusters with hidden breakthroughs | {summary['hidden_fraction']:.3f} | {paper[1]} |", "",
        ]  # fmt: skip
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
