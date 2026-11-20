"""The paper's Figures 3, 4 and 7, and Table 8, for a run.

  Figure 3  tokens clustered on their exact-loss trajectories (the exact-loss baseline of
            Table 1): (a) mean +- std exact loss per cluster, (b) place and carry composition
            of each cluster. Needs step 1.
  Figure 4  the POLCA clusters of basis vectors #1 and #2 (computed at steps 200 and 400), one row
            per vector: (a) median and quartiles of each cluster's POLCA trajectory, (b) the exact
            loss of the same clusters, (c) composition with each cluster's carry homogeneity.
            Uses --order (default second, the paper's POLCA). Needs steps 3 and 4.
  Figure 7  median and quartiles of the exact loss of each answer place over training, and the
            same split by whether a carry flows into the place (the paper's Appendix F; the figure
            numbers of these per-digit plots differ between arXiv versions). Needs step 1.
  Table 8   top 3 carry homogeneities (with recall and F1) and the fraction of clusters with hidden
            breakthroughs, for exact loss (the Figure 3 clusters), POLCA and first-order POLCA
            (step 5), next to the paper's values. Two scorings: with the Appendix G filter (only
            clusters whose carry tokens are > 85% in the 10s/100s place, as the paper describes)
            and without it (as the released code scores; the paper's numbers match this one).
            Needs steps 1-5.

In the composition panels, every cluster has one bar per answer place, with the carry tokens
(hatched) stacked under the no-carry tokens. Only clustered tokens are counted: HDBSCAN outliers,
and for Figure 4 the tokens removed by the filter in step 3, are left out.

Figure 3 clusters the exact-loss trajectories of all 5,000 test tokens with cluster.py's method and
settings, and scores the clusters as stats.py does. Paper (Table 1): maximum carry homogeneity
0.514, no hidden breakthroughs.

Outputs in RUN_DIR/figures/:
  figure3.png, figure4.png, figure7.png (per place), figure7_carry.png (per place and carry)
  exact_losses.npz      exact loss of every test token at every saved checkpoint (cached)
  figure3_clusters.npz  Figure 3 cluster labels (-1 = outlier), figure3_clusters.csv their scores,
  figure3_summary.json  the Table 1 numbers
  table8.md             Table 8

Usage:
  python paper_figures.py RUN_DIR [--figures 3 4 7 table8] [--vectors 0 1] [--order second]
"""

import argparse
import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from cluster import cluster  # noqa: E402
from common import PLACES, checkpoint_path, device, load_checkpoint, loss_fn  # noqa: E402
from stats import breakthrough_steps, cluster_stats  # noqa: E402

COLORS = ["#c45ca2", "#b5835a", "#3c8dc5", "#e8c547"]  # 1000s, 100s, 10s, 1s, as in the paper
CLUSTER_COLORS = plt.get_cmap("tab10")
# Paper Table 8: top 3 (homogeneity, recall, F1) and the hidden-breakthrough fraction
PAPER_TABLE8 = {
    "Exact loss": ([(0.514, 0.771, 0.617)], 0.0),
    "POLCA": ([(0.973, 0.736, 0.838), (0.946, 0.773, 0.850), (0.903, 0.762, 0.827)], 0.355),
    "POLCA (first order)": ([(0.948, 0.767, 0.848), (0.928, 0.769, 0.841), (0.887, 0.751, 0.813)], 0.307),
}


# ---------------------------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------------------------


def exact_losses(run_dir, out):
    """(steps, losses (tokens, checkpoints)) for every saved model_step_*.pt, cached in out."""
    files = glob.glob(os.path.join(run_dir, "checkpoints", "model_step_*.pt"))
    steps = np.array(sorted(int(re.findall(r"model_step_(\d+)\.pt", f)[0]) for f in files))
    cache = os.path.join(out, "exact_losses.npz")
    if os.path.exists(cache):
        c = np.load(cache)
        if np.array_equal(c["steps"], steps):
            return steps, c["losses"].astype(np.float64)
    tokens = torch.tensor(np.load(os.path.join(run_dir, "test_tokens.npy"))).to(device)
    losses = np.zeros((tokens.shape[0] * 4, len(steps)), dtype=np.float32)
    with torch.no_grad():
        for t, step in enumerate(steps):
            model = load_checkpoint(checkpoint_path(run_dir, step))
            losses[:, t] = loss_fn(model(tokens), tokens, per_token=True).reshape(-1).cpu().numpy()
            if t % 100 == 0:
                print(f"exact loss: evaluated step {step} ({t + 1}/{len(steps)})", flush=True)
    np.savez(cache, steps=steps, losses=losses)
    return steps, losses.astype(np.float64)


def band(ax, steps, values, color, label, linestyle="-"):
    """Median line with an interquartile band."""
    q1, median, q3 = np.percentile(values, [25, 50, 75], axis=0)
    ax.fill_between(steps, q1, q3, color=color, alpha=0.2, linewidth=0)
    ax.plot(steps, median, color=color, linestyle=linestyle, linewidth=1.8, label=label)


def composition(ax, labels, ids, place, carry, title):
    """One bar per place within each cluster; carry tokens (hatched) under no-carry tokens."""
    x, width = np.arange(len(ids)), 0.2
    for p in range(4):
        n_carry = np.array([np.sum((labels == c) & (place == p) & (carry == 1)) for c in ids])
        n_none = np.array([np.sum((labels == c) & (place == p) & (carry == 0)) for c in ids])
        offset = (p - 1.5) * width
        ax.bar(x + offset, n_carry, width, color=COLORS[p], hatch="///", edgecolor="black", linewidth=0.5)
        ax.bar(x + offset, n_none, width, bottom=n_carry, color=COLORS[p], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x, [str(i + 1) for i in range(len(ids))])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Tokens")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_title(title)
    handles = [Patch(facecolor=COLORS[p], edgecolor="black", label=PLACES[p]) for p in range(4)] + [
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Carry"),
        Patch(facecolor="white", edgecolor="black", label="No carry"),
    ]
    ax.legend(handles=handles, fontsize=7)


def describe(prefix, labels, ids, place, carry):
    for i, c in enumerate(ids):
        m = labels == c
        counts = [f"{np.sum(m & (place == p) & (carry == 1))}/{np.sum(m & (place == p))}" for p in range(4)]
        print(
            f"{prefix} cluster {i + 1}: {m.sum()} tokens, homogeneity {carry[m].mean():.3f}, "
            f"carry/total per place [1000s, 100s, 10s, 1s] {counts}"
        )


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------


def figure3(steps, losses, place, carry, args, out):
    min_samples = args.min_samples or len(losses) // 15
    labels = cluster(losses, args.min_cluster_size, min_samples)
    rows = cluster_stats(labels, carry, place, breakthrough_steps(losses, steps, args.delta), args.tau)
    ids = sorted(set(labels) - {-1})

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, c in enumerate(ids):
        m = labels == c
        mean, std = losses[m].mean(0), losses[m].std(0)
        a.plot(steps, mean, color=CLUSTER_COLORS(i % 10), label=str(i + 1))
        a.fill_between(steps, mean - std, mean + std, color=CLUSTER_COLORS(i % 10), alpha=0.25)
    a.set_xlim(0, args.max_step)
    a.set_xlabel("Iteration")
    a.set_ylabel("Full-rank loss")
    a.set_title("(a) Mean exact loss per cluster")
    a.legend(title="Cluster", fontsize=8)
    composition(b, labels, ids, place, carry, "(b) Skill composition of each cluster")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figure3.png"), dpi=150)
    plt.close(fig)

    np.savez(os.path.join(out, "figure3_clusters.npz"), labels=labels)
    with open(os.path.join(out, "figure3_clusters.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["cluster"])
        writer.writeheader()
        writer.writerows(rows)
    central = [r for r in rows if r["central_carry_fraction"] > 0.85]
    summary = {
        "min_cluster_size": args.min_cluster_size, "min_samples": min_samples, "clusters": len(rows),
        "outliers": int(np.sum(labels == -1)),
        "max_homogeneity_appendix_g": max((r["homogeneity"] for r in central), default=None),
        "max_homogeneity_unfiltered": max((r["homogeneity"] for r in rows), default=None),
        "hidden_fraction": float(np.mean([r["hidden"] for r in rows])) if rows else float("nan"),
        "top3_appendix_g": sorted(central, key=lambda r: -r["homogeneity"])[:3],
        "top3_unfiltered": sorted(rows, key=lambda r: -r["homogeneity"])[:3],
        "paper": {"max_homogeneity": 0.514, "hidden_fraction": 0.0},
    }  # fmt: skip
    with open(os.path.join(out, "figure3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Figure 3: {len(rows)} clusters, {summary['outliers']} outliers (min_samples {min_samples})")
    describe("  exact loss", labels, ids, place, carry)

    def fmt(v):
        return "-" if v is None else f"{v:.3f}"

    print(
        f"  max carry homogeneity {fmt(summary['max_homogeneity_unfiltered'])} (no filter), "
        f"{fmt(summary['max_homogeneity_appendix_g'])} (Appendix G filter); paper 0.514"
    )
    print(f"  hidden-breakthrough fraction {summary['hidden_fraction']:.3f}; paper 0.0")
    return summary


def figure4(run_dir, args, out):
    tdir = os.path.join(run_dir, "trajectories")
    labels_file = os.path.join(run_dir, "clusters", f"labels_{args.order}.npz")
    if not os.path.exists(labels_file):
        print("Figure 4 skipped: run steps 3 and 4 (trajectories.py, cluster.py) first")
        return
    T = np.load(os.path.join(tdir, f"trajectories_{args.order}.npz"))
    traj, steps, place, carry, basis_steps = T["traj"], T["steps"], T["place"], T["carry"], T["basis_steps"]
    loss = np.load(os.path.join(tdir, "projections.npz"))["loss"]
    labels = np.load(labels_file)["labels"]

    fig, axes = plt.subplots(len(args.vectors), 3, figsize=(16, 3.9 * len(args.vectors)), squeeze=False)
    for row, k in enumerate(args.vectors):
        ax_traj, ax_loss, ax_bar = axes[row]
        letters = "abcdefghijklmnopqrstuvwxyz"[3 * row : 3 * row + 3]
        lab = labels[:, k]
        ids = sorted(set(lab) - {-1, -2})
        name = f"vector #{k + 1} (step {basis_steps[k]})"
        if not ids:
            for ax in axes[row]:
                ax.axis("off")
            ax_traj.set_title(f"{name}: no clusters (vector filtered out or too few tokens)")
            continue
        for i, c in enumerate(ids):
            m = lab == c
            band(ax_traj, steps, traj[m, :, k], CLUSTER_COLORS(i % 10), str(i + 1))
            band(ax_loss, steps, loss[m], CLUSTER_COLORS(i % 10), str(i + 1))
        ax_traj.axvline(basis_steps[k], color="gray", linestyle="--", linewidth=1)
        ax_traj.set_title(f"({letters[0]}) Median projected loss, {name}")
        ax_traj.set_ylabel(f"{T['order']}-order POLCA")
        ax_loss.set_title(f"({letters[1]}) Median exact loss, same clusters")
        ax_loss.set_ylabel("Exact loss")
        for ax in (ax_traj, ax_loss):
            ax.set_xlim(0, args.max_step)
            ax.set_xlabel("Iteration")
            ax.legend(title="Cluster", fontsize=8)
        homogeneity = ", ".join(f"{carry[lab == c].mean():.2f}" for c in ids)
        composition(ax_bar, lab, ids, place, carry, f"({letters[2]}) Composition; carry homogeneity {homogeneity}")
        describe(f"Figure 4: {name},", lab, ids, place, carry)
    fig.tight_layout()
    name = "figure4.png" if args.vectors == [0, 1] else f"figure4_v{'_'.join(map(str, args.vectors))}.png"
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)


def figure7(steps, losses, place, carry, out):
    for split_carry, name in ((False, "figure7.png"), (True, "figure7_carry.png")):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
        for p, ax in enumerate(axes.flat):
            m = place == p
            if not split_carry:
                band(ax, steps, losses[m], COLORS[p], f"{PLACES[p]} (n={m.sum()})")
            else:
                for c, color, linestyle, label in ((0, COLORS[p], "-", "No carry"), (1, "black", "--", "Carry")):
                    mc = m & (carry == c)
                    if mc.any():
                        band(ax, steps, losses[mc], color, f"{label}, {PLACES[p]} (n={mc.sum()})", linestyle)
            ax.set_title(f"({'abcd'[p]}) {PLACES[p]} place")
            ax.set_xlim(0, steps[-1])
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Exact loss")
            ax.legend(fontsize=8)
        what = "each digit and carry combination" if split_carry else "each digit"
        fig.suptitle(f"Median and quartiles of the exact-loss trajectories for {what}")
        fig.tight_layout()
        fig.savefig(os.path.join(out, name), dpi=150)
        plt.close(fig)


def table8(run_dir, exact, out):
    """Markdown Table 8 for exact loss (Figure 3 clusters) and both POLCA orders (step 5)."""
    results = {"Exact loss": exact}
    for name, order in (("POLCA", "second"), ("POLCA (first order)", "first")):
        path = os.path.join(run_dir, "stats", f"summary_{order}.json")
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
        else:
            print(f"Table 8: {path} missing (run steps 3-5); row left out")

    def cells(rows):
        return "<br>".join(f"{r['homogeneity']:.3f} ({r['recall']:.2f}, {r['f1']:.2f})" for r in rows) or "-"

    lines = [
        "# Table 8: carry homogeneity of the clusters", "",
        "Top 3 clusters as homogeneity (recall, F1). Appendix G filter: only clusters whose carry tokens are",
        "> 85% in the 10s/100s place. No filter: as the released code scores; the paper's numbers match this.", "",
        "| Trajectory | Top 3, Appendix G filter | Top 3, no filter | Hidden | Paper top 3 | Paper hidden |",
        "|---|---|---|---:|---|---:|",
    ]  # fmt: skip
    for name, r in results.items():
        paper_rows, paper_hidden = PAPER_TABLE8[name]
        paper = "<br>".join(f"{h:.3f} ({rec:.2f}, {f1:.2f})" for h, rec, f1 in paper_rows)
        lines.append(
            f"| {name} | {cells(r['top3_appendix_g'])} | {cells(r['top3_unfiltered'])} | "
            f"{r['hidden_fraction']:.3f} | {paper} | {paper_hidden:.3f} |"
        )
    with open(os.path.join(out, "table8.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--figures", nargs="+", default=["3", "4", "7", "table8"], choices=["3", "4", "7", "table8"])
    ap.add_argument("--order", choices=["first", "second"], default="second", help="Figure 4 trajectory type")
    ap.add_argument("--vectors", type=int, nargs="+", default=[0, 1], help="Figure 4 basis vectors (rows), 0-based")
    ap.add_argument("--max_step", type=int, default=4000, help="x-axis limit of Figures 3 and 4 (the paper's)")
    ap.add_argument("--min_cluster_size", type=int, default=150, help="Figure 3 clustering")
    ap.add_argument("--min_samples", type=int, default=0, help="Figure 3 clustering; 0 = tokens // 15 (paper)")
    ap.add_argument("--delta", type=int, default=100, help="breakthrough window D in steps (Figure 3 scores)")
    ap.add_argument("--tau", type=int, default=1000, help="hidden-breakthrough threshold (Figure 3 scores)")
    a = ap.parse_args()

    out = os.path.join(a.run_dir, "figures")
    os.makedirs(out, exist_ok=True)
    lab = np.load(os.path.join(a.run_dir, "test_labels.npz"))
    place, carry = lab["place"].reshape(-1), lab["carry"].reshape(-1)
    exact = None
    if {"3", "7", "table8"} & set(a.figures):
        steps, losses = exact_losses(a.run_dir, out)
        if "7" in a.figures:
            figure7(steps, losses, place, carry, out)
        if {"3", "table8"} & set(a.figures):  # Table 8's exact-loss row is Figure 3's clustering
            exact = figure3(steps, losses, place, carry, a, out)
    if "4" in a.figures:
        figure4(a.run_dir, a, out)
    if "table8" in a.figures:
        table8(a.run_dir, exact, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
