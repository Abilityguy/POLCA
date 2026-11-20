"""Step 3: POLCA trajectories for every test token and basis vector, then the paper's filters.

1. Projections (GPU). At every POLCA checkpoint t = 0, 20, ..., 9,980, for the 5,000 test answer
   tokens x and every basis vector b_k:
     g[x, t, k] = <b_k, grad L(x; theta_t)>      gradient along b_k (forward-mode JVP)
     d[t, k]    = <b_k, theta_t - theta_{t-1}>   weight step along b_k (0 at the first checkpoint)
     loss[x, t] = L(x; theta_t)
   Saved to projections.npz. If that file exists it is reused (--recompute to redo it).

2. Trajectories (paper Eqs. 2-8), cumulated over checkpoints, for each --orders (default both):
     first   L_k(x, t) = sum_{i<t} g[x, i, k] d[i+1, k]
     second  first order plus, for each interval i -> i+1,
             1/2 lam_k d[i+1, k]^2 (g[x, i+1, k] - g[x, i, k]) / (G[i+1, k] - G[i, k]),
             with G[t, k] = sum_x g[x, t, k] and lam_k the basis eigenvalue (the paper's POLCA;
             its Table 8 also reports first order)

3. Filters (Section 4.1): a basis vector is kept only if the mean trajectory decreases over
   training (mean_x L_k(x, T) < 0); within a kept vector, a token is kept only if its own
   trajectory decreases (L_k(x, T) < 0).

Outputs in RUN_DIR/trajectories/:
  projections.npz   steps, grad_proj (tokens, T, K) float32, delta_proj (T, K), loss (tokens, T)
  trajectories_{order}.npz  traj (tokens, T, K) float64, keep (tokens, K) bool, vector_kept (K,)
                    bool, steps, place, carry, eigenvalues, basis_steps, order (about 1 GB each)
  info.json

Usage:
  python trajectories.py RUN_DIR [--orders first second] [--interval 20] [--recompute]
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.func import functional_call, jvp

from common import checkpoint_path, device, load_checkpoint, load_config, loss_fn


def project(model, tokens, basis):
    """Per-token gradients along each basis row (tokens, K), the per-token losses, and the flat weights."""
    names = [n for n, _ in model.named_parameters()]
    params = tuple(p.detach() for _, p in model.named_parameters())
    sizes = [p.numel() for p in params]

    def per_token_loss(*ps):
        return loss_fn(functional_call(model, dict(zip(names, ps, strict=True)), (tokens,)), tokens, True).reshape(-1)

    grads = torch.empty(tokens.shape[0] * 4, len(basis), dtype=torch.float64, device=device)
    with torch.no_grad():
        for k in range(len(basis)):
            tangent = tuple(t.view_as(p) for t, p in zip(torch.split(basis[k], sizes), params, strict=True))
            loss, grads[:, k] = jvp(per_token_loss, params, tangent)
    return grads, loss, torch.cat([p.reshape(-1) for p in params])


def compute_projections(run_dir, basis, steps):
    tokens = torch.tensor(np.load(os.path.join(run_dir, "test_tokens.npy"))).to(device)
    n_tokens = tokens.shape[0] * 4

    # Check the JVP against autograd per-token gradients on a few tokens.
    model = load_checkpoint(checkpoint_path(run_dir, steps[0]))
    grads, _, _ = project(model, tokens, basis)
    losses = loss_fn(model(tokens[:4]), tokens[:4], per_token=True).reshape(-1)
    err = 0.0
    for i in range(len(losses)):
        g = torch.autograd.grad(losses[i], list(model.parameters()), retain_graph=True)
        ref = basis.double() @ torch.cat([x.reshape(-1) for x in g]).double()
        err = max(err, ((grads[i] - ref).abs().max() / ref.abs().max()).item())
    print(f"check: JVP vs autograd per-token gradients, max relative error {err:.1e}", flush=True)

    grad_proj = np.zeros((n_tokens, len(steps), len(basis)), dtype=np.float32)
    delta_proj = np.zeros((len(steps), len(basis)))
    loss = np.zeros((n_tokens, len(steps)), dtype=np.float32)
    basis64, prev, start = basis.double(), None, time.time()
    for t, step in enumerate(steps):
        model = load_checkpoint(checkpoint_path(run_dir, step))
        grads, step_loss, theta = project(model, tokens, basis)
        grad_proj[:, t], loss[:, t] = grads.cpu().numpy(), step_loss.cpu().numpy()
        if prev is not None:
            delta_proj[t] = (basis64 @ (theta.double() - prev)).cpu().numpy()
        prev = theta.double()
        if t % 50 == 0:
            print(f"projections at step {step} ({t + 1}/{len(steps)}), {time.time() - start:.0f}s", flush=True)
    return grad_proj, delta_proj, loss, err


def build(order, g, d, lam):
    """Trajectory (tokens, T) of one basis vector. g: (tokens, T), d: (T,), lam: eigenvalue."""
    increment = np.zeros_like(g)
    increment[:, 1:] = g[:, :-1] * d[None, 1:]  # interval i -> i+1, stored at i+1
    if order == "first":
        return np.cumsum(increment, axis=1)
    G = g.sum(0)
    dG = np.zeros(len(G))
    dG[1:] = G[1:] - G[:-1]
    ok = np.abs(dG) > 1e-12
    ratio = np.zeros_like(g)
    ratio[:, 1:] = (g[:, 1:] - g[:, :-1]) / np.where(ok, dG, 1.0)[None, 1:]
    ratio[:, ~ok] = 0.0
    return np.cumsum(increment + 0.5 * lam * d[None] ** 2 * ratio, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument(
        "--orders",
        nargs="+",
        choices=["first", "second"],
        default=["first", "second"],
        help="trajectory types to build (paper's POLCA: second)",
    )
    ap.add_argument("--interval", type=int, default=20, help="steps between POLCA checkpoints")
    ap.add_argument("--recompute", action="store_true", help="recompute projections.npz even if it exists")
    a = ap.parse_args()

    cfg = load_config(a.run_dir)
    if a.interval % cfg["ckpt_interval"]:
        raise SystemExit(
            f"--interval {a.interval} must be a multiple of the run's ckpt_interval {cfg['ckpt_interval']}"
        )
    basis_dir = os.path.join(a.run_dir, "basis")
    eigenvalues = np.load(os.path.join(basis_dir, "eigenvalues.npy"))
    basis_steps = np.load(os.path.join(basis_dir, "basis_steps.npy"))
    labels = np.load(os.path.join(a.run_dir, "test_labels.npz"))
    place, carry = labels["place"].reshape(-1), labels["carry"].reshape(-1)
    out = os.path.join(a.run_dir, "trajectories")
    os.makedirs(out, exist_ok=True)
    steps = np.arange(0, cfg["num_steps"], a.interval)

    proj_file = os.path.join(out, "projections.npz")
    jvp_err = None
    if os.path.exists(proj_file) and not a.recompute:
        print(f"reusing {proj_file}", flush=True)
        P = np.load(proj_file)
        assert np.array_equal(P["steps"], steps), "projections.npz has other steps; use --recompute"
        grad_proj, delta_proj, loss = P["grad_proj"], P["delta_proj"], P["loss"]
    else:
        basis = torch.tensor(np.load(os.path.join(basis_dir, "basis.npy")), device=device)
        print(f"projections: {len(steps)} checkpoints, {len(basis)} basis vectors", flush=True)
        grad_proj, delta_proj, loss, jvp_err = compute_projections(a.run_dir, basis, steps)
        np.savez(proj_file, steps=steps, grad_proj=grad_proj, delta_proj=delta_proj, loss=loss)

    n_tokens, _, K = grad_proj.shape
    info = {"args": vars(a), "tokens": n_tokens, "checkpoints": len(steps), "vectors": K,
            "jvp_check_max_rel_err": jvp_err}  # fmt: skip
    for order in a.orders:
        traj = np.zeros((n_tokens, len(steps), K))
        keep = np.zeros((n_tokens, K), dtype=bool)
        vector_kept = np.zeros(K, dtype=bool)
        for k in range(K):
            traj[:, :, k] = build(order, grad_proj[:, :, k].astype(np.float64), delta_proj[:, k], float(eigenvalues[k]))
            final = traj[:, -1, k]
            vector_kept[k] = final.mean() < 0
            keep[:, k] = vector_kept[k] & (final < 0)
        print(
            f"{order}-order trajectories: {vector_kept.sum()}/{K} vectors kept; tokens kept per kept vector "
            f"{keep[:, vector_kept].sum(0).min() if vector_kept.any() else 0}-{keep.sum(0).max()}",
            flush=True,
        )
        path = os.path.join(out, f"trajectories_{order}.npz")
        np.savez(
            path,
            traj=traj,
            keep=keep,
            vector_kept=vector_kept,
            steps=steps,
            place=place,
            carry=carry,
            eigenvalues=eigenvalues,
            basis_steps=basis_steps,
            order=order,
        )
        info[order] = {"vectors_kept": int(vector_kept.sum()), "tokens_kept": keep.sum(0).tolist()}
        print(f"written {path}", flush=True)
        del traj
    with open(os.path.join(out, "info.json"), "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
