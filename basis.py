"""Step 2: the POLCA basis, by the paper's Algorithm 1.

At T = 50 checkpoints, every 200 steps (steps 200, 400, ..., 9,800 and the last step 9,999), the
basis B gains one row: the top eigenvector of Pi H, where
  H  = Hessian of the mean answer-token loss on a fresh sample of 1,250 problems, and
  Pi = I - B^T (B B^T)^-1 B projects out the directions already in B.
So each new row is the most curved direction that the earlier rows do not already cover. The
eigenvalue of Pi H is saved with it (POLCA's second-order term uses it).

Hessian samples: as in the released code, each Hessian gets the next 1,250-problem draw of the
data generator that produced the test set (data_generator(1250, seed)), right after the test set.

Eigensolver: power iteration as in hessian_eigenthings 0.0.2, which the released code uses
(at most 20 steps, stop when the eigenvalue estimate changes by less than 1e-4 relative). The
Hessian-vector product is averaged over the batches of 64 of the sample, one pass per product.
The projection and normalisation are done in float64. The code below reproduces the released
library's random numbers, so with the same checkpoints and --seed it gives the same basis.

Outputs in RUN_DIR/basis/:
  basis.npy        (50, 9475594) float32, one unit row per checkpoint, in model.parameters() order
  eigenvalues.npy  (50,) eigenvalue of Pi H for each row
  basis_steps.npy  (50,) training step of each row
  info.json        settings, timings and how far B is from orthonormal

Usage:
  python basis.py RUN_DIR [--n_hessians 50] [--seed 0]
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import (
    BATCH_SIZE,
    TEST_SIZE,
    checkpoint_path,
    data_generator,
    device,
    load_checkpoint,
    load_config,
    loss_fn,
)


class HessianVectorProduct:
    """v -> H v, with H the Hessian of the mean answer-token loss, averaged over the batches of a
    shuffled loader (one full pass per product), as hessian_eigenthings' HVPOperator."""

    def __init__(self, model, loader):
        self.model, self.loader = model, loader
        self.params = list(model.parameters())
        self.size = sum(p.numel() for p in self.params)
        self.batches = iter(loader)

    def next_batch(self):
        try:
            return next(self.batches)
        except StopIteration:
            self.batches = iter(self.loader)
            return next(self.batches)

    def __call__(self, v):
        total = None
        for _ in range(len(self.loader)):
            tokens, _ = self.next_batch()
            tokens = tokens.to(device)
            grads = torch.autograd.grad(loss_fn(self.model(tokens), tokens), self.params, create_graph=True)
            grad = torch.cat([g.contiguous().view(-1) for g in grads])
            hv = torch.autograd.grad(grad, self.params, grad_outputs=v, only_inputs=True)
            hv = torch.cat([h.contiguous().view(-1) for h in hv])
            total = hv if total is None else total + hv
        return total / len(self.loader)


def power_iteration(apply, size, steps=20, tol=1e-4):
    """Top eigenpair of the operator `apply` (hessian_eigenthings 0.0.2's power_iteration)."""
    vec = torch.rand(size).to(device)
    prev_lambda = 0.0
    for _ in range(steps):
        new_vec = apply(vec)
        if torch.norm(new_vec).item() == 0.0:
            return 0.0, new_vec
        lam = vec.dot(new_vec).item()
        vec = new_vec.detach() / torch.norm(new_vec)
        error = 1.0 if lam == 0.0 else abs((lam - prev_lambda) / lam)
        if error < tol:
            break
        prev_lambda = lam
    return lam, vec


def nullspace_projector(B):
    """v -> (I - B^T (B B^T)^-1 B) v, in float64. (B B^T)^-1 B is solved once, on the CPU."""
    B = B.to(torch.float64)
    M = torch.linalg.lstsq(B @ B.T, B).solution.to(device)
    B = B.to(device)
    return lambda v: (v.double() - B.T @ (M @ v.double())).float()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--n_hessians", type=int, default=50, help="basis size T; a Hessian every num_steps / T steps")
    ap.add_argument("--seed", type=int, default=0, help="seed of the loader shuffling and power-iteration starts")
    a = ap.parse_args()

    cfg = load_config(a.run_dir)
    num_steps, interval = cfg["num_steps"], cfg["num_steps"] // a.n_hessians
    steps = [min(t * interval, num_steps - 1) for t in range(1, a.n_hessians + 1)]
    missing = [s for s in steps if not os.path.exists(checkpoint_path(a.run_dir, s))]
    if missing:
        raise SystemExit(
            f"no checkpoint for steps {missing[:3]}...: the Hessian interval {interval} must be a "
            f"multiple of the run's ckpt_interval {cfg['ckpt_interval']}"
        )
    out = os.path.join(a.run_dir, "basis")
    os.makedirs(out, exist_ok=True)

    # Hessian samples: the draws after the test set. Taken before any model is built, because
    # building a model advances the global RNG.
    gen = data_generator(TEST_SIZE, seed=cfg["seed"])
    test_tokens, _ = next(gen)
    assert np.array_equal(test_tokens.cpu().numpy(), np.load(os.path.join(a.run_dir, "test_tokens.npy")))
    samples = [next(gen)[0] for _ in steps]

    torch.manual_seed(a.seed)
    print(f"{len(steps)} Hessians at steps {steps[0]}, {steps[1]}, ..., {steps[-1]}; output {out}", flush=True)
    B, eigenvalues, seconds, start = None, [], [], time.time()
    for j, (step, sample) in enumerate(zip(steps, samples, strict=True)):
        t0 = time.time()
        model = load_checkpoint(checkpoint_path(a.run_dir, step))
        loader = DataLoader(TensorDataset(sample, sample), BATCH_SIZE, shuffle=True, num_workers=0)
        hvp = HessianVectorProduct(model, loader)
        if B is None:
            apply = hvp
        else:
            project = nullspace_projector(B)

            def apply(v, hvp=hvp, project=project):
                return project(hvp(v))  # Pi H v

        lam, vec = power_iteration(apply, hvp.size)
        row = torch.nn.functional.normalize(vec.cpu()[None].double()).float()
        B = row if B is None else torch.cat([B, row])
        eigenvalues.append(lam)
        seconds.append(time.time() - t0)
        print(f"[{j + 1}/{len(steps)}] step {step}: eigenvalue {lam:.4g}, {seconds[-1]:.1f}s", flush=True)
        del model, hvp, apply
        if device == "cuda":
            torch.cuda.empty_cache()

    np.save(os.path.join(out, "basis.npy"), B.numpy())
    np.save(os.path.join(out, "eigenvalues.npy"), np.array(eigenvalues))
    np.save(os.path.join(out, "basis_steps.npy"), np.array(steps))
    gram = (B.to(device, torch.float64) @ B.to(device, torch.float64).T).cpu().numpy()
    info = {
        "args": vars(a), "steps": steps, "shape": list(B.shape), "eigenvalues": eigenvalues,
        "seconds_per_hessian": seconds, "total_seconds": time.time() - start,
        "max_abs_gram_minus_identity": float(np.abs(gram - np.eye(len(gram))).max()),
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
    }  # fmt: skip
    with open(os.path.join(out, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(
        f"done: basis {tuple(B.shape)} in {info['total_seconds']:.0f}s, max |B B^T - I| "
        f"{info['max_abs_gram_minus_identity']:.1e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
