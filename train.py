"""Step 1: train the arithmetic model and save checkpoints.

Released settings: AdamW (lr 1e-5, betas (0.9, 0.95), weight decay 0.1), 100-step linear
warmup, batch size 64, 10,000 steps, loss on the 4 answer tokens. The test set is the first
1,250-problem draw of the data generator with the same seed.

Outputs in RUN_DIR:
  config.json                          settings
  checkpoints/model_init.pt            weights before training
  checkpoints/model_step_{e}.pt        weights after loop iteration e, every --ckpt_interval steps
                                       and at the last step (e = 0, 20, ..., 9980, 9999)
  test_tokens.npy, test_labels.npz     test problems (1,250 x 12) and per-token place / carry labels
  train_losses.npy                     per-step training loss of each answer place (steps, 4)

Usage:
  python train.py RUN_DIR [--seed 42] [--num_steps 10000] [--ckpt_interval 20]
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from common import (
    BATCH_SIZE,
    BETAS,
    LR,
    MODEL_CFG,
    TEST_SIZE,
    WARMUP,
    WEIGHT_DECAY,
    checkpoint_path,
    data_generator,
    device,
    loss_fn,
    make_model,
    save_checkpoint,
    token_labels,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--seed", type=int, default=42, help="seed of the data (test set and training batches)")
    ap.add_argument("--init_seed", type=int, default=None, help="seed of the initial weights (default: --seed)")
    ap.add_argument("--num_steps", type=int, default=10000)
    ap.add_argument("--ckpt_interval", type=int, default=20, help="must divide the POLCA and Hessian intervals")
    a = ap.parse_args()
    init_seed = a.seed if a.init_seed is None else a.init_seed
    os.makedirs(os.path.join(a.run_dir, "checkpoints"), exist_ok=True)

    torch.manual_seed(init_seed)
    model = make_model().to(device)
    save_checkpoint(model, os.path.join(a.run_dir, "checkpoints", "model_init.pt"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / WARMUP, 1.0))

    test_tokens, _ = next(data_generator(TEST_SIZE, seed=a.seed))
    np.save(os.path.join(a.run_dir, "test_tokens.npy"), test_tokens.cpu().numpy())
    place, carry = token_labels(test_tokens)
    np.savez(os.path.join(a.run_dir, "test_labels.npz"), place=place, carry=carry)

    config = {
        "seed": a.seed, "init_seed": init_seed, "num_steps": a.num_steps, "ckpt_interval": a.ckpt_interval,
        "batch_size": BATCH_SIZE, "test_size": TEST_SIZE, "lr": LR, "betas": BETAS, "weight_decay": WEIGHT_DECAY,
        "warmup": WARMUP, "model_cfg": MODEL_CFG, "n_params": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__, "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
    }  # fmt: skip
    with open(os.path.join(a.run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    batches = data_generator(BATCH_SIZE, seed=a.seed)  # reseeds the global RNG, as the released code
    train_losses = np.zeros((a.num_steps, 4), dtype=np.float32)
    start = time.time()
    for step in range(a.num_steps):
        tokens, _ = next(batches)
        per_place = loss_fn(model(tokens), tokens, per_token=True).mean(0)
        per_place.mean().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        train_losses[step] = per_place.detach().cpu().numpy()
        if step % a.ckpt_interval == 0 or step == a.num_steps - 1:
            save_checkpoint(model, checkpoint_path(a.run_dir, step))
        if step % 500 == 0:
            print(
                f"step {step}: train loss {per_place.mean().item():.4f}, per place "
                f"{np.round(train_losses[step], 4).tolist()}, {time.time() - start:.0f}s",
                flush=True,
            )

    np.save(os.path.join(a.run_dir, "train_losses.npy"), train_losses)
    print(f"done: {a.num_steps} steps in {time.time() - start:.0f}s; checkpoints in {a.run_dir}/checkpoints")


if __name__ == "__main__":
    main()
