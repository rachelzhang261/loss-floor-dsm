"""
train.py -- train one diffusion model on CIFAR-10 over a given log-SNR range.

The quality axis of the experiment comes for free from CHECKPOINTS: one run per
configuration, saved at several step counts, gives a whole ladder of model
qualities at no extra compute.  Only one run per SNR configuration is therefore
needed.

    python train.py --config wide   --steps 60000 --out runs/wide
    python train.py --config narrow --steps 60000 --out runs/narrow

Smoke test (seconds, no CIFAR download needed):
    python train.py --smoke
"""

import argparse
import json
import os
import time

import torch

from common import (CONFIGS, TensorLoader, UNet, cifar_tensors, count_params,
                    dsm_loss_terms)

CKPT_STEPS = [2000, 5000, 10000, 20000, 35000, 60000]


def save(model, kwargs, step, cfg, out):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"step{step:06d}.pt")
    torch.save({"model": model.state_dict(), "model_kwargs": kwargs,
                "step": step, "config": cfg}, path)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="wide", choices=list(CONFIGS))
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="runs/wide")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end run on random data")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device if torch.cuda.is_available() else "cpu"

    kwargs = dict(ch=args.ch, ch_mult=(1, 2, 2), nres=2, attn_res=(16,),
                  dropout=0.1)
    if args.smoke:
        kwargs = dict(ch=32, ch_mult=(1, 2), nres=1, attn_res=(16,), dropout=0.0)

    model = UNet(**kwargs).to(dev)
    print(f"device={dev}  params={count_params(model)/1e6:.2f}M  "
          f"config={args.config} {CONFIGS[args.config]}")

    if args.smoke:
        x = torch.randn(4, 3, 32, 32, device=dev)
        lo, hi = CONFIGS[args.config]
        loss = dsm_loss_terms(model, x, lo, hi).mean()
        loss.backward()
        gn = sum(p.grad.pow(2).sum().item() for p in model.parameters()
                 if p.grad is not None) ** 0.5
        print(f"smoke: forward+backward OK, loss={loss.item():.3f}, "
              f"grad-norm={gn:.3e}")
        from floor_est import integrand_at
        for pr in model.parameters():
            pr.requires_grad_(False)
        v = integrand_at(model, x, 0.0, n_probe=2)
        print(f"smoke: floor integrand at lambda=0 -> {v.mean().item():.1f} "
              f"(must lie in [0, 3072])")
        assert 0.0 <= v.mean().item() <= 3072.0 * 1.05
        print("smoke OK")
        return

    data = cifar_tensors(args.data, train=True)
    loader = TensorLoader(data, args.batch, seed=args.seed, device=dev)
    lo, hi = CONFIGS[args.config]

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ckpts = [s for s in CKPT_STEPS if s <= args.steps]
    if args.steps not in ckpts:
        ckpts.append(args.steps)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"config": args.config, "range": CONFIGS[args.config],
                   "model_kwargs": kwargs, "args": vars(args)}, f, indent=2)

    model.train()
    t0, running = time.time(), 0.0
    for step in range(1, args.steps + 1):
        for gparam in opt.param_groups:
            gparam["lr"] = args.lr * min(1.0, step / max(args.warmup, 1))
        y = next(loader)
        loss = dsm_loss_terms(model, y, lo, hi).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running += loss.item()
        if step % 500 == 0:
            el = time.time() - t0
            print(f"step {step:6d}/{args.steps}  loss {running/500:10.1f}  "
                  f"{step/el:5.1f} it/s  eta {(args.steps-step)/(step/el)/60:5.1f} min",
                  flush=True)
            running = 0.0
        if step in ckpts:
            print("saved", save(model, kwargs, step, args.config, args.out))


if __name__ == "__main__":
    main()
