"""Paired high-SNR comparison with the integer-pixel denoisers."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import CONFIGS, alpha_sigma, cifar_tensors
from evaluate import ROUNDING_START, integrated_stats, load
from floor_core import log_snr_grid


def lattice_predictors(x, sigma, radius=2):
    """Predict noise for x = y + sigma * eps on the 8-bit pixel lattice."""
    u = 127.5 * (x.double() + 1)
    r = 127.5 * sigma
    nearest = u.round().clamp(0, 255)
    hard = (u - nearest) / r
    offsets = torch.arange(-radius, radius + 1, device=x.device)
    candidates = nearest[..., None] + offsets
    delta = (u[..., None] - candidates) / r
    allowed = (candidates >= 0) & (candidates <= 255)
    weights = (-0.5 * delta.square()).masked_fill(~allowed, -torch.inf).softmax(-1)
    return hard, (weights * delta).sum(-1)


@torch.no_grad()
def evaluate(model, data, grid, device, batch=50, batches=8, seed=2027):
    noise = torch.Generator(device=device).manual_seed(seed)
    indices = torch.Generator().manual_seed(seed)
    curves = {}
    tail = {}

    for lam in grid:
        time = torch.full((batch,), float(lam), device=device)
        a, s = alpha_sigma(time[:1])
        samples = []
        for _ in range(batches):
            y = data[torch.randint(len(data), (batch,), generator=indices)].to(device)
            eps = torch.randn(y.shape, device=device, generator=noise)
            x = a * y + s * eps
            model_loss = (eps - model(x, time)).square().flatten(1).sum(1)
            if lam < ROUNDING_START:
                samples.append(model_loss[:, None].cpu())
                continue

            # At lambda >= 13 the noise SD in pixel-index units is < 0.192.
            # Omitted candidates are at least 2.5 pixel spacings away.
            hard, soft = lattice_predictors(x.double() / a.double(), s.double() / a.double())
            hard_loss = (eps - hard).square().flatten(1).sum(1)
            soft_loss = (eps - soft).square().flatten(1).sum(1)
            samples.append(torch.stack((model_loss, hard_loss, soft_loss,
                                        model_loss - hard_loss, model_loss - soft_loss), 1).cpu())

        values = torch.cat(samples).double()
        mean = values.mean(dim=0).tolist()
        se = (values.std(dim=0) / len(values)**0.5).tolist()
        curves[float(lam)] = (mean[0], se[0])
        if lam >= ROUNDING_START:
            tail[float(lam)] = (mean, se)
    return curves, tail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()

    model, cfg, step = load(args.ckpt, args.device)
    lo, hi = CONFIGS[cfg]
    if not lo < ROUNDING_START < hi:
        parser.error("checkpoint must cover the rounding comparison interval")

    grid = log_snr_grid(lo, hi, 219, [lo, ROUNDING_START, hi])
    curves, tail = evaluate(model, cifar_tensors(args.data, train=False), grid, args.device)
    tail_grid = np.asarray(list(tail))
    means, errors = map(np.asarray, zip(*tail.values()))
    
    fields = ("network", "rounding", "soft_discrete", "network_minus_rounding",
              "network_minus_soft_discrete")
    
    summary = {field: integrated_stats(tail_grid, means[:, i], errors[:, i], ROUNDING_START, hi)
               for i, field in enumerate(fields)}
    
    result = {
        "checkpoint": f"{cfg}@{step}", "grid": grid.tolist(),
        "network": np.asarray(list(curves.values())).tolist(),
        "tail_grid": tail_grid.tolist(), "tail_fields": fields,
        "tail_mean": means.tolist(), "tail_mc_se": errors.tolist(), "tail_integrals": summary,
        "images_per_node": 400, "seed": 2027,
    }
    
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
