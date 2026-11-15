"""
floor_est.py -- model-based loss floor via the Hutchinson trace.

    floor_integrand(l) = d - sigma(l) * E_x [ tr( d eps_theta / dx ) ]
    FLOOR[lo,hi]       = 1/2 * int_lo^hi floor_integrand(l) dl

The identity is derived and validated in floor_jacobian.py (agreement with the
exact posterior covariance to ~1e-11 on a Gaussian mixture).

tr J is estimated with Rademacher probes using a vector-Jacobian product:
    v^T J  = autograd.grad(eps_theta, x, grad_outputs=v)
    tr J  ~ mean over v of  <v, v^T J>
The trace estimate averages over inputs and probes. The resulting floor
estimate is exact in expectation for the Bayes denoiser.
"""

import argparse
import json
import os

import numpy as np
import torch

from common import CONFIGS, alpha_sigma, cifar_tensors
from floor_core import integrate_floor as integrate, log_snr_grid


@torch.enable_grad()
def integrand_at(model, y, lam_val, n_probe=1, generator=None, *, paired=False,
                 trace_mode="vjp"):
    """
    Estimate  d - sigma * E[tr d eps/dx]  at a single log-SNR.
    y : (B,3,32,32) clean images on the right device
    """
    device = y.device
    B = y.shape[0]
    D = int(np.prod(y.shape[1:]))
    lam = torch.full((B,), float(lam_val), device=device)
    a, s = alpha_sigma(lam)
    a4 = a.view(-1, 1, 1, 1)
    s4 = s.view(-1, 1, 1, 1)

    eps = torch.randn(y.shape, device=device, generator=generator)
    x = (a4 * y + s4 * eps).detach()
    if trace_mode == "vjp":
        x.requires_grad_(True)
        pred = model(x, lam)

    tr = torch.zeros_like(s)
    for k in range(n_probe):
        # Rademacher probe via torch.rand (accepts a device generator on all
        # backends; torch.randint's dtype handling is less portable)
        u = torch.rand(y.shape, device=device, generator=generator)
        v = (u < 0.5).to(y.dtype).mul_(-2.0).add_(1.0)   # +-1, no scalar where
        if trace_mode == "vjp":
            (product,) = torch.autograd.grad(pred, x, grad_outputs=v,
                                             retain_graph=k < n_probe - 1)
        else:
            with torch.no_grad():
                pred, product = torch.func.jvp(lambda z: model(z, lam), (x,), (v,))
        tr += (v * product).flatten(1).sum(1)
    tr /= n_probe

    jacobian = (D - s * tr).detach()
    if not paired:
        return jacobian

    pred = pred.detach()
    dsm = (eps - pred).square().flatten(1).sum(1)
    # Conditional Gaussian integration by parts gives E[eps^T pred] = sigma E[tr J].
    stein = D - (eps * pred).flatten(1).sum(1)
    return torch.stack((jacobian, dsm, stein, dsm - jacobian, stein - jacobian), dim=1)


def floor_curve(model, data, lams, batch=64, n_batches=8, n_probe=1,
                seed=0, device="cuda", *, paired=False, trace_mode="vjp"):
    """Return node means and standard errors, with paired diagnostics on request."""
    g = torch.Generator(device=device).manual_seed(seed)
    idx_g = torch.Generator().manual_seed(seed + 1)
    means, errs = [], []
    for lam in lams:
        vals = []
        for _ in range(n_batches):
            idx = torch.randint(0, len(data), (batch,), generator=idx_g)
            y = data[idx].to(device)
            vals.append(integrand_at(model, y, lam, n_probe, g, paired=paired,
                                     trace_mode=trace_mode).cpu())
        v = torch.cat(vals).double()
        means.append(v.mean(dim=0).numpy())
        errs.append((v.std(dim=0, unbiased=True) / np.sqrt(len(v))).numpy())
    return np.array(means), np.array(errs)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True,
                   help="one or more checkpoints to compare")
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="results/floor_curves.json")
    p.add_argument("--n-grid", type=int, default=29)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--n-probe", type=int, default=1)
    p.add_argument("--trace-mode", choices=("vjp", "jvp"), default="vjp")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if (args.batch < 1 or args.n_batches < 1 or args.batch * args.n_batches < 2
            or args.n_probe < 1 or args.n_grid < 2):
        p.error("need at least two grid nodes and floor samples, and positive batch/probe counts")

    from common import UNet
    data = cifar_tensors(args.data, train=False)      # held-out images

    lo = min(v[0] for v in CONFIGS.values())
    hi = max(v[1] for v in CONFIGS.values())
    lams = log_snr_grid(lo, hi, args.n_grid, [e for r in CONFIGS.values() for e in r])

    out = {"lams": lams.tolist(), "curves": {}, "configs": CONFIGS}
    for ck in args.ckpt:
        blob = torch.load(ck, map_location="cpu", weights_only=True)
        model = UNet(**blob["model_kwargs"]).to(args.device)
        model.load_state_dict(blob["model"])
        model.eval()
        for prm in model.parameters():
            prm.requires_grad_(False)      # we differentiate wrt x, not params

        mu, se = floor_curve(model, data, lams, args.batch, args.n_batches,
                             args.n_probe, seed=0, device=args.device,
                             trace_mode=args.trace_mode)
        cfg, step = blob["config"], blob["step"]
        name = f"{cfg}@{step}"
        if name in out["curves"]:
            p.error(f"duplicate checkpoint configuration and step: {name}")
        out["curves"][name] = {"mean": mu.tolist(), "stderr": se.tolist(), "config": cfg}
        trained_lo, trained_hi = CONFIGS[cfg]
        line = "  ".join(f"{c}={integrate(lams, mu, *r):.1f}"
                         for c, r in CONFIGS.items() if trained_lo <= r[0] and r[1] <= trained_hi)
        print(f"[{name}]  floor per config:  {line}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")

    # model-independence check
    names = list(out["curves"])
    if len(names) > 1:
        print("\nSensitivity of the plug-in floor estimate to checkpoint:")
        ref_curve = out["curves"][names[0]]
        ref = np.array(ref_curve["mean"])
        for n in names[1:]:
            curve = out["curves"][n]
            lo1, hi1 = CONFIGS[ref_curve["config"]]
            lo2, hi2 = CONFIGS[curve["config"]]
            mask = (lams >= max(lo1, lo2)) & (lams <= min(hi1, hi2))
            cur = np.array(curve["mean"])
            rel = np.abs(cur[mask] - ref[mask]) / np.maximum(np.abs(ref[mask]), 1e-9)
            print(f"  {n:28s} vs {names[0]:28s}  "
                  f"median rel.diff = {np.median(rel):.3e}, "
                  f"max = {rel.max():.3e}")


if __name__ == "__main__":
    main()
