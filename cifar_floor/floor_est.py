"""
floor_est.py -- model-based loss floor via the Hutchinson trace.

    floor_integrand(l) = d - sigma(l) * E_x [ tr( d eps_theta / dx ) ]
    FLOOR[lo,hi]       = 1/2 * int_lo^hi floor_integrand(l) dl

The identity is derived and validated in floor_jacobian.py (agreement with the
exact posterior covariance to ~1e-11 on a Gaussian mixture).

tr J is estimated with Rademacher probes using a vector-Jacobian product:
    v^T J  = autograd.grad(eps_theta, x, grad_outputs=v)
    tr J  ~ mean over v of  <v, v^T J>
One probe per evaluation sample suffices, because the quantity needed is
E_x[tr J] and the batch average already performs the variance reduction (see
floor_jacobian.py, V2).
"""

import argparse
import json
import os

import numpy as np
import torch

from common import CONFIGS, alpha_sigma, cifar_tensors


@torch.enable_grad()
def integrand_at(model, y, lam_val, n_probe=1, generator=None):
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
    x = (a4 * y + s4 * eps).detach().requires_grad_(True)
    pred = model(x, lam)

    tr = torch.zeros(B, device=device)
    for k in range(n_probe):
        # Rademacher probe via torch.rand (accepts a device generator on all
        # backends; torch.randint's dtype handling is less portable)
        u = torch.rand(y.shape, device=device, generator=generator)
        v = (u < 0.5).to(y.dtype).mul_(-2.0).add_(1.0)   # +-1, no scalar where
        (vJ,) = torch.autograd.grad(pred, x, grad_outputs=v,
                                    retain_graph=(k < n_probe - 1),
                                    create_graph=False)
        tr += (v * vJ).flatten(1).sum(1)
    tr /= n_probe

    return (D - s * tr).detach()


def floor_curve(model, data, lams, batch=64, n_batches=8, n_probe=1,
                seed=0, device="cuda"):
    """returns (mean integrand, stderr) on the given lambda grid"""
    g = torch.Generator(device=device).manual_seed(seed)
    idx_g = torch.Generator().manual_seed(seed + 1)
    means, errs = [], []
    for lam in lams:
        vals = []
        for _ in range(n_batches):
            idx = torch.randint(0, len(data), (batch,), generator=idx_g)
            y = data[idx].to(device)
            vals.append(integrand_at(model, y, lam, n_probe, g).cpu())
        v = torch.cat(vals).double()
        means.append(v.mean().item())
        errs.append((v.std(unbiased=True) / np.sqrt(len(v))).item())
    return np.array(means), np.array(errs)


def integrate(lams, vals, lo, hi):
    """1/2 * int_lo^hi vals dlambda, restricted to the grid points in [lo,hi]"""
    m = (lams >= lo - 1e-9) & (lams <= hi + 1e-9)
    return 0.5 * float(np.trapezoid(np.asarray(vals)[m], np.asarray(lams)[m]))


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True,
                   help="one or more checkpoints; the floor from each is "
                        "reported so that model-independence can be checked")
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="results/floor_curves.json")
    p.add_argument("--n-grid", type=int, default=29)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--n-probe", type=int, default=1)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from common import UNet
    data = cifar_tensors(args.data, train=False)      # held-out images

    lo = min(v[0] for v in CONFIGS.values())
    hi = max(v[1] for v in CONFIGS.values())
    lams = np.linspace(lo, hi, args.n_grid)

    out = {"lams": lams.tolist(), "curves": {}, "configs": CONFIGS}
    for ck in args.ckpt:
        blob = torch.load(ck, map_location=args.device)
        model = UNet(**blob["model_kwargs"]).to(args.device)
        model.load_state_dict(blob["model"])
        model.eval()
        for prm in model.parameters():
            prm.requires_grad_(False)      # we differentiate wrt x, not params

        mu, se = floor_curve(model, data, lams, args.batch, args.n_batches,
                             args.n_probe, seed=0, device=args.device)
        name = os.path.basename(ck).replace(".pt", "")
        out["curves"][name] = {"mean": mu.tolist(), "stderr": se.tolist()}
        line = "  ".join(f"{c}={integrate(lams, mu, *r):.1f}"
                         for c, r in CONFIGS.items())
        print(f"[{name}]  floor per config:  {line}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")

    # model-independence check
    names = list(out["curves"])
    if len(names) > 1:
        print("\nmodel-independence of the floor (conditional-variance identity):")
        ref = np.array(out["curves"][names[0]]["mean"])
        for n in names[1:]:
            cur = np.array(out["curves"][n]["mean"])
            rel = np.abs(cur - ref) / np.maximum(np.abs(ref), 1e-9)
            print(f"  {n:28s} vs {names[0]:28s}  "
                  f"median rel.diff = {np.median(rel):.3e}, "
                  f"max = {rel.max():.3e}")


if __name__ == "__main__":
    main()
