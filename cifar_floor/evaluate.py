"""
evaluate.py -- the ranking-inversion analysis.

For every checkpoint we compute three numbers on held-out CIFAR-10:

  reported   L(own config)      what a paper trained with that config reports
  true       L(common config)   the same objective on ONE fixed yardstick range
  floor      FLOOR(own config)  consensus model-independent floor
  excess     reported - floor

All losses use COMMON RANDOM NUMBERS: every model sees the identical stream of
(image, lambda, eps) triples, so differences between models are not obscured by
Monte-Carlo noise.  This matters because the quality differences we are trying
to rank are much smaller than the floor.

Outputs results/analysis.json and a markdown table on stdout.
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np
import torch

from common import CONFIGS, COMMON, UNet, alpha_sigma, cifar_tensors
from floor_est import floor_curve, integrate


# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_loss(model, data, lo, hi, n_batches, batch, seed, device):
    """1/2 (hi-lo) E||eps - eps_theta||^2, nats/image, common random numbers"""
    g = torch.Generator(device=device).manual_seed(seed)
    gi = torch.Generator().manual_seed(seed)
    tot, n = 0.0, 0
    for _ in range(n_batches):
        idx = torch.randint(0, len(data), (batch,), generator=gi)
        y = data[idx].to(device)
        u = (torch.rand(1, device=device, generator=g)
             + torch.arange(batch, device=device) / batch) % 1.0
        lam = lo + (hi - lo) * u
        a, s = alpha_sigma(lam)
        eps = torch.randn(y.shape, device=device, generator=g)
        x = a.view(-1, 1, 1, 1) * y + s.view(-1, 1, 1, 1) * eps
        per = (eps - model(x, lam)).pow(2).flatten(1).sum(1)
        tot += (0.5 * (hi - lo) * per).sum().item()
        n += batch
    return tot / n


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float)))
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float)))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def load(ck, device):
    blob = torch.load(ck, map_location=device)
    m = UNet(**blob["model_kwargs"]).to(device)
    m.load_state_dict(blob["model"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, blob["config"], blob["step"]


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="results/analysis.json")
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--floor-batches", type=int, default=8)
    p.add_argument("--n-grid", type=int, default=29)
    p.add_argument("--min-step-for-floor", type=int, default=20000,
                   help="only converged checkpoints contribute to the "
                        "consensus floor")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    data = cifar_tensors(args.data, train=False)
    lo_all = min(v[0] for v in CONFIGS.values())
    hi_all = max(v[1] for v in CONFIGS.values())
    lams = np.linspace(lo_all, hi_all, args.n_grid)

    rows, curves = [], {}
    for ck in args.ckpt:
        model, cfg, step = load(ck, dev)
        name = f"{cfg}@{step}"
        own = CONFIGS[cfg]
        rows.append({
            "name": name, "config": cfg, "step": step, "ckpt": ck,
            "reported": eval_loss(model, data, *own, args.n_batches,
                                  args.batch, 1234, dev),
            "true": eval_loss(model, data, *CONFIGS[COMMON], args.n_batches,
                              args.batch, 1234, dev),
        })
        mu, se = floor_curve(model, data, lams, args.batch // 2,
                             args.floor_batches, 1, seed=7, device=dev)
        curves[name] = {"mean": mu.tolist(), "stderr": se.tolist(),
                        "step": step, "config": cfg}
        print(f"  {name:20s} reported={rows[-1]['reported']:10.1f}  "
              f"true={rows[-1]['true']:10.1f}", flush=True)

    # ---- consensus floor -------------------------------------------------
    # A model may only contribute to the floor at lambda values INSIDE its own
    # training range.  Outside it the network is untrained, its Jacobian is
    # ~0, and the integrand saturates at d -- which silently inflates the
    # floor and can push it above the loss it is meant to bound.
    floors, consensus_by_cfg, spread_by_cfg = {}, {}, {}
    for cname, (lo, hi) in CONFIGS.items():
        good = [n for n, c in curves.items()
                if CONFIGS[c["config"]][1] >= hi - 1e-9
                and c["step"] >= args.min_step_for_floor]
        if not good:
            good = [n for n, c in curves.items()
                    if CONFIGS[c["config"]][1] >= hi - 1e-9] or list(curves)
        stack = np.stack([curves[n]["mean"] for n in good])
        mu = stack.mean(0)
        m = (lams >= lo - 1e-9) & (lams <= hi + 1e-9)
        sp = (stack.std(0) / np.maximum(np.abs(mu), 1e-9))[m]
        floors[cname] = integrate(lams, mu, lo, hi)
        consensus_by_cfg[cname] = mu.tolist()
        spread_by_cfg[cname] = float(np.median(sp))
        print(f"\n[{cname}] floor={floors[cname]:.1f} from {len(good)} in-range "
              f"models, median spread {100*np.median(sp):.2f}%")

    consensus = np.array(consensus_by_cfg[max(CONFIGS, key=lambda c: CONFIGS[c][1])])
    spread = np.array([spread_by_cfg[c] for c in CONFIGS])
    print("\nconsensus floor per configuration (nats/image)")
    for c, v in floors.items():
        print(f"  {c:8s} lambda in {str(CONFIGS[c]):14s}  floor = {v:10.1f}")

    for r in rows:
        r["floor"] = floors[r["config"]]
        r["excess"] = r["reported"] - r["floor"]
        if r["excess"] < 0:
            print(f"  WARNING {r['name']}: floor {r['floor']:.1f} exceeds "
                  f"reported loss {r['reported']:.1f}. The floor is a lower "
                  f"bound, so this means the estimate is biased high -- most "
                  f"likely an undertrained model (Jacobian too small).")
        r["floor_frac"] = r["floor"] / r["reported"] if r["reported"] else float("nan")

    # ---- ranking analysis ------------------------------------------------
    true = [r["true"] for r in rows]
    rho_raw = spearman([r["reported"] for r in rows], true)
    rho_cor = spearman([r["excess"] for r in rows], true)

    inversions = []
    for x, y in combinations(rows, 2):
        if x["config"] == y["config"]:
            continue
        better, worse = (x, y) if x["true"] < y["true"] else (y, x)
        if better["reported"] > worse["reported"]:
            inversions.append({
                "better_model": better["name"], "worse_model": worse["name"],
                "better_reported": better["reported"],
                "worse_reported": worse["reported"],
                "better_excess": better["excess"], "worse_excess": worse["excess"],
                "repaired": better["excess"] < worse["excess"],
            })

    print(f"\nSpearman rank correlation with true quality")
    print(f"  raw reported loss   : {rho_raw:+.4f}")
    print(f"  floor-corrected     : {rho_cor:+.4f}")
    rep = sum(i["repaired"] for i in inversions)
    print(f"\ncross-configuration ranking inversions: {len(inversions)}"
          f"   repaired by floor subtraction: {rep}")
    for i in inversions[:8]:
        print(f"   {i['better_model']:16s} reports {i['better_reported']:10.1f}  >  "
              f"{i['worse_model']:16s} reports {i['worse_reported']:10.1f}   "
              f"| excess {i['better_excess']:9.1f} vs {i['worse_excess']:9.1f}"
              f"  {'REPAIRED' if i['repaired'] else 'not repaired'}")

    print("\n| model | reported | floor | floor % | excess | true (common) |")
    print("|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda z: z["true"]):
        print(f"| {r['name']} | {r['reported']:.1f} | {r['floor']:.1f} | "
              f"{100*r['floor_frac']:.1f}% | {r['excess']:.1f} | {r['true']:.1f} |")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "floors": floors, "configs": CONFIGS,
                   "common": COMMON, "lams": lams.tolist(),
                   "consensus_floor_curve": consensus.tolist(),
                   "floor_curves": curves,
                   "floor_spread_median": float(np.median(spread)),
                   "floor_spread_max": float(spread.max()),
                   "spearman_raw": rho_raw, "spearman_corrected": rho_cor,
                   "inversions": inversions}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
