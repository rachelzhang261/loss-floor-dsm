"""
evaluate.py -- the ranking-inversion analysis.

For every checkpoint we compute three numbers on held-out CIFAR-10:

  reported   L(own config)      what a paper trained with that config reports
  true       L(common config)   the same objective on ONE fixed yardstick range
  floor      pooled checkpoint estimate on the model's training range
  excess     reported - floor (a diagnostic subtraction)

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

from common import CONFIGS, COMMON, UNet, cifar_tensors, dsm_loss_terms
from floor_core import cifar_rounding_risk_bound, log_snr_grid
from floor_est import floor_curve, integrate

ROUNDING_START = 13.0
PAIRED_FIELDS = ("jacobian", "dsm", "stein", "dsm_minus_jacobian", "stein_minus_jacobian")


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
        tot += dsm_loss_terms(model, y, lo, hi, generator=g).double().sum().item()
        n += batch
    return tot / n


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[inverse]


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float((ra @ rb) / denom) if denom else None


def integrated_stats(lams, mean, stderr, lo, hi):
    """Trapezoidal mean and Monte Carlo SE assuming independent grid nodes."""
    mask = (lams >= lo) & (lams <= hi)
    grid = lams[mask]
    gaps = np.diff(grid, prepend=grid[0], append=grid[-1])
    weights = (gaps[:-1] + gaps[1:]) / 4
    return {
        "mean": float(weights @ mean[mask]),
        "mc_se": float(np.linalg.norm(weights * stderr[mask])),
    }


def load(ck, device):
    blob = torch.load(ck, map_location="cpu", weights_only=True)
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
    p.add_argument("--n-batches", type=int, default=100)
    p.add_argument("--floor-batches", type=int, default=8)
    p.add_argument("--n-probe", type=int, default=1)
    p.add_argument("--trace-mode", choices=("vjp", "jvp"), default="vjp")
    p.add_argument("--n-grid", type=int, default=29, help="uniform nodes before inserting endpoints")
    p.add_argument("--seed", type=int, default=0, help="offset for the held-out sampling seeds")
    p.add_argument("--min-step-for-floor", type=int, default=20000,
                   help="minimum step for inclusion in the pooled estimate")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if (args.batch < 2 or args.n_batches < 1 or args.floor_batches < 1
            or args.batch // 2 * args.floor_batches < 2 or args.n_probe < 1 or args.n_grid < 2):
        p.error("need at least two grid nodes and floor samples, and positive batch/probe counts")

    dev = args.device if torch.cuda.is_available() else "cpu"
    data = cifar_tensors(args.data, train=False)
    lo_all = min(v[0] for v in CONFIGS.values())
    hi_all = max(v[1] for v in CONFIGS.values())
    endpoints = [e for bounds in CONFIGS.values() for e in bounds]
    rounding_tail_bound = None
    if lo_all < ROUNDING_START < hi_all:
        endpoints.append(ROUNDING_START)
        tail_grid = np.linspace(ROUNDING_START, hi_all, 801)
        # The rounding-risk bound decreases throughout this tail. Left rectangles
        # give an upper sum, so quadrature also preserves the bound.
        rounding_tail_bound = float(
            0.5 * np.diff(tail_grid) @ cifar_rounding_risk_bound(tail_grid[:-1]))
    lams = log_snr_grid(lo_all, hi_all, args.n_grid, endpoints)

    rows, curves = [], {}
    for ck in args.ckpt:
        model, cfg, step = load(ck, dev)
        name = f"{cfg}@{step}"
        if name in curves:
            p.error(f"duplicate checkpoint configuration and step: {name}")
        own = CONFIGS[cfg]
        rows.append({
            "name": name, "config": cfg, "step": step, "ckpt": ck,
            "reported": eval_loss(model, data, *own, args.n_batches,
                                  args.batch, 1234 + args.seed, dev),
            "true": eval_loss(model, data, *CONFIGS[COMMON], args.n_batches,
                              args.batch, 1234 + args.seed, dev),
        })
        mu, se = floor_curve(model, data, lams, args.batch // 2,
                             args.floor_batches, args.n_probe, seed=7 + args.seed,
                             device=dev, paired=True, trace_mode=args.trace_mode)
        diagnostics = {
            field: integrated_stats(lams, mu[:, j], se[:, j], *own)
            for j, field in enumerate(PAIRED_FIELDS)
        }
        if rounding_tail_bound is not None and own[0] < ROUNDING_START and own[1] == hi_all:
            prefix = integrated_stats(lams, mu[:, 1], se[:, 1], own[0], ROUNDING_START)
            tail_jacobian = integrated_stats(lams, mu[:, 0], se[:, 0], ROUNDING_START, own[1])
            tail_dsm = integrated_stats(lams, mu[:, 1], se[:, 1], ROUNDING_START, own[1])
            diagnostics["rounding_tail"] = {
                "start_log_snr": ROUNDING_START,
                "jacobian": tail_jacobian,
                "dsm": tail_dsm,
                "stein_minus_jacobian": integrated_stats(
                    lams, mu[:, 4], se[:, 4], ROUNDING_START, own[1]),
                "rounding_risk_upper_bound": rounding_tail_bound,
                "tail_excess_lower_bound_estimate": tail_dsm["mean"] - rounding_tail_bound,
                "hybrid_floor_upper_bound_estimate": prefix["mean"] + rounding_tail_bound,
                "hybrid_mc_se": prefix["mc_se"],
            }
        curves[name] = {"mean": mu[:, 0].tolist(), "stderr": se[:, 0].tolist(),
                        "step": step, "config": cfg, "diagnostics": diagnostics}
        print(f"  {name:20s} reported={rows[-1]['reported']:10.1f}  "
              f"common={rows[-1]['true']:10.1f}", flush=True)

    # Pool only eligible checkpoints covering the full integration range.
    floors, consensus_by_cfg, spread_by_cfg, contributors = {}, {}, {}, {}
    for cname, (lo, hi) in CONFIGS.items():
        good = [n for n, c in curves.items()
                if CONFIGS[c["config"]][0] <= lo and CONFIGS[c["config"]][1] >= hi
                and c["step"] >= args.min_step_for_floor]
        contributors[cname] = good
        if not good:
            floors[cname] = None
            continue

        stack = np.stack([curves[n]["mean"] for n in good])
        mu = stack.mean(0)
        mask = (lams >= lo) & (lams <= hi)
        relative_spread = stack.std(0) / np.maximum(np.abs(mu), 1e-9)
        floors[cname] = integrate(lams, mu, lo, hi)
        consensus_by_cfg[cname] = mu.tolist()
        spread_by_cfg[cname] = float(np.median(relative_spread[mask]))
        print(f"[{cname}] pooled estimate={floors[cname]:.1f} ({len(good)} eligible checkpoints)")

    for row in rows:
        floor = floors[row["config"]]
        row["floor"] = floor
        row["excess"] = row["reported"] - floor if floor is not None else None
        row["floor_frac"] = floor / row["reported"] if floor is not None and row["reported"] else None
        if row["excess"] is not None and row["excess"] < 0:
            print(f"WARNING {row['name']}: pooled plug-in estimate exceeds measured DSM loss.")

    common = [r["true"] for r in rows]
    rho_raw = spearman([r["reported"] for r in rows], common)
    rho_cor = spearman([r["excess"] for r in rows], common) if all(
        r["excess"] is not None for r in rows) else None
    inversions = []
    for x, y in combinations(rows, 2):
        if x["config"] == y["config"] or x["true"] == y["true"]:
            continue
        better, worse = (x, y) if x["true"] < y["true"] else (y, x)
        if better["reported"] > worse["reported"]:
            eb, ew = better["excess"], worse["excess"]
            inversions.append({
                "better_model": better["name"], "worse_model": worse["name"],
                "better_reported": better["reported"], "worse_reported": worse["reported"],
                "better_excess": eb, "worse_excess": ew,
                "repaired": eb < ew if eb is not None and ew is not None else None,
            })

    print("\n| model | grid DSM | Jacobian estimate | paired difference | common-range DSM |")
    print("|---|---|---|---|---|")
    for row in sorted(rows, key=lambda r: r["true"]):
        d = curves[row["name"]]["diagnostics"]
        print(f"| {row['name']} | {d['dsm']['mean']:.1f} | {d['jacobian']['mean']:.1f} | "
              f"{d['dsm_minus_jacobian']['mean']:.1f} | {row['true']:.1f} |")
        tail = d.get("rounding_tail")
        if tail:
            print(f"  {row['name']} tail [{ROUNDING_START:g}, {hi_all:g}]: "
                  f"Jacobian {tail['jacobian']['mean']:.1f}, rounding bound {rounding_tail_bound:.1f}")

    consensus = consensus_by_cfg.get(max(CONFIGS, key=lambda c: CONFIGS[c][1]))
    spread = list(spread_by_cfg.values())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "floors": floors, "configs": CONFIGS,
                   "common": COMMON, "lams": lams.tolist(),
                   "consensus_floor_curve": consensus,
                   "floor_curves": curves, "floor_contributors": contributors,
                   "sampling": {"seed_offset": args.seed, "batch": args.batch,
                                "dsm_batches": args.n_batches, "floor_batches": args.floor_batches,
                                "floor_probes": args.n_probe, "trace_mode": args.trace_mode},
                   "floor_spread_median": float(np.median(spread)) if spread else None,
                   "floor_spread_max": float(np.max(spread)) if spread else None,
                   "spearman_raw": rho_raw, "spearman_corrected": rho_cor,
                   "inversions": inversions}, f, indent=2, allow_nan=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
