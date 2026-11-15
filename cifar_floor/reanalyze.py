"""Recompute endpoint-aware floor diagnostics from a saved evaluation."""

import argparse
import json

import numpy as np

from floor_core import integrate_floor


def main(path="results/analysis.json", out="results/analysis_corrected.json"):
    with open(path) as f:
        analysis = json.load(f)
    lams = np.asarray(analysis["lams"])
    curves = analysis["floor_curves"]
    configs = analysis["configs"]
    rows = {row["name"]: row for row in analysis["rows"]}
    order = sorted(configs, key=lambda name: configs[name][1])
    contributors = analysis.get("floor_contributors")
    if contributors is None:
        contributors = {
            cfg: sorted(name for name, curve in curves.items()
                        if configs[curve["config"]][0] <= lo
                        and configs[curve["config"]][1] >= hi and curve["step"] >= 20000)
            for cfg, (lo, hi) in configs.items()
        }

    per_model, floors, spread, share, ladders = {}, {}, {}, {}, {}
    for cfg in order:
        lo, hi = configs[cfg]
        per_model[cfg] = [integrate_floor(lams, curves[n]["mean"], lo, hi)
                          for n in contributors[cfg]]
        estimates = per_model[cfg]
        floor = float(np.mean(estimates)) if estimates else None
        floors[cfg] = floor
        spread[cfg] = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else None

        ladder = []
        names = sorted((name for name in rows if rows[name]["config"] == cfg),
                       key=lambda name: rows[name]["step"])
        for name in names:
            row = rows[name]
            others = [value for n, value in zip(contributors[cfg], estimates) if n != name]
            ladder.append({
                "step": row["step"], "reported": row["reported"],
                "excess": row["reported"] - floor if floor is not None else None,
                "excess_loo": row["reported"] - float(np.mean(others)) if others else None,
            })
        ladders[cfg] = ladder
        if ladder:
            reported = ladder[-1]["reported"]
            share[cfg] = {
                "reported": reported, "floor": floor,
                "pct": 100 * floor / reported if floor is not None and reported else None,
                "model": ladder[-1]["excess"],
            }
        if floor is not None:
            print(f"{cfg:7s} pooled plug-in floor {floor:9.1f} "
                  f"from {len(estimates)} eligible checkpoints")
        for n, estimate in zip(contributors[cfg], estimates):
            print(f"  {n:16s} individual estimate {estimate:9.1f}")

    wide, common = order[-1], analysis["common"]
    wide_rows = sorted((row for row in rows.values() if row["config"] == wide),
                       key=lambda row: row["step"])
    inversion = None
    if len(wide_rows) >= 2 and floors[wide] is not None and floors[common] is not None:
        later, earlier = wide_rows[-1], wide_rows[0]
        rb, rw = later["reported"], earlier["true"]
        eb, ew = rb - floors[wide], rw - floors[common]
        inversion = {
            "best": later["name"], "worst": earlier["name"],
            "best_step": later["step"], "worst_step": earlier["step"],
            "reported_best": rb, "reported_worst": rw, "excess_best": eb,
            "excess_worst": ew, "ratio": rb / rw if rw else None,
            "repaired": rb > rw and eb < ew,
            "floor_diff": floors[wide] - floors[common],
        }
        print("Cross-range comparison uses different objectives even after subtraction.")

    with open(out, "w") as f:
        json.dump(dict(floors=floors, floor_sd=spread, contributors=contributors,
                       per_model_floor=per_model, share=share, ladders=ladders,
                       inversion=inversion), f, indent=2, allow_nan=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", default="results/analysis.json")
    parser.add_argument("--out", default="results/analysis_corrected.json")
    args = parser.parse_args()
    main(args.analysis, args.out)
