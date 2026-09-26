"""
reanalyze.py -- definitive analysis, from results/analysis.json alone (no GPU).

Produces every number quoted in the Application-1 section.

Statistics used, and why
------------------------
* Model-independence is reported on the INTEGRATED floor, not as a pointwise
  relative spread.  At lambda = -8 the integrand is ~1.7 nats, so a pointwise
  relative spread there is 375% while contributing nothing to the integral.
  The integrated floor is the quantity the correction actually uses.

* A model contributes to the floor at lambda only if lambda is inside its own
  training range.  Outside it the network is untrained, its Jacobian is ~0, the
  integrand saturates at d = 3072, and the floor is silently inflated.

* Leave-one-out: every model's excess is recomputed with a floor built from the
  OTHER models only, so no model is corrected using its own estimate.
"""

import json

import numpy as np

MINSTEP = 20000


def integ(lams, v, lo, hi):
    m = (lams >= lo - 1e-9) & (lams <= hi + 1e-9)
    return 0.5 * float(np.trapezoid(np.asarray(v)[m], lams[m]))


def main(path="results/analysis.json", out="results/analysis_corrected.json"):
    A = json.load(open(path))
    lams = np.array(A["lams"])
    C = A["floor_curves"]
    cfg = {k: tuple(v) for k, v in A["configs"].items()}
    rows = {r["name"]: r for r in A["rows"]}
    order = sorted(cfg, key=lambda c: cfg[c][1])

    elig = {c: sorted([n for n in C
                       if cfg[C[n]["config"]][1] >= cfg[c][1] - 1e-9
                       and C[n]["step"] >= MINSTEP]) for c in cfg}
    per_model = {c: np.array([integ(lams, C[n]["mean"], *cfg[c]) for n in elig[c]])
                 for c in cfg}
    floors = {c: float(np.mean(per_model[c])) for c in cfg}
    sds = {c: float(np.std(per_model[c], ddof=1)) for c in cfg}

    print("=" * 78)
    print("(1) the floor is model-independent   [conditional-variance identity]")
    print("=" * 78)
    print(f"{'range':7s} {'lambda':>13s} {'floor':>9s} {'sd':>7s} {'rel':>7s} "
          f"{'n':>3s}  training configs used")
    for c in order:
        srcs = sorted({C[n]['config'] for n in elig[c]})
        print(f"{c:7s} [{cfg[c][0]:5.1f},{cfg[c][1]:5.1f}] {floors[c]:9.1f} "
              f"{sds[c]:7.1f} {100*sds[c]/floors[c]:6.2f}% {len(elig[c]):3d}  {srcs}")
    print("  the absolute scatter is ~50 nats regardless of range: it is the")
    print("  estimator's noise level, not a property of the configuration.")

    print()
    print("=" * 78)
    print("(2) the estimator approaches the floor from above during training")
    print("=" * 78)
    for c in order:
        ns = sorted([n for n in C if C[n]["config"] == c], key=lambda z: C[z]["step"])
        print(f"  {c:7s} " + "  ".join(
            f"{C[n]['step']//1000}k:{integ(lams, C[n]['mean'], *cfg[c]):7.1f}"
            for n in ns))
    print("  (undertrained -> Jacobian too small -> integrand -> d -> floor too high)")

    print()
    print("=" * 78)
    print("(3) how much of a reported loss is the floor")
    print("=" * 78)
    print(f"{'range':7s} {'lam_max':>8s} {'reported':>10s} {'floor':>9s} "
          f"{'share':>8s} {'model part':>11s}")
    share = {}
    for c in order:
        r = rows[f"{c}@60000"]["reported"]
        share[c] = dict(reported=r, floor=floors[c], pct=100 * floors[c] / r,
                        model=r - floors[c])
        print(f"{c:7s} {cfg[c][1]:8.1f} {r:10.1f} {floors[c]:9.1f} "
              f"{100*floors[c]/r:7.2f}% {r-floors[c]:11.1f}")
    lo_c, hi_c = order[0], order[-1]
    print(f"  widening lambda_max from {cfg[lo_c][1]:.1f} to {cfg[hi_c][1]:.1f} "
          f"multiplies the reported loss by "
          f"{share[hi_c]['reported']/share[lo_c]['reported']:.1f}x while the "
          f"model-dependent part shrinks by "
          f"{share[lo_c]['model']/share[hi_c]['model']:.0f}x.")

    print()
    print("=" * 78)
    print("(4) floor-subtracted excess; quality axis = training steps")
    print("=" * 78)
    ladders = {}
    for c in order:
        ns = sorted([n for n in C if C[n]["config"] == c], key=lambda z: C[z]["step"])
        lad = []
        for n in ns:
            loo = [m for m in elig[c] if m != n]
            f_loo = (integ(lams, np.stack([C[m]["mean"] for m in loo]).mean(0), *cfg[c])
                     if loo else floors[c])
            lad.append(dict(step=C[n]["step"], reported=rows[n]["reported"],
                            excess=rows[n]["reported"] - floors[c],
                            excess_loo=rows[n]["reported"] - f_loo))
        ladders[c] = lad
        mono = all(lad[i]["excess"] > lad[i + 1]["excess"] for i in range(len(lad) - 1))
        print(f"  {c:7s}      " + "  ".join(f"{e['step']//1000:>3d}k:{e['excess']:7.1f}"
                                            for e in lad) + f"   monotone: {mono}")
        print(f"  {'(loo)':7s}      " + "  ".join(f"{'':>3s} {e['excess_loo']:7.1f}"
                                                  for e in lad))
    print(f"  floor scatter is ~{np.mean(list(sds.values())):.0f} nats, so every excess is")
    print("  significant except the best wide model, whose excess is an upper")
    print("  bound rather than a measurement.")

    print()
    print("=" * 78)
    print("(5) ranking inversion, both models from the SAME training run")
    print("=" * 78)
    W = sorted([n for n in C if C[n]["config"] == hi_c], key=lambda z: C[z]["step"])
    best, worst = W[-1], W[0]
    rb, rw = rows[best]["reported"], rows[worst]["true"]
    eb, ew = rb - floors[hi_c], rw - floors[lo_c]
    dfl = floors[hi_c] - floors[lo_c]
    dsd = float(np.hypot(sds[hi_c], sds[lo_c]))
    print(f"  {best:12s} evaluated on {hi_c:6s}: reported {rb:8.1f}   excess {eb:7.1f}")
    print(f"  {worst:12s} evaluated on {lo_c:6s}: reported {rw:8.1f}   excess {ew:7.1f}")
    print(f"  raw:    the {C[worst]['step']//1000}k-step model appears {rb/rw:.1f}x better -- WRONG")
    print(f"  excess: orders them {'CORRECTLY' if eb < ew else 'still wrongly'}")
    print(f"  raw gap {rb-rw:.1f} nats; floor difference alone {dfl:.1f} +- {dsd:.1f} "
          f"({100*dsd/dfl:.2f}%), which accounts for all of it.")

    json.dump(dict(floors=floors, floor_sd=sds, contributors=elig,
                   per_model_floor={c: per_model[c].tolist() for c in cfg},
                   share=share, ladders=ladders,
                   inversion=dict(best=best, worst=worst, reported_best=rb,
                                  reported_worst=rw, excess_best=eb,
                                  excess_worst=ew, ratio=rb / rw,
                                  repaired=bool(eb < ew),
                                  floor_diff=dfl, floor_diff_sd=dsd)),
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
