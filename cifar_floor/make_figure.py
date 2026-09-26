"""
make_figure.py -- the Application-1 figure (three panels).

(a) floor integrand vs log-SNR, one faint curve per converged checkpoint, with
    the consensus overlaid: the floor is model-independent.
(b) reported loss vs training steps, one series per configuration: the series
    are separated by the floor, not by quality.
(c) the same after subtracting the floor: the curves collapse onto a common
    decreasing trend, and the ordering by training budget is recovered.

Reads results/analysis.json and results/analysis_corrected.json.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = {"narrow": "#1f77b4", "mid": "#2ca02c", "wide": "#d62728"}
MINSTEP = 20000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="results/analysis.json")
    ap.add_argument("--corrected", default="results/analysis_corrected.json")
    ap.add_argument("--out", default="fig_cifar_floor.pdf")
    args = ap.parse_args()

    A = json.load(open(args.analysis))
    B = json.load(open(args.corrected))
    lams = np.array(A["lams"])
    C = A["floor_curves"]
    cfg = {k: tuple(v) for k, v in A["configs"].items()}
    order = sorted(cfg, key=lambda c: cfg[c][1])
    floors, sds = B["floors"], B["floor_sd"]

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.5))

    # ---- (a) model independence -----------------------------------------
    # each model's curve is plotted ONLY over its own training range: outside
    # it the network is untrained, the Jacobian vanishes and the integrand
    # saturates at d, which is an artefact rather than a floor estimate.
    for n, c in C.items():
        if c["step"] < MINSTEP:
            continue
        lo, hi = cfg[c["config"]]
        m = (lams >= lo - 1e-9) & (lams <= hi + 1e-9)
        ax[0].plot(lams[m], np.array(c["mean"])[m], lw=1.0, alpha=0.55,
                   color=COL[c["config"]])
    hi = order[-1]
    st = np.stack([C[n]["mean"] for n in B["contributors"][hi]])
    ax[0].plot(lams, st.mean(0), "k-", lw=1.8, label="consensus")
    ax[0].axhline(3072, color="0.5", ls="--", lw=0.8)
    ax[0].text(lams[0] + 0.3, 3072 * 0.955, r"$d=3072$", fontsize=7, color="0.4")
    for c in order:
        ax[0].axvline(cfg[c][1], color=COL[c], ls=":", lw=1)
    ax[0].set_xlabel(r"$\lambda$  (log-SNR)")
    ax[0].set_ylabel(r"$d-\sigma\,\mathbb{E}[\mathrm{tr}\,\partial_x\epsilon_\theta]$")
    ax[0].set_title("(a) the estimator agrees across networks", fontsize=10)
    ax[0].legend(fontsize=7, frameon=False, loc="upper left")
    ax[0].set_ylim(-120, 3300)

    # ---- (b), (c) --------------------------------------------------------
    for panel, key, ttl in ((1, "reported", "(b) reported loss"),
                            (2, "excess", "(c) after subtracting the floor")):
        for c in order:
            lad = B["ladders"][c]
            x = [e["step"] for e in lad]
            y = [e[key] for e in lad]
            ax[panel].plot(x, y, "o-", ms=4, color=COL[c],
                           label=rf"{c}:  $\lambda\!\leq\!{cfg[c][1]:g}$")
        ax[panel].set_xscale("log")
        ax[panel].set_xlabel("training steps")
        ax[panel].legend(fontsize=7, frameon=False,
                         loc="lower left" if panel == 1 else "upper right")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("loss (nats / image)")
    ax[2].set_ylabel("excess (nats / image)")
    ax[1].set_title("(b) reported loss: ranked by range, not quality", fontsize=10)
    ax[2].set_title("(c) excess: ranked by quality", fontsize=10)

    # annotate the inversion on panel (b)
    inv = B["inversion"]
    ax[1].annotate("", xy=(60000, inv["reported_best"]),
                   xytext=(2000, inv["reported_worst"]),
                   arrowprops=dict(arrowstyle="<->", color="0.35", lw=1))
    ax[1].text(11000, np.sqrt(inv["reported_best"] * inv["reported_worst"]) * 1.25,
               f"{inv['ratio']:.1f}$\\times$ apparent gap\nis the floor",
               fontsize=7, color="0.25", ha="center")

    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)

    print("\ncaption numbers")
    for c in order:
        print(f"  {c:7s} floor {floors[c]:8.1f} +- {sds[c]:4.1f} "
              f"({100*floors[c]/B['share'][c]['reported']:.2f}% of reported), "
              f"model part {B['share'][c]['model']:.1f}")


if __name__ == "__main__":
    main()
