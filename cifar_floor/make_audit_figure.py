"""Plot the held-out CIFAR-10 audit from the saved evaluations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from floor_core import cifar_rounding_risk_bound


def main(results, out):
    baselines = json.loads((results / "wide240k_paired_baselines.json").read_text())
    audit = json.loads((results / "wide240k_audit_seed1_grid219.json").read_text())
    
    grid = np.asarray(audit["lams"])
    
    jac = np.asarray(audit["floor_curves"]["wide@240000"]["mean"])
    tail = np.asarray(baselines["tail_grid"])
    dsm, hard, soft = np.asarray(baselines["tail_mean"]).T[:3]
    
    dsm_color, jac_color = "#263f72", "#a0462e"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), layout="constrained")
    
    ax = axes[0]
    ax.plot(baselines["grid"], np.asarray(baselines["network"])[:, 0],
            color=dsm_color, label="Network DSM")
    ax.plot(grid, jac, color=jac_color, ls="--", label="Learned Jacobian")
    ax.axvspan(13, 13.8, color="#f3eadb")
    ax.set(title="Wide evaluation range", xlabel="Log-SNR",
           ylabel="Squared noise error per image", xlim=(-8, 13.8))
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    ax.plot(tail, dsm, color=dsm_color, label="Network DSM")
    ax.plot(grid[grid >= 13], jac[grid >= 13], color=jac_color, ls="--",
            label="Learned Jacobian")
    ax.plot(tail, hard, color="#2a675b", marker="o", ms=3, label="Hard rounding")
    ax.plot(tail, soft, color="#54545b", marker="s", ms=3, label="Soft discrete")
    fine = np.linspace(13, 13.8, 300)
    ax.plot(fine, cifar_rounding_risk_bound(fine), color="#2a675b", ls=":",
            label="Floor upper bound")
    ax.set(title="High-SNR tail", xlabel="Log-SNR", xlim=(13, 13.8), yscale="log")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    
    for ax in axes:
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.results, args.out)

