# The Loss Floor of Denoising Score Matching: Fisher Geometry from Schrödinger Bridges

Code and saved results accompanying the approved merged manuscript. The paper
relates the denoising loss floor to conditional endpoint Fisher geometry and
information flow, and audits learned floor estimates with independent reference
predictors on CIFAR-10 and ImageNet-64.

## Contents

| Directory | Contents |
|---|---|
| `analytic_checks/` | Low-dimensional identity checks for Appendices B and D. |
| `cifar_floor/` | CIFAR-10 training, learned-Jacobian diagnostics, discrete-reference comparisons, and the ImageNet-64 audit. |
| `interpolants/` | Appendix F checks for affine target families and stochastic interpolants. |
| `figures/` | Both final manuscript figures, plus the exact TeX and saved data for Figure 2. |

## Installation

Use Python 3.10 or newer in a virtual environment. Choose the dependencies needed:

```sh
python -m pip install -r requirements.txt
# For CIFAR-10 training/evaluation:
python -m pip install -r requirements-cifar.txt
# For the ImageNet-64 audit and data acquisition:
python -m pip install -r requirements-imagenet.txt
```

Choose a PyTorch build appropriate to the available hardware. NumPy 2 or later is required for
[`numpy.trapezoid`](https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html).

## Reproduce the reported image results without training

The numerical records for the revised Figure 1 and Table 1 are in
`cifar_floor/results/`; see its README for the file-to-result mapping. To redraw
Figure 1 from those records:

```sh
cd cifar_floor
python make_audit_figure.py --results results --out ../figures/fig_cifar_rebuttal.pdf
```

The ImageNet result is recorded in `results/imagenet_edm_primary.json` under
`integrated_nats_per_dim`. The CIFAR paired risks are in
`results/wide240k_paired_baselines.json` under `tail_integrals`.
These saved measurements can be inspected without PyTorch, datasets, or weights.

## Analytic checks and manuscript locations

Run each command from the repository root:

| Command | Manuscript location | Purpose |
|---|---|---|
| `python analytic_checks/pullback.py` | Section 3.1; Appendix B | Ambient variation, immersion, pullback, and radial identities in finite examples. |
| `python analytic_checks/general_gap.py` | Appendix D.1 | Information-flow identity for a nonlinear corruption drift. |
| `python analytic_checks/loss_floor.py` | Appendix D.2 | Schedule identity. |
| `python cifar_floor/floor_jacobian.py` | Appendix D.3 | Bayes-Jacobian identity for a Gaussian mixture. |
| `python analytic_checks/make_figures.py` | Appendix D.4 | Numerical floor/entropy curves. Its plotting labels predate the final notation; use `figures/` for the exact manuscript rendering. |
| `python analytic_checks/calibrated_loss.py` | Appendix D.5 | Effect of changing the SNR interval on both loss components; not a model-ranking correction. |
| `python analytic_checks/bregman.py` | Section 6.1; Appendix D.6 | Masked-diffusion entropy identity and endpoint quadrature behavior. |
| `python interpolants/interpolants.py` | Appendix F | Target covariance, weighting, and flow/velocity coefficient identities. |

Figure 2's final typography and notation can be rebuilt from the saved numerical
data with pdfLaTeX and PGFPlots:

```sh
cd figures
pdflatex -interaction=nonstopmode -halt-on-error fig_lossfloor_identity_merged_v1.tex
```

## Interpretation and reproducibility

The Jacobian expression is exact for the Bayes predictor. Substituting a learned
network can bias it in either direction; agreement with its Stein form does not
remove that bias. The current experiments establish paired improvements on the
same denoising objective and an independent upper bound on the floor.

The file schemas retain historical keys such as `floor`, `floor_frac`, `true`,
and `pooled_floor`. Read `floor` as a learned plug-in estimate unless explicitly
identified as analytic, and `true` as the common narrow-range DSM comparison;
neither key asserts an oracle population value. Saved JSON files are unchanged.

The training and evaluation scripts are unchanged from the working project.
Datasets, trained checkpoints, and downloaded third-party code are excluded from
this source release. See `cifar_floor/README.md` for acquisition and training
commands, fixed data revision, and the external EDM commit. New training runs
need not reproduce a saved checkpoint exactly.

The code is packaged with anonymous project metadata. Third-party datasets,
weights, and code remain subject to their upstream terms; this package does not
add a new license or redistribute those assets.
