# The loss floor on CIFAR-10

Shows that reported diffusion training losses are not comparable across log-SNR
ranges, that the gap is the model-independent loss floor, and that subtracting
the floor repairs the ranking.

## What the experiment establishes

1. **The floor is model-independent** (a conditional variance cannot involve the
   regressor). Floor curves
   estimated from independently trained checkpoints agree within error bars.
2. **The floor dominates the reported loss.** Reported as `floor %` in the
   output table.
3. **Raw reported losses mis-rank models of known relative quality**, and
   subtracting the floor repairs the ranking (an explicit inversion pair, and
   its floor-subtracted correction, are reported).

## Design

The quality axis comes from **checkpoints**, not from separate runs: one
training run per SNR configuration, saved at 6 step counts, gives a ladder of
model qualities for free. Only two training runs are needed.

* configurations (`common.py: CONFIGS`), all VP with `alpha^2 = sigmoid(lambda)`:
  * `narrow` `lambda in [-8, 6.0]`
  * `mid`    `lambda in [-8, 9.2]`   (`sigma_min ~ 1e-2`)
  * `wide`   `lambda in [-8, 13.8]`  (`sigma_min ~ 1e-3`)
* ground truth quality: every checkpoint is also evaluated on the single
  `COMMON` range, with **common random numbers** across models.
* floor estimator: `floor_integrand(l) = d - sigma(l) E[tr d eps/dx]`,
  Hutchinson with one Rademacher probe. Derivation and validation in
  `floor_jacobian.py`.

## Running

```bash
python train.py --smoke                                    # ~10 s, no download
python train.py --config wide   --steps 60000 --out runs/wide
python train.py --config narrow --steps 60000 --out runs/narrow
python train.py --config mid    --steps 60000 --out runs/mid   # optional, strengthens the rank statistic
python evaluate.py --ckpt runs/*/step*.pt --out results/analysis.json
python make_figure.py --analysis results/analysis.json --out fig_cifar_floor.pdf
```

Two runs of 60k steps with the default 12M-parameter UNet at batch 128. Budget
roughly 1.5-4 h per run on one modern GPU, plus ~20 min of evaluation.

`results/analysis.json` and `results/analysis_corrected.json` are the pair the
paper's numbers and figure are drawn from — both included, so the table and
Fig. 1 can be reproduced without retraining:

```bash
python make_figure.py --analysis results/analysis.json --corrected results/analysis_corrected.json
```

(these are `make_figure.py`'s defaults, so a bare `python make_figure.py` from
this directory works too). `reanalyze.py` reproduces `analysis_corrected.json`
from `analysis.json` (leave-one-out excess, per-model floor spread, and the
inversion pair), and is the right entry point if you only want to recompute
the summary statistics from an existing evaluation run.

## Files

| file | role |
|---|---|
| `floor_core.py` | numpy reference: exact empirical-posterior floor, analytic Gaussian case. Run directly to validate. |
| `floor_jacobian.py` | derivation and validation of the Hutchinson floor estimator used at scale. Run directly. |
| `common.py` | schedule, CIFAR loader, UNet, ELBO-weighted DSM loss |
| `train.py` | one training run per configuration, with checkpoints |
| `floor_est.py` | torch floor estimator; standalone model-independence check |
| `evaluate.py` | the ranking analysis; writes `results/analysis.json` |
| `reanalyze.py` | leave-one-out floor recomputation and the inversion statistics; writes `results/analysis_corrected.json` |
| `make_figure.py` | the paper figure |

## Two caveats

**The exact empirical-posterior floor does not work at this dimension.** For the
empirical distribution over 50k images the total information is capped at
`ln N ~ 10.8` nats, while the population floor over a useful SNR range is
thousands of nats; in `d = 3072` the posterior collapses onto the nearest
training image by about `lambda = 0`, where it retains only ~12% of the
population value (reproduce with `floor_core.py`). This is why the floor is
estimated from the denoiser Jacobian instead. The empirical estimator remains
valid at low SNR and is kept as a cross-check.

**Floor subtraction is necessary but not sufficient.** Excesses computed on
different SNR ranges are still integrals over different domains, so removing the
floor removes the dominant model-independent confound without making the
residuals exactly commensurate. The recommendation the experiment supports is:
report the floor-subtracted excess *and* a common-range evaluation.
