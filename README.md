# The Loss Floor of Denoising Score Matching: Fisher Geometry from Schrödinger Bridges

Code accompanying the paper. Three self-contained parts, one per set of numerical
claims made in the paper.

```
requirements.txt          numpy, matplotlib (analytic_checks/, interpolants/)
                           + torch, torchvision (cifar_floor/ training only)

analytic_checks/          Appendix D — identity checks on toy/analytic models
cifar_floor/              Sec. 5 + Appendix E — the CIFAR-10 measurement
interpolants/             Appendix F — flow matching / stochastic interpolants
```

## `analytic_checks/` — Appendix D

Each script is self-contained, numpy-only, and runs in seconds.

| script | paper reference | what it checks |
|---|---|---|
| `general_gap.py` | App. D.1 | de Bruijn identity for a **nonlinear** corruption drift $f(x)=-ax^3$, solved by direct Fokker–Planck integration — nothing affine or Gaussian is assumed |
| `loss_floor.py` | App. D.2 | the schedule lemma $\gamma(t)\,\alpha_t^2/\sigma_t^4 = d\lambda_t/dt$ for four interpolant families |
| `make_figures.py` | App. D.3, Fig. 2 | the floor–entropy identity on a two-mode Gaussian mixture; regenerates `fig_lossfloor_identity.pdf` |
| `calibrated_loss.py` | App. D.4 | the ranking-inversion example in closed form: two exact-score models scaled by $c=0.97,0.90$ on two SNR ranges |
| `bregman.py` | App. D.5, §6.2 | the masked-diffusion floor as a Bregman-information identity; reproduces the data entropy under four masking schedules |
| `pullback.py` | §3.1, App. B | the pullback construction of the latent Fisher metric (ambient second variation → immersion → pullback) on three reference kernels, including a nonlinear-drift case, checking Prop. 1 (endpoint split) holds without affine structure |

```bash
cd analytic_checks
python general_gap.py
python loss_floor.py
python make_figures.py
python calibrated_loss.py
python bregman.py
python pullback.py
```

## `cifar_floor/` — Sec. 5, Appendix E

The CIFAR-10 measurement: floor share, model-independence, and the ranking
inversion, on a 12M-parameter UNet. See `cifar_floor/README.md` for the full
design, caveats, and commands to reproduce a run from scratch (needs a GPU,
~1.5–4 h per configuration). `results/` contains the analysis output the
paper's numbers are drawn from, so the tables and Fig. 1 can be reproduced
without retraining:

```bash
cd cifar_floor
python make_figure.py    # reads results/analysis.json + results/analysis_corrected.json by default
```

## `interpolants/` — Appendix F

Verifies the floor identity for the whole family of affine-interpolant
objectives ($\epsilon$-, $x_0$-, $v$-prediction, flow matching, stochastic
interpolants): the conditional-covariance ratio, the decomposition
$\mathcal L = \mathcal L_{\rm ideal} + \mathrm{floor}$, the $\omega$-reweighting,
the Jacobian estimator, the table of $c_t$ coefficients, and the
flow-matching/$v$-prediction correspondence $c^{\rm fm}_t/c^v_t = 1/(\alpha_t^2+\sigma_t^2)$.

```bash
cd interpolants
python interpolants.py           # ~2 min, numpy only; see results/interpolants_check.txt for a saved run
```

## Notes

- All analytic checks use exact Gaussian-mixture posteriors, so most identities
  are verified to machine precision (`1e-11`–`1e-16`); the few that compare two
  Monte Carlo estimators report a paired standard-error test instead of a fixed
  tolerance, since a fixed tolerance would only measure sample size.
- `cifar_floor/README.md` also documents why the exact empirical-posterior
  floor fails at this dimension and is replaced by the Jacobian estimator used
  here — worth reading before extending the experiment.
