"""
floor_core.py -- reference (numpy) implementation of the exact loss floor.

CONVENTIONS
-----------
Variance-preserving corruption parameterised by log-SNR lambda:

    alpha(l)^2 = sigmoid(l),   sigma(l)^2 = sigmoid(-l),   SNR(l) = e^l
    x_l = alpha(l) * y + sigma(l) * eps,    eps ~ N(0, I)

The continuous-time diffusion loss (Kingma et al. VDM, eps-prediction, ELBO /
"bridge" weighting) is

    L = 1/2 * int_{l_min}^{l_max}  E || eps - eps_theta(x_l, l) ||^2  dl.

Since eps = (x_l - alpha y)/sigma, the optimal predictor is the posterior mean
and the irreducible residual at fixed lambda is

    floor_integrand(l) = SNR(l) * E_{x_l} [ tr Cov(y | x_l) ]            (*)

so that

    FLOOR = 1/2 * int floor_integrand(l) dl.

By I-MMSE (dI/dsnr = mmse/2) this equals exactly

    FLOOR = I(y; x_{l_max}) - I(y; x_{l_min}) = Delta I,

which is the paper's Theorem (loss floor = information gained).

EXACT EMPIRICAL FLOOR
---------------------
For the EMPIRICAL data distribution over a finite set {y_i}, the posterior is an
explicit softmax and (*) is computable with NO model:

    p_i(x) propto exp( -||x - alpha y_i||^2 / (2 sigma^2) )
    tr Cov(y|x) = sum_i p_i ||y_i||^2  -  || sum_i p_i y_i ||^2

This is the quantity the paper says is model-independent, and here it is
literally so.  The models are trained on this empirical distribution, so this is
the correct reference floor for comparing their training losses.

Run this file directly to validate against the analytic Gaussian case.
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------
def alpha_sigma(lam):
    """VP coefficients from log-SNR.  alpha^2 = sigmoid(lam)."""
    lam = np.asarray(lam, dtype=np.float64)
    return np.exp(-0.5 * np.logaddexp(0.0, -lam)), np.exp(-0.5 * np.logaddexp(0.0, lam))


def snr(lam):
    return np.exp(np.asarray(lam, dtype=np.float64))


# ---------------------------------------------------------------------------
# exact empirical posterior
# ---------------------------------------------------------------------------
def posterior_stats(x, Y, lam, ynorm2=None, chunk=256):
    """
    x    : (B, D) noisy states
    Y    : (N, D) the empirical data set defining the posterior
    lam  : scalar log-SNR
    returns tr Cov(y|x) for each row of x, shape (B,)

    Chunked over rows of x: the (B, N) weight matrix is the memory bottleneck.
    """
    a, s = alpha_sigma(lam)
    if ynorm2 is None:
        ynorm2 = np.sum(Y * Y, axis=1)                       # (N,)

    out = np.empty(x.shape[0], dtype=np.float64)
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk]
        # ||x - a y||^2 = ||x||^2 - 2a x.y + a^2 ||y||^2 ; the ||x||^2 term is
        # constant across i and cancels in the softmax, so drop it.
        logits = (a * (xb @ Y.T) - 0.5 * (a ** 2) * ynorm2[None, :]) / (s ** 2)
        logits -= logits.max(axis=1, keepdims=True)
        w = np.exp(logits)
        w /= w.sum(axis=1, keepdims=True)                    # (b, N)

        mean = w @ Y                                         # (b, D)
        e_y2 = w @ ynorm2                                    # (b,)
        out[i:i + chunk] = e_y2 - np.sum(mean * mean, axis=1)
    return out


def floor_integrand(Y, lam, n_samples, rng, Y_eval=None):
    """
    Monte-Carlo estimate of  SNR(l) * E_{x_l}[ tr Cov(y|x_l) ].

    Y      : (N, D) data set defining the POSTERIOR (e.g. the training set)
    Y_eval : (M, D) data set the noisy states are drawn from (defaults to Y).
             Use a held-out set here to avoid evaluating on the same points
             that define the posterior.
    """
    if Y_eval is None:
        Y_eval = Y
    a, s = alpha_sigma(lam)
    idx = rng.integers(0, Y_eval.shape[0], size=n_samples)
    x = a * Y_eval[idx] + s * rng.standard_normal((n_samples, Y.shape[1]))
    mmse = posterior_stats(x, Y, lam)
    return float(snr(lam) * mmse.mean()), float(snr(lam) * mmse.std(ddof=1)
                                                / np.sqrt(n_samples))


def floor_curve(Y, lams, n_samples=256, seed=0, Y_eval=None):
    """floor integrand on a grid of log-SNR values; returns (values, stderrs)"""
    rng = np.random.default_rng(seed)
    vals, errs = [], []
    for lam in lams:
        v, e = floor_integrand(Y, lam, n_samples, rng, Y_eval=Y_eval)
        vals.append(v)
        errs.append(e)
    return np.array(vals), np.array(errs)


def log_snr_grid(lo, hi, n_grid, endpoints):
    """Keep the requested uniform nodes and include every range endpoint."""
    grid = np.linspace(lo, hi, n_grid)
    edges = np.asarray(sorted(set(endpoints)))
    near_edge = np.isclose(grid[:, None], edges, atol=1e-9, rtol=0).any(axis=1)
    return np.sort(np.concatenate([grid[~near_edge], edges]))


def integrate_floor(lams, vals, lo=None, hi=None):
    """Half the trapezoidal integral, interpolating endpoints on legacy grids."""
    lams, vals = np.asarray(lams), np.asarray(vals)
    lo = lams[0] if lo is None else lo
    hi = lams[-1] if hi is None else hi
    if lo < lams[0] - 1e-9 or hi > lams[-1] + 1e-9 or lo >= hi:
        raise ValueError("integration interval must be inside the grid")
    interior = (lams > lo + 1e-9) & (lams < hi - 1e-9)
    x = np.concatenate(([lo], lams[interior], [hi]))
    y = np.concatenate(([np.interp(lo, lams, vals)], vals[interior],
                        [np.interp(hi, lams, vals)]))
    return 0.5 * float(np.trapezoid(y, x))


def cifar_rounding_risk_bound(lams, dimension=3072):
    """Upper bound on epsilon MSE for rounding x/alpha to the 8-bit lattice.

    The clean pixel spacing in [-1, 1] is 2/255. In lattice units the
    corruption is r*Z for Z standard normal. After rounding and clipping,
    the squared index error is bounded by min(abs(round(r*Z)), 255)**2.
    """
    values = []
    for lam in lams:
        r = 127.5 * math.exp(-0.5 * float(lam))
        risk = sum((2 * k - 1) * math.erfc((k - 0.5) / (math.sqrt(2) * r))
                   for k in range(1, 256))
        values.append(dimension * risk / r**2)
    return np.asarray(values)


# ---------------------------------------------------------------------------
# analytic reference: isotropic Gaussian data N(0, s2 I_d)
# ---------------------------------------------------------------------------
def gauss_integrand(lam, d, s2):
    a, s = alpha_sigma(lam)
    return d * s2 * a ** 2 / (s ** 2 + s2 * a ** 2)


def gauss_floor(lam_min, lam_max, d, s2):
    """Delta I for the Gaussian channel = 1/2 d log(1 + s2 SNR) evaluated"""
    return 0.5 * d * (np.log1p(s2 * snr(lam_max)) - np.log1p(s2 * snr(lam_min)))


# ---------------------------------------------------------------------------
def _validate():
    rng = np.random.default_rng(0)
    d, s2 = 8, 0.7
    N = 20000

    print("=" * 76)
    print("validation 1: empirical posterior vs analytic Gaussian integrand")
    print("=" * 76)
    Y = np.sqrt(s2) * rng.standard_normal((N, d))
    Yev = np.sqrt(s2) * rng.standard_normal((2000, d))
    lams = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    print("    lambda    empirical      analytic      rel.err   (mc stderr)")
    for lam in lams:
        v, e = floor_integrand(Y, lam, 1200, rng, Y_eval=Yev)
        g = gauss_integrand(lam, d, s2)
        print(f"   {lam:7.2f}  {v:12.6f}  {g:12.6f}  {abs(v-g)/g:10.2e}"
              f"   ({e:.2e})")

    print()
    print("=" * 76)
    print("validation 2: integrated floor vs Delta I  (I-MMSE)")
    print("=" * 76)
    for lo, hi in [(-6.0, 6.0), (-3.0, 3.0), (-6.0, 2.0)]:
        grid = np.linspace(lo, hi, 41)
        vals, _ = floor_curve(Y, grid, n_samples=1200, seed=1, Y_eval=Yev)
        num = integrate_floor(grid, vals)
        ana = gauss_floor(lo, hi, d, s2)
        # analytic integrand integrated on the same grid, to separate
        # quadrature error from Monte-Carlo error
        quad = integrate_floor(grid, gauss_integrand(grid, d, s2))
        print(f"   lambda in [{lo:5.1f},{hi:5.1f}]  empirical={num:10.6f}  "
              f"quad(analytic)={quad:10.6f}  DeltaI={ana:10.6f}  "
              f"rel.err={abs(num-ana)/ana:.2e}")

    print()
    print("=" * 76)
    print("validation 3: the memorisation caveat")
    print("=" * 76)
    print("""   As lambda grows the empirical posterior collapses onto the nearest
   stored point and tr Cov -> 0, so the EMPIRICAL floor falls below the
   population floor.  This is not an error: it is the exact floor of the
   empirical distribution, which is what the models are trained on.  It does
   mean lambda_max must be reported, and that the empirical floor should not
   be read as an estimate of the population floor at high SNR.""")
    small = Y[:2000]
    print("\n    lambda   N=2000      N=60000     ratio  (collapse onto"
          " nearest neighbour)")
    for lam in [2.0, 5.0, 8.0, 11.0]:
        v_s, _ = floor_integrand(small, lam, 1500, rng, Y_eval=Yev)
        v_l, _ = floor_integrand(Y, lam, 1500, rng, Y_eval=Yev)
        print(f"   {lam:7.2f}  {v_s:10.6f}  {v_l:10.6f}  {v_s/max(v_l,1e-12):7.3f}")


if __name__ == "__main__":
    _validate()
