"""
floor_jacobian.py -- the cheap, model-based floor estimator, validated.

IDENTITY
--------
Tweedie:            grad_x log P(x) = (alpha E[y|x] - x) / sigma^2
Differentiate:      d/dx E[y|x] = (alpha / sigma^2) Cov[y|x]

so with J_eps = d eps_theta / dx for an eps-prediction network,
eps_theta = (x - alpha yhat)/sigma, one gets

    tr Cov[y|x] = (sigma^2/alpha) tr(d yhat/dx)
    floor_integrand(l) = SNR(l) * tr Cov[y|x]
                       = alpha * tr(d yhat/dx)
                       = d - sigma * tr(d eps_theta / dx)          (**)

(**) is the estimator used at CIFAR scale.  It needs only the trained denoiser
and one Hutchinson probe per sample:

    tr J ~ E_v [ v^T J v ],   v Rademacher,

and v^T J is one reverse-mode VJP (or J v one forward-mode JVP).  Cost is one
extra backward pass per evaluation sample.  No dataset scan, no posterior.

Because the floor is model-independent (a conditional variance cannot involve
the regressor), estimates from different
trained models must agree.  That is a testable prediction and it is what
removes the circularity of correcting a model's loss with a model-derived
floor: we check agreement, then correct every model with the consensus curve.

This file validates (**) exactly on a Gaussian-mixture data distribution where
E[y|x] and Cov[y|x] are available in closed form, and checks that Hutchinson
converges to the exact trace.
"""

import numpy as np

from floor_core import alpha_sigma, snr


# ---------------------------------------------------------------------------
# exact denoiser for a mixture of isotropic Gaussians  sum_k pi_k N(mu_k, s2 I)
# ---------------------------------------------------------------------------
class GaussianMixture:
    def __init__(self, mus, pis, s2):
        self.mus = np.asarray(mus, dtype=np.float64)      # (K, D)
        self.pis = np.asarray(pis, dtype=np.float64)
        self.pis /= self.pis.sum()
        self.s2 = float(s2)
        self.D = self.mus.shape[1]

    def posterior(self, x, lam):
        """returns (weights (B,K), component means (B,K,D), component var v)"""
        a, s = alpha_sigma(lam)
        s2, mus = self.s2, self.mus
        v = s2 * s ** 2 / (s ** 2 + a ** 2 * s2)          # scalar
        # mixture weights: pi_k N(x; a mu_k, (s^2 + a^2 s2) I)
        var = s ** 2 + a ** 2 * s2
        diff = x[:, None, :] - a * mus[None, :, :]        # (B,K,D)
        logw = np.log(self.pis)[None, :] - 0.5 * np.sum(diff ** 2, -1) / var
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)                 # (B,K)
        # component posterior means
        m = v * (mus[None, :, :] / s2 + a * x[:, None, :] / s ** 2)
        return w, m, v

    def yhat(self, x, lam):
        w, m, _ = self.posterior(x, lam)
        return np.einsum("bk,bkd->bd", w, m)

    def eps_star(self, x, lam):
        a, s = alpha_sigma(lam)
        return (x - a * self.yhat(x, lam)) / s

    def tr_cov(self, x, lam):
        w, m, v = self.posterior(x, lam)
        mean = np.einsum("bk,bkd->bd", w, m)
        e_m2 = np.einsum("bk,bk->b", w, np.sum(m ** 2, -1))
        return self.D * v + e_m2 - np.sum(mean ** 2, -1)

    def sample(self, n, rng):
        k = rng.choice(len(self.pis), size=n, p=self.pis)
        return self.mus[k] + np.sqrt(self.s2) * rng.standard_normal(
            (n, self.D))


# ---------------------------------------------------------------------------
def trace_jac_fd(fn, x, h=1e-5):
    """exact trace of d fn/dx by central differences, O(D) evaluations"""
    D = x.shape[1]
    tr = np.zeros(x.shape[0])
    for j in range(D):
        e = np.zeros(D)
        e[j] = h
        tr += (fn(x + e)[:, j] - fn(x - e)[:, j]) / (2 * h)
    return tr


def trace_jac_hutchinson(fn, x, n_probe, rng, h=1e-5):
    """Hutchinson estimate  tr J ~ E_v[v^T J v], central-difference JVP"""
    acc = np.zeros(x.shape[0])
    for _ in range(n_probe):
        v = rng.choice([-1.0, 1.0], size=x.shape)
        jv = (fn(x + h * v) - fn(x - h * v)) / (2 * h)
        acc += np.sum(v * jv, axis=1)
    return acc / n_probe


# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    D, K = 12, 4
    gm = GaussianMixture(mus=1.4 * rng.standard_normal((K, D)),
                         pis=rng.dirichlet(np.ones(K)), s2=0.25)

    print("=" * 78)
    print("(V1) identity:  d - sigma*tr(d eps/dx)  ==  SNR * tr Cov[y|x]")
    print("=" * 78)
    print("   lambda     d - s*trJ        SNR*trCov        rel.err")
    for lam in [-4.0, -2.0, 0.0, 1.0, 2.0, 4.0]:
        a, s = alpha_sigma(lam)
        x = a * gm.sample(64, rng) + s * rng.standard_normal((64, D))
        trJ = trace_jac_fd(lambda z: gm.eps_star(z, lam), x)
        lhs = float(np.mean(D - s * trJ))
        rhs = float(np.mean(snr(lam) * gm.tr_cov(x, lam)))
        print(f"  {lam:7.2f}  {lhs:14.6f}  {rhs:14.6f}   "
              f"{abs(lhs-rhs)/abs(rhs):9.2e}")

    print()
    print("=" * 78)
    print("(V2) Hutchinson convergence to the exact trace")
    print("=" * 78)
    lam = 0.5
    a, s = alpha_sigma(lam)
    x = a * gm.sample(256, rng) + s * rng.standard_normal((256, D))
    exact = float(np.mean(D - s * trace_jac_fd(lambda z: gm.eps_star(z, lam), x)))
    print(f"   exact integrand at lambda={lam}: {exact:.6f}")
    print("   n_probe   Hutchinson estimate   rel.err")
    for npb in [1, 4, 16, 64]:
        est = float(np.mean(D - s * trace_jac_hutchinson(
            lambda z: gm.eps_star(z, lam), x, npb, rng)))
        print(f"   {npb:7d}   {est:19.6f}   {abs(est-exact)/abs(exact):8.2e}")
    print("""
   Note the averaging over the evaluation batch already does most of the
   variance reduction, so ONE probe per sample is enough in practice: the
   quantity we need is E_x[tr J], not tr J pointwise.""")

    print()
    print("=" * 78)
    print("(V3) integrated floor from the Jacobian vs from the posterior")
    print("=" * 78)
    grid = np.linspace(-6, 6, 49)
    jac, post = [], []
    for lam in grid:
        a, s = alpha_sigma(lam)
        x = a * gm.sample(128, rng) + s * rng.standard_normal((128, D))
        jac.append(np.mean(D - s * trace_jac_hutchinson(
            lambda z: gm.eps_star(z, lam), x, 1, rng)))
        post.append(np.mean(snr(lam) * gm.tr_cov(x, lam)))
    fj = 0.5 * np.trapezoid(jac, grid)
    fp = 0.5 * np.trapezoid(post, grid)
    print(f"   floor (Jacobian, 1 probe) = {fj:.6f}")
    print(f"   floor (exact posterior)   = {fp:.6f}")
    print(f"   relative difference       = {abs(fj-fp)/fp:.2e}")


if __name__ == "__main__":
    main()
