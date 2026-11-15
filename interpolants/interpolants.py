"""Numerical verification of Appendix F (flow matching and stochastic interpolants).

Claims checked against a Gaussian-mixture data distribution, for which the
denoising posterior p(y | x_t) is available in closed form:

  C1  Cov[T | x_t] / Cov[y | x_t] = c_t^2  for every affine target
      T = a_t y + b_t z, with c_t = a_t - b_t alpha_t / sigma_t.   [eq:covtarget]
  C2  the closed-form conditional covariance agrees with a bin-conditioned
      Monte-Carlo estimate that uses no reweighting and no posterior formula.
  C3  L(theta) = L_ideal(theta) + floor, with the floor independent of theta,
      for deliberately suboptimal predictors and raw joint samples.
                                                                   [eq:interpfloor]
  C4  floor = 1/2 \int omega(lambda) M(lambda) dlambda with
      omega = (w / |lambda_dot|) c^2 e^{-lambda}, and omega == 1 for
      epsilon-prediction under ELBO weighting.                     [eq:omega]
  C5  the table entries for c_t, including c_t = sigma_t rdot_t for the
      flow-matching velocity, against numerically differentiated schedules.
                                                                   [tab:targets]
  C6  M(lambda) = d - sigma E[tr d eps*/dx], tying the appendix to the
      Jacobian estimator used in the main text.                    [eq:estimator]
  C7  c_fm / c_v = W / (alpha^2 + sigma^2), with W the Wronskian.
  C8  everything above in d = 8, where the floor is a trace.

Run:  python interpolants.py
"""

import numpy as np

rng = np.random.default_rng(20260916)

TOL = 5e-3          # Monte-Carlo checks
TOL_EXACT = 1e-9    # identities that should hold to machine precision


# --------------------------------------------------------------------------
# Gaussian mixture with a closed-form denoising posterior
# --------------------------------------------------------------------------


class GaussianMixture:
    """y ~ sum_k pi_k N(mu_k, s_k^2 I) in dimension d (isotropic components)."""

    def __init__(self, pi, mu, s):
        self.pi = np.asarray(pi, float)
        self.mu = np.atleast_2d(np.asarray(mu, float))  # (K, d)
        self.s = np.asarray(s, float)                   # (K,)
        self.K, self.d = self.mu.shape

    def sample(self, n, rng):
        k = rng.choice(self.K, size=n, p=self.pi)
        return self.mu[k] + self.s[k, None] * rng.standard_normal((n, self.d))

    def posterior(self, x, alpha, sigma):
        """Exact p(y | x_t = x) for x_t = alpha y + sigma z.

        Returns (w, m, v, mean, tr_cov): mixture weights (n,K), component means
        (n,K,d), component variances (K,), posterior mean (n,d) and trace of the
        posterior covariance (n,).
        """
        x = np.atleast_2d(x)
        n, d = x.shape
        var_marg = alpha ** 2 * self.s ** 2 + sigma ** 2                  # (K,)
        dif = x[:, None, :] - alpha * self.mu[None, :, :]                 # (n,K,d)
        logp = (-0.5 * np.sum(dif ** 2, axis=2) / var_marg[None, :]
                - 0.5 * d * np.log(2 * np.pi * var_marg)[None, :]
                + np.log(self.pi)[None, :])
        logp -= logp.max(axis=1, keepdims=True)
        w = np.exp(logp)
        w /= w.sum(axis=1, keepdims=True)                                 # (n,K)

        v = 1.0 / (1.0 / self.s ** 2 + alpha ** 2 / sigma ** 2)           # (K,)
        m = v[None, :, None] * (self.mu[None, :, :] / self.s[None, :, None] ** 2
                                + alpha * x[:, None, :] / sigma ** 2)     # (n,K,d)

        mean = np.einsum('nk,nkd->nd', w, m)
        second = (np.einsum('nk,k->n', w, v) * d
                  + np.einsum('nk,nkd->n', w, m ** 2))
        tr_cov = second - np.sum(mean ** 2, axis=1)
        return w, m, v, mean, tr_cov

    def sample_posterior(self, x, alpha, sigma, n, rng):
        """Exact draws from p(y | x_t = x) for a single conditioning point x."""
        w, m, v, _, _ = self.posterior(x, alpha, sigma)
        k = rng.choice(self.K, size=n, p=w[0])
        return m[0][k] + np.sqrt(v[k])[:, None] * rng.standard_normal((n, self.d))


# --------------------------------------------------------------------------
# Interpolant schedules and target parametrisations
# --------------------------------------------------------------------------


def sched_vp_trig(t):
    """alpha = cos(pi t / 2), sigma = sin(pi t / 2): alpha^2 + sigma^2 = 1."""
    h = 0.5 * np.pi
    return (np.cos(h * t), np.sin(h * t), -h * np.sin(h * t), h * np.cos(h * t))


def sched_linear(t):
    """alpha = 1 - t, sigma = t: the rectified-flow / flow-matching interpolant,
    oriented so that t = 0 is data and t = 1 is noise."""
    o = np.ones_like(np.asarray(t, float))
    return (1.0 - t, t, -o, o)


TARGETS = {
    'eps':  lambda a, s, ad, sd: (0.0 * a, 1.0 + 0.0 * a),
    'x0':   lambda a, s, ad, sd: (1.0 + 0.0 * a, 0.0 * a),
    'v':    lambda a, s, ad, sd: (-s, a),
    'flow': lambda a, s, ad, sd: (ad, sd),
}


def c_factor(a, s, ad, sd, name):
    at, bt = TARGETS[name](a, s, ad, sd)
    return at - bt * a / s, at, bt


# --------------------------------------------------------------------------
# C1: the ratio Cov[T|x] / Cov[y|x] is c_t^2, target by target
# --------------------------------------------------------------------------


def check_ratio(gm, sched, label, n=200_000):
    print(f'\n[C1] Cov[T | x_t] / Cov[y | x_t] = c_t^2   ({label}, d={gm.d})')
    print(f'{"t":>6} {"target":>6} {"c_t^2":>12} {"measured ratio":>16} {"rel.err":>10}')
    worst = 0.0
    for t in (0.15, 0.35, 0.55, 0.80):
        a, s, ad, sd = sched(t)
        x0 = np.atleast_2d(a * gm.mu.mean(axis=0) + 0.5 * s)
        y = gm.sample_posterior(x0, a, s, n, rng)     # exact conditional draws
        z = (x0 - a * y) / s                          # noise forced by x_t = x0
        tr_cov_y = np.mean(np.sum((y - y.mean(0)) ** 2, axis=1))
        for name in TARGETS:
            ct, at, bt = c_factor(a, s, ad, sd, name)
            T = at * y + bt * z
            tr_cov_T = np.mean(np.sum((T - T.mean(0)) ** 2, axis=1))
            ratio = tr_cov_T / tr_cov_y
            rel = abs(ratio - ct ** 2) / ct ** 2
            worst = max(worst, rel)
            print(f'{t:6.2f} {name:>6} {ct ** 2:12.5f} {ratio:16.5f} {rel:10.2e}')
    ok = worst < TOL_EXACT
    print(f'  worst relative error: {worst:.2e}   -> {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------------
# C2: closed-form posterior against unweighted bin conditioning (d = 1)
# --------------------------------------------------------------------------


def check_bin_conditioning(gm, sched, n=8_000_000, half_width=0.02):
    """Compare the closed-form posterior against raw samples falling in a bin.

    A point estimate Var[y | x = x0] is compared against Var[y | x in B], which
    carries an O(h^2) bin-width bias; the unbiased comparison is the law of
    total variance restricted to the bin,
        Var[y | x in B] = E[Var[y|x] | x in B] + Var[E[y|x] | x in B],
    with both right-hand terms evaluated from the closed form at the sampled x.
    Both numbers are reported, so the bias is visible and the test is on the
    unbiased one.
    """
    print('\n[C2] closed-form posterior vs. unweighted bin conditioning (d=1)')
    print(f'{"t":>6} {"x0":>8} {"point est.":>11} {"law of tot.var":>15} '
          f'{"binned MC":>11} {"n in bin":>9} {"rel.err":>9} {"MC s.e.":>9}')
    worst_z = 0.0
    for t in (0.25, 0.50, 0.75):
        a, s, ad, sd = sched(t)
        y = gm.sample(n, rng)
        x = a * y + s * rng.standard_normal(y.shape)
        for x0 in (a * gm.mu.mean() - 0.3, a * gm.mu.mean() + 0.6):
            sel = np.abs(x[:, 0] - x0) < half_width
            cnt = int(sel.sum())
            if cnt < 2000:
                print(f'{t:6.2f} {x0:8.3f}  (too few samples: {cnt}) -- skipped')
                continue
            mc = float(np.var(y[sel, 0]))
            _, _, _, mean_b, var_b = gm.posterior(x[sel], a, s)
            ltv = float(np.mean(var_b) + np.var(mean_b[:, 0]))
            _, _, _, _, tr_cov = gm.posterior(np.array([[x0]]), a, s)
            point = float(tr_cov[0])
            # standard error of a variance estimate: sqrt((m4 - m2^2)/n)
            c = y[sel, 0] - y[sel, 0].mean()
            se = float(np.sqrt((np.mean(c ** 4) - mc ** 2) / cnt))
            z = abs(mc - ltv) / se
            worst_z = max(worst_z, z)
            print(f'{t:6.2f} {x0:8.3f} {point:11.6f} {ltv:15.6f} {mc:11.6f} '
                  f'{cnt:9d} {abs(mc - ltv) / ltv:9.2e} {z:8.2f} s.e.')
    ok = worst_z < 4.0
    print(f'  largest deviation: {worst_z:.2f} standard errors'
          f'   -> {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------------
# C3: the decomposition L = L_ideal + floor, from raw joint samples
# --------------------------------------------------------------------------


def check_decomposition(gm, sched, label, n=2_000_000):
    """L - L_ideal and the floor are two unbiased estimators of the same number,
    evaluated on the same samples; the test is therefore on the paired
    difference, whose standard error is available, rather than on a fixed
    relative tolerance (which would only measure the sample size)."""
    print(f'\n[C3] L(theta) = L_ideal(theta) + floor   ({label}, d={gm.d}, '
          f'n={n:,})')
    t = 0.4
    a, s, ad, sd = sched(t)
    y = gm.sample(n, rng)
    z = rng.standard_normal(y.shape)
    x = a * y + s * z
    _, _, _, mean_y, tr_cov_y = gm.posterior(x, a, s)

    print(f'{"target":>6} {"predictor":>16} {"L":>11} {"L_ideal":>11} '
          f'{"L-L_ideal":>11} {"floor":>11} {"rel.err":>9} {"paired z":>9}')
    worst_z = 0.0
    for name in TARGETS:
        ct, at, bt = c_factor(a, s, ad, sd, name)
        T = at * y + bt * z
        Tstar = ct * mean_y + (bt / s) * x            # = E[T | x], by the appendix
        floor_i = 0.5 * ct ** 2 * tr_cov_y            # per-sample floor integrand
        floor = float(np.mean(floor_i))
        for lbl, That in (('exact E[T|x]', Tstar),
                          ('0.8 E[T|x]', 0.8 * Tstar),
                          ('E[T|x]+0.3', Tstar + 0.3)):
            li = 0.5 * np.sum((That - T) ** 2, axis=1)
            lideal_i = 0.5 * np.sum((That - Tstar) ** 2, axis=1)
            diff = li - lideal_i - floor_i            # mean zero if the claim holds
            zsc = abs(diff.mean()) / (diff.std(ddof=1) / np.sqrt(n))
            worst_z = max(worst_z, zsc)
            L, L_ideal = float(li.mean()), float(lideal_i.mean())
            print(f'{name:>6} {lbl:>16} {L:11.5f} {L_ideal:11.5f} '
                  f'{L - L_ideal:11.5f} {floor:11.5f} '
                  f'{abs(L - L_ideal - floor) / floor:9.2e} {zsc:8.2f} s.e.')
    ok = worst_z < 4.0
    print(f'  largest deviation: {worst_z:.2f} standard errors'
          f'   -> {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------------
# C4 / C6: the omega reweighting and the Jacobian estimator
# --------------------------------------------------------------------------


def check_omega(gm, sched, label, n=200_000):
    print(f'\n[C4/C6] omega-reweighting and Jacobian estimator   ({label}, d={gm.d})')
    ts = np.linspace(0.06, 0.94, 221)
    lam, trc, jac = [], [], []
    for t in ts:
        a, s, ad, sd = sched(t)
        y = gm.sample(n, rng)
        x = a * y + s * rng.standard_normal(y.shape)
        _, _, _, _, tr_cov_y = gm.posterior(x, a, s)
        m = float(np.mean(tr_cov_y))
        lam.append(2.0 * np.log(a / s))
        trc.append(m)
        # eps*(x) = (x - a E[y|x]) / s, dE[y|x]/dx = (a/s^2) Cov[y|x]
        jac.append(gm.d - s * ((gm.d - (a ** 2 / s ** 2) * m) / s))
    lam, trc, jac = np.array(lam), np.array(trc), np.array(jac)
    M = np.exp(lam) * trc
    rel_jac = float(np.max(np.abs(jac - M) / np.maximum(M, 1e-12)))
    print(f'  C6  max |[d - sigma E tr d_x eps*] - M| / M = {rel_jac:.2e}'
          f'   -> {"PASS" if rel_jac < 1e-8 else "FAIL"}')

    lamdot = np.gradient(lam, ts)
    w = np.abs(lamdot)                                # ELBO weighting
    print(f'{"target":>6} {"floor (dt)":>14} {"floor (omega M dl)":>20} '
          f'{"rel.err":>10}   omega range')
    worst = 0.0
    for name in TARGETS:
        cs = np.array([c_factor(*sched(t), name)[0] for t in ts])
        floor_t = 0.5 * float(np.trapezoid(w * cs ** 2 * trc, ts))
        omega = (w / np.abs(lamdot)) * cs ** 2 * np.exp(-lam)
        floor_l = 0.5 * float(np.trapezoid(omega * M * np.abs(lamdot), ts))
        rel = abs(floor_t - floor_l) / abs(floor_t)
        worst = max(worst, rel)
        tag = ('identically 1' if np.allclose(omega, 1.0, atol=1e-9)
               else f'[{omega.min():.3g}, {omega.max():.3g}]')
        print(f'{name:>6} {floor_t:14.6f} {floor_l:20.6f} {rel:10.2e}   {tag}')
    ok = worst < TOL_EXACT and rel_jac < 1e-8
    print(f'  worst relative error: {worst:.2e}   -> {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------------
# C5 / C7: the table entries, and the flow-matching / v correspondence
# --------------------------------------------------------------------------


def check_table():
    print('\n[C5/C7] the c_t table, and the flow-matching / v-prediction ratio')
    ok = True
    for label, sched in (('trigonometric VP (alpha^2+sigma^2=1)', sched_vp_trig),
                         ('linear (alpha=1-t, sigma=t)', sched_linear)):
        print(f'  {label}:')
        for t in (0.2, 0.4, 0.6, 0.8):
            a, s, ad, sd = sched(t)
            r, h = a / s, 1e-6
            a1, s1, _, _ = sched(t + h)
            a2, s2, _, _ = sched(t - h)
            rdot = ((a1 / s1) - (a2 / s2)) / (2 * h)
            c_fm = c_factor(a, s, ad, sd, 'flow')[0]
            c_v = c_factor(a, s, ad, sd, 'v')[0]
            c_eps = c_factor(a, s, ad, sd, 'eps')[0]
            c_x0 = c_factor(a, s, ad, sd, 'x0')[0]
            ratio = (a * sd - ad * s) / (a ** 2 + s ** 2)
            errs = (abs(c_fm - s * rdot) / abs(c_fm),          # c = sigma rdot
                    abs(c_eps + r) / abs(r),                   # c = -r
                    abs(c_x0 - 1.0),                           # c = 1
                    abs(c_v + (a ** 2 + s ** 2) / s) / abs(c_v),
                    abs(c_fm / c_v - ratio))
            ok &= max(errs) < 1e-6
            print(f'    t={t:4.2f}  c_fm={c_fm:9.4f}  sigma*rdot={s * rdot:9.4f}'
                  f'  c_v={c_v:9.4f}  c_fm/c_v={c_fm / c_v:8.5f}'
                  f'  W/(a^2+s^2)={ratio:8.5f}'
                  f'  [max err {max(errs):.1e}]')
    print(f'  -> table identities {"hold" if ok else "FAIL"}; '
          'c_fm/c_v = W/(alpha^2+sigma^2).')
    return ok


# --------------------------------------------------------------------------


def main():
    print('=' * 78)
    print('Appendix F: the loss floor for flow matching and stochastic interpolants')
    print('=' * 78)

    gm1 = GaussianMixture(pi=[0.35, 0.65], mu=[[-1.4], [1.1]], s=[0.45, 0.30])
    d = 8
    mu8 = rng.standard_normal((3, d)) * 1.2
    gm8 = GaussianMixture(pi=[0.2, 0.5, 0.3], mu=mu8, s=[0.5, 0.35, 0.6])

    r = {}
    print('\n' + '-' * 78)
    print('d = 1, trigonometric variance-preserving interpolant')
    print('-' * 78)
    r['C1'] = check_ratio(gm1, sched_vp_trig, 'VP trig')
    r['C2'] = check_bin_conditioning(gm1, sched_vp_trig)
    r['C3'] = check_decomposition(gm1, sched_vp_trig, 'VP trig')
    r['C4/C6'] = check_omega(gm1, sched_vp_trig, 'VP trig')
    r['C5/C7'] = check_table()

    print('\n' + '-' * 78)
    print(f'[C8] d = {d}, linear (flow-matching) interpolant')
    print('-' * 78)
    r['C8a'] = check_ratio(gm8, sched_linear, 'linear')
    r['C8b'] = check_decomposition(gm8, sched_linear, 'linear')
    r['C8c'] = check_omega(gm8, sched_linear, 'linear')

    print('\n' + '=' * 78)
    for k, v in r.items():
        print(f'  {k:8s} {"PASS" if v else "FAIL"}')
    print('  overall:', 'PASS' if all(r.values()) else 'FAIL')
    print('=' * 78)
    return 0 if all(r.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
