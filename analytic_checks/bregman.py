"""
bregman.py
===========

The training loss is a Bregman divergence.  Two consequences are tested here.

(B1) THE DECOMPOSITION IS A BREGMAN LAW, NOT AN L2 FACT.
     For any Bregman divergence B_phi and any random target X with posterior
     mean mu = E[X],
            E[B_phi(X, z)]  =  E[B_phi(X, mu)]  +  B_phi(mu, z),
     exactly, for every z.  The first term is model-independent: it is the
     "Bregman information" of X (Banerjee-Guo-Wang).  The paper's loss floor
     is this quantity for phi = |.|^2/2.  We verify the identity for a
     NON-quadratic generator (negative entropy on the simplex), where the
     divergence is a KL and the floor is a conditional entropy.

(B2) THE DISCRETE LOSS FLOOR IS SCHEDULE-INVARIANT AND EQUALS H(data).
     For masked diffusion the loss is a cross-entropy = Bregman divergence with
     the negative-entropy generator.  With a perfect model the residual is
            floor = int_0^1 (1/m) E_U[ sum_{i not in U} H(y_i | y_U) ] dm
     over mask probability m.  We check numerically that this equals the joint
     entropy H(y_1..y_n) for correlated tokens, and that integrating in t with
     the schedule weight -m'(t)/m(t) gives the same number for several
     different schedules m(t).

     This is the discrete analogue of "the floor is fixed once the endpoints of
     the SNR range are fixed": it depends only on the endpoint mask rates, not
     on the shape of the schedule.
"""

import itertools
import numpy as np

rng = np.random.default_rng(7)
np.set_printoptions(precision=6, suppress=True)


# ---------------------------------------------------------------------------
# (B1) Bregman decomposition for a non-quadratic generator
# ---------------------------------------------------------------------------
def neg_entropy(p):
    q = np.clip(p, 1e-300, None)
    return np.sum(q * np.log(q))


def grad_neg_entropy(p):
    return np.log(np.clip(p, 1e-300, None)) + 1.0


def bregman(p, q):
    return neg_entropy(p) - neg_entropy(q) - grad_neg_entropy(q) @ (p - q)


def test_b1(K=5, n_samples=9, trials=4):
    print("=" * 72)
    print("(B1) E[B(X,z)] = E[B(X,mu)] + B(mu,z)   [phi = negative entropy]")
    print("=" * 72)
    print("   trial     E[B(X,z)]     E[B(X,mu)]      B(mu,z)      residual")
    for tr in range(trials):
        # random distribution over random targets X (each a point on simplex)
        X = rng.dirichlet(np.ones(K) * 0.7, size=n_samples)
        w = rng.dirichlet(np.ones(n_samples))
        mu = w @ X                                   # posterior mean
        z = rng.dirichlet(np.ones(K) * 1.3)          # arbitrary "model"

        lhs = sum(w[i] * bregman(X[i], z) for i in range(n_samples))
        floor = sum(w[i] * bregman(X[i], mu) for i in range(n_samples))
        model = bregman(mu, z)
        print(f"   {tr:5d} {lhs:14.8f} {floor:14.8f} {model:13.8f} "
              f"{abs(lhs - floor - model):12.2e}")
    print("\n   -> the split into (model-independent floor) + (model term) is a"
          "\n      property of Bregman divergences, not of squared error.\n")


# ---------------------------------------------------------------------------
# (B2) masked-diffusion floor
# ---------------------------------------------------------------------------
def entropy(p):
    q = p[p > 0]
    return float(-(q * np.log(q)).sum())


def cond_entropy(joint, i, U):
    """H(y_i | y_U) for a joint over n tokens, U a tuple of indices"""
    n = joint.ndim
    keep = tuple(sorted(set(U) | {i}))
    axes = tuple(a for a in range(n) if a not in keep)
    sub = joint.sum(axis=axes) if axes else joint
    pos = keep.index(i)
    if len(U) == 0:
        return entropy(sub.ravel())
    # H(i|U) = H(i,U) - H(U)
    axesU = (pos,)
    subU = sub.sum(axis=axesU)
    return entropy(sub.ravel()) - entropy(subU.ravel())


def floor_integrand(joint, m):
    """(1/m) * E_U[ sum_{i not in U} H(y_i | y_U) ],  U ~ each token kept w.p. 1-m"""
    n = joint.ndim
    total = 0.0
    for mask in itertools.product([0, 1], repeat=n):     # 1 = masked
        pr = np.prod([m if b else (1 - m) for b in mask])
        if pr == 0:
            continue
        U = tuple(i for i in range(n) if not mask[i])
        s = sum(cond_entropy(joint, i, U) for i in range(n) if mask[i])
        total += pr * s
    return total / m


def test_b2(n=3, K=3):
    print("=" * 72)
    print("(B2) masked-diffusion loss floor: value and schedule-invariance")
    print("=" * 72)

    # correlated joint distribution over n tokens
    joint = rng.dirichlet(np.ones(K ** n) * 0.35).reshape((K,) * n)
    H_joint = entropy(joint.ravel())
    H_marg = sum(entropy(joint.sum(axis=tuple(a for a in range(n) if a != i)).ravel())
                 for i in range(n))
    print(f"   tokens n={n}, alphabet K={K}")
    print(f"   H(y_1..y_n)        = {H_joint:.8f}")
    print(f"   sum_i H(y_i)       = {H_marg:.8f}   "
          f"(total correlation = {H_marg - H_joint:.4f} nats)\n")

    # integrate directly in m
    ms = np.linspace(1e-7, 1.0, 20001)
    vals = np.array([floor_integrand(joint, m) for m in ms])
    direct = np.trapezoid(vals, ms)
    print(f"   floor = int_0^1 integrand dm = {direct:.8f}")
    print(f"   relative error vs H(y_1..y_n) = "
          f"{abs(direct - H_joint) / H_joint:.3e}\n")

    # integrate in t, with weight -m'(t)/m(t), for several schedules
    print("   schedule-invariance: integrate in t with weight -m'(t)/m(t)")
    print("      schedule m(t)                  floor        rel.err vs H")
    schedules = {
        "linear      m=1-t": (lambda t: 1 - t, lambda t: -np.ones_like(t)),
        "cosine      m=cos(pi t/2)": (lambda t: np.cos(np.pi * t / 2),
                                      lambda t: -np.pi / 2 * np.sin(np.pi * t / 2)),
        "cubic       m=(1-t)^3": (lambda t: (1 - t) ** 3,
                                  lambda t: -3 * (1 - t) ** 2),
        "sqrt        m=sqrt(1-t)": (lambda t: np.sqrt(1 - t),
                                    lambda t: -0.5 / np.sqrt(np.clip(1 - t, 1e-12, None))),
    }
    def t_integral(mf, dmf, ts):
        mt = np.clip(mf(ts), 1e-12, 1.0)
        dmt = dmf(ts)
        # dLoss = (-dm/m) * E_U[sum_{i masked} H(y_i|y_U)] = (-dm) * integrand(m)
        integ = np.array([floor_integrand(joint, m) for m in mt])
        return np.trapezoid(-dmt * integ, ts)

    uniform = np.linspace(0.0, 1.0 - 1e-7, 6001)
    for name, (mf, dmf) in schedules.items():
        val = t_integral(mf, dmf, uniform)
        print(f"      {name:28s} {val:12.8f}   "
              f"{abs(val - H_joint) / H_joint:.3e}")

    print("\n   -> the discrete floor depends only on the endpoint mask rates,"
          "\n      not on the shape of the schedule, and equals H(data).\n")

    # The sqrt schedule is the one case where a UNIFORM t-grid is inadequate:
    # -m'/m = 1/(2(1-t)) has an integrable singularity at t = 1.  This is a
    # quadrature artefact, not a violation of the identity.  Resolving the
    # endpoint, either by clustering the grid there, or by the substitution
    # u = m(t) that the schedule-independence argument itself uses, recovers H.
    print("   the sqrt schedule on a uniform grid: a quadrature artefact")
    mf, dmf = schedules["sqrt        m=sqrt(1-t)"]
    val_uni = t_integral(mf, dmf, uniform)
    # grid clustered at t = 1 via t = 1 - s^2
    s = np.linspace(0.0, 1.0, 6001)
    clustered = 1.0 - s ** 2
    clustered = np.clip(clustered[::-1], 0.0, 1.0 - 1e-12)
    val_clu = t_integral(mf, dmf, clustered)
    print(f"      uniform grid in t              {val_uni:12.8f}   "
          f"{abs(val_uni - H_joint) / H_joint:.3e}")
    print(f"      grid clustered at t=1          {val_clu:12.8f}   "
          f"{abs(val_clu - H_joint) / H_joint:.3e}")
    print(f"      substitution u = m(t)          {direct:12.8f}   "
          f"{abs(direct - H_joint) / H_joint:.3e}")
    print("   -> the discrepancy is in the quadrature, not the identity.\n")


if __name__ == "__main__":
    test_b1()
    test_b2()
