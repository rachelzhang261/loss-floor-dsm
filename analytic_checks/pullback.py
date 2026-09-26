"""
pullback.py
============

Tests the reordered derivation of the latent Fisher-Rao metric:

    ambient second variation  ->  finite-parameter immersion  ->  pullback.

Three claims are checked, on three reference kernels that differ only in the
structure of the coupling map A(x,t):

  (T1) The loss floor equals the trace of the PULLBACK of the ambient Fisher
       form, for ANY kernel of exponential-family type -- no Gaussianity, no
       affinity.  This is the ambient Fisher form plus the endpoint-split
       proposition, and should hold in all cases.

  (T2) The e-embedding curvature of the latent immersion.  This is what
       distinguishes flat from curved latent geometry.

  (T3) Whether the image of the immersion lies in an affine subspace of the
       natural-parameter space (the coordinate-free version of flatness).

Cases
-----
  A  Gaussian reference.              s = (y, -y^2/2),        A = (x/s2, 1/s2)
  B  Gaussian with NONLINEAR drift.   s = (y, -y^2/2),        A = (h(x)/s2, 1/s2)
                                      h(x) = x + c x^3
  C  Cubic reference, codimension 1.  s = (y, -y^2/2, y^3),   A = (x/s2, 1/s2,
                                                                   g x^2/s2^2)

The interesting prediction is that B is FLAT despite the nonlinear drift -- a
nonlinear A(.,t) that is still a diffeomorphism onto an open set is a
reparametrization of latent space, not a change of geometry.  Curvature needs
CODIMENSION, i.e. a sufficient statistic richer than the latent dimension.
That is case C.

Everything is done by quadrature on a bounded y-grid; the reference kernel need
not be normalized in x (it is a propagator, not a conditional law).
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ----------------------------------------------------------------------------
# carrier: a bimodal data density on a bounded grid
# ----------------------------------------------------------------------------
YMAX, NY = 6.0, 4001
Y = np.linspace(-YMAX, YMAX, NY)
DY = Y[1] - Y[0]


def carrier(y):
    return (0.6 * np.exp(-0.5 * ((y - 1.3) / 0.5) ** 2) / 0.5
            + 0.4 * np.exp(-0.5 * ((y + 1.1) / 0.7) ** 2) / 0.7)


NU = carrier(Y)
NU /= np.trapezoid(NU, Y)


# ----------------------------------------------------------------------------
# the three cases: sufficient statistic s(y) and coupling map A(x,t)
# ----------------------------------------------------------------------------
def sigma2(t):
    """noise variance schedule"""
    return 0.15 + 1.6 * t


C_DRIFT = 0.30   # nonlinearity of the drift in case B
GAMMA = 0.02     # codimension-inducing coupling in case C


def make_case(name):
    if name == "A":
        S = np.stack([Y, -0.5 * Y ** 2])

        def A(x, t):
            s2 = sigma2(t)
            return np.array([x / s2, 1.0 / s2])

    elif name == "B":
        S = np.stack([Y, -0.5 * Y ** 2])

        def A(x, t):
            s2 = sigma2(t)
            h = x + C_DRIFT * x ** 3
            return np.array([h / s2, 1.0 / s2])

    elif name == "C":
        S = np.stack([Y, -0.5 * Y ** 2, Y ** 3])

        def A(x, t):
            s2 = sigma2(t)
            return np.array([x / s2, 1.0 / s2, GAMMA * x ** 2 / s2 ** 2])

    else:
        raise ValueError(name)

    return S, A


# ----------------------------------------------------------------------------
# core quantities
# ----------------------------------------------------------------------------
def posterior(theta, S):
    """p(y | theta) prop to nu(y) exp(theta . s(y)); returns normalized density"""
    ln_w = theta @ S
    shift = ln_w.max()
    w = NU * np.exp(ln_w - shift)
    Z = np.trapezoid(w, Y)
    return w / Z, np.log(Z) + shift


def eta_and_cov(theta, S):
    p, _ = posterior(theta, S)
    eta = np.trapezoid(p * S, Y, axis=1)
    Sc = S - eta[:, None]
    cov = np.trapezoid(p * (Sc[:, None, :] * Sc[None, :, :]), Y, axis=2)
    return eta, cov, p


def ln_psi(x, t, S, A):
    _, lz = posterior(A(x, t), S)
    return lz


def d_num(f, z, h):
    return (f(z + h) - f(z - h)) / (2 * h)


def d2_num(f, z, h):
    return (f(z + h) - 2 * f(z) + f(z - h)) / h ** 2


# ----------------------------------------------------------------------------
# T1: loss floor  ==  trace of the pullback
# ----------------------------------------------------------------------------
def test_floor(x, t, S, A, h=1e-4):
    """
    LHS (direct):  E_p[ (d_x ln G)^2 ] - (d_x ln psi)^2
                   with d_x ln G obtained by finite-differencing ln G(y|x,t)
                   in x, pointwise in y.  Independent of the A-Jacobian route.
    RHS (pullback): d_x A^a  d_x A^b  Cov_p[s_a, s_b]
    """
    theta = A(x, t)
    p, _ = posterior(theta, S)

    # d_x ln G(y|x,t) by finite difference of the log-kernel itself
    lnG_p = A(x + h, t) @ S
    lnG_m = A(x - h, t) @ S
    dlnG = (lnG_p - lnG_m) / (2 * h)

    E1 = np.trapezoid(p * dlnG, Y)
    E2 = np.trapezoid(p * dlnG ** 2, Y)
    lhs = E2 - E1 ** 2

    # marginal-score consistency: E_p[d_x ln G] should equal d_x ln psi
    dlnpsi = d_num(lambda z: ln_psi(z, t, S, A), x, h)

    # pullback
    dA = d_num(lambda z: A(z, t), x, h)
    _, cov, _ = eta_and_cov(theta, S)
    rhs = dA @ cov @ dA

    return lhs, rhs, E1, dlnpsi


# ----------------------------------------------------------------------------
# T2: e-embedding curvature  H = Pi^perp [ d_i d_j A ]
# ----------------------------------------------------------------------------
def embedding_curvature(x, t, S, A, h=1e-3):
    theta = A(x, t)
    _, M, _ = eta_and_cov(theta, S)          # ambient Fisher metric Cov[s]

    e1 = d_num(lambda z: A(z, t), x, h)      # d/dx
    e2 = d_num(lambda z: A(x, z), t, h)      # d/dt
    T = np.stack([e1, e2], axis=1)           # m x 2 tangent frame

    d2 = {
        "xx": d2_num(lambda z: A(z, t), x, h),
        "tt": d2_num(lambda z: A(x, z), t, h),
        "xt": (A(x + h, t + h) - A(x + h, t - h)
               - A(x - h, t + h) + A(x - h, t - h)) / (4 * h * h),
    }

    # M-orthogonal projection onto span(T)
    Gm = T.T @ M @ T
    out = {}
    for k, v in d2.items():
        coef = np.linalg.solve(Gm, T.T @ M @ v)
        perp = v - T @ coef
        out[k] = np.sqrt(max(perp @ M @ perp, 0.0))
    return out


# ----------------------------------------------------------------------------
# T3: is the image of the immersion contained in an affine subspace?
# ----------------------------------------------------------------------------
def affine_hull(S, A, nx=25, nt=25):
    xs = np.linspace(-1.2, 1.2, nx)
    ts = np.linspace(0.05, 0.95, nt)
    P = np.array([A(x, t) for x in xs for t in ts])
    P = P - P.mean(0)
    sv = np.linalg.svd(P, compute_uv=False)
    return sv / sv[0]


# ----------------------------------------------------------------------------
# T4: how the latent tangent plane sits inside the full bridge tangent plane
#
# Ambient tangent space at a bridge = tilt directions dw(y), an infinite-
# dimensional function space, with Fisher inner product <a,b> = Cov_p[a,b].
# The latent tangent plane is the image of d(iota), i.e. span{ d_i A^a s_a }
# = span{ s_1..s_m } when dA has full rank.  We measure what fraction of the
# Fisher norm of a given tilt direction is visible to latent motion.
# ----------------------------------------------------------------------------
def visible_fraction(x, t, S, A, probes, h=1e-4):
    theta = A(x, t)
    p, _ = posterior(theta, S)

    def ip(a, b):
        ma = np.trapezoid(p * a, Y)
        mb = np.trapezoid(p * b, Y)
        return np.trapezoid(p * (a - ma) * (b - mb), Y)

    # basis of the latent tangent plane, as tilt directions
    dAx = d_num(lambda z: A(z, t), x, h)
    dAt = d_num(lambda z: A(x, z), t, h)
    B = [dAx @ S, dAt @ S]

    Gm = np.array([[ip(bi, bj) for bj in B] for bi in B])
    out = {}
    for nm, w in probes.items():
        v = np.array([ip(bi, w) for bi in B])
        num = v @ np.linalg.solve(Gm, v)
        den = ip(w, w)
        out[nm] = num / den if den > 1e-14 else np.nan
    return out


PROBES = {
    "y   (linear)": Y,
    "y^2 (quadratic)": Y ** 2,
    "y^3 (cubic)": Y ** 3,
    "y^4 (quartic)": Y ** 4,
    "sin 3y": np.sin(3 * Y),
    "tanh 2y": np.tanh(2 * Y),
}


# ----------------------------------------------------------------------------
# T5: the Markov / hierarchical structure of the encoder
#
# For an affine encoder the natural parameter is
#       theta(x,t) = SNR_t * (x_tilde, 1),
# i.e. latent spacetime is a CONE in natural-parameter space: the spatial
# coordinate is the DIRECTION of theta and the time coordinate is its RADIUS.
# The Markov semigroup acts by dilation along the radius, and the hierarchy
# (information monotonicity) is the statement that the radius measures how much
# the latent knows about the endpoint.
#
# Two checks:
#   (a) d_t A is parallel to A  (the time direction is radial)
#   (b) K(r) := KL(p_{rn} || p_0) satisfies  dK/dr = r * g_rr,
#       where g_rr = n^T Cov_p[s] n is the radial component of the metric.
#       So the information the latent carries about the endpoint is the radial
#       integral of the radial metric -- a pointwise ancestor of the loss-floor
#       theorem.
# ----------------------------------------------------------------------------
def radial_structure(S, A, x=0.4, ts=(0.15, 0.35, 0.55, 0.80)):
    rows = []
    for t in ts:
        a = A(x, t)
        da = d_num(lambda z: A(x, z), t, 1e-4)
        cos = abs(a @ da) / (np.linalg.norm(a) * np.linalg.norm(da) + 1e-300)
        rows.append((t, np.sqrt(max(1 - cos ** 2, 0.0))))
    return rows


def kl_from_carrier(theta, S):
    p, lz = posterior(theta, S)
    _, lz0 = posterior(np.zeros_like(theta), S)
    eta = np.trapezoid(p * S, Y, axis=1)
    return theta @ eta - (lz - lz0)


def radial_information(S, A, x=0.4, t=0.35, h=1e-4):
    """check dK/dr == r * g_rr along the ray through A(x,t)"""
    a = A(x, t)
    r0 = np.linalg.norm(a)
    n = a / r0
    out = []
    for r in [0.25 * r0, 0.5 * r0, r0, 1.5 * r0]:
        dK = (kl_from_carrier((r + h) * n, S)
              - kl_from_carrier((r - h) * n, S)) / (2 * h)
        _, cov, _ = eta_and_cov(r * n, S)
        grr = n @ cov @ n
        out.append((r, kl_from_carrier(r * n, S), dK, r * grr))
    return out


# ----------------------------------------------------------------------------
def main():
    pts = [(-0.8, 0.15), (-0.3, 0.35), (0.2, 0.55), (0.9, 0.80)]

    for name in ["A", "B", "C"]:
        S, A = make_case(name)
        label = {"A": "Gaussian (affine drift)",
                 "B": f"Gaussian, NONLINEAR drift h(x)=x+{C_DRIFT}x^3",
                 "C": f"cubic family, codim 1 (gamma={GAMMA})"}[name]
        print("=" * 74)
        print(f"CASE {name}:  {label}")
        print(f"  latent dim = 2 (x,t)   ambient dim m = {S.shape[0]}")
        print("=" * 74)

        print("\n  (T1) loss floor  vs  trace of pullback")
        print("   x      t      floor(direct)   tr(pullback)    rel.err "
              "  |E[dlnG]-dln psi|")
        worst = 0.0
        for x, t in pts:
            lhs, rhs, E1, dlp = test_floor(x, t, S, A)
            rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
            worst = max(worst, rel)
            print(f"  {x:5.2f} {t:5.2f}   {lhs:13.6e} {rhs:13.6e} "
                  f"{rel:10.2e}   {abs(E1 - dlp):10.2e}")
        print(f"   -> worst relative error on the floor identity: {worst:.2e}")

        print("\n  (T2) e-embedding curvature  ||Pi_perp d_i d_j A||_M")
        for x, t in pts:
            c = embedding_curvature(x, t, S, A)
            print(f"  {x:5.2f} {t:5.2f}   xx={c['xx']:.3e}  "
                  f"tt={c['tt']:.3e}  xt={c['xt']:.3e}")

        print("\n  (T3) normalized singular values of the centred image")
        sv = affine_hull(S, A)
        print("       ", np.array2string(sv, precision=3))
        n_eff = int((sv > 1e-6).sum())
        print(f"        affine hull dimension = {n_eff}  (latent dim = 2)")
        if n_eff <= 2:
            print("        -> image lies in a 2-plane: immersion is e-AFFINE, "
                  "latent geometry FLAT")
        else:
            print("        -> image does NOT lie in a 2-plane: immersion is "
                  "CURVED")

        print("\n  (T4) fraction of each tilt direction visible to latent "
              "motion")
        print("        (Fisher-orthogonal projection onto the latent tangent "
              "plane)")
        hdr = "   direction        " + "".join(
            f"  x={x:+.1f},t={t:.2f}" for x, t in pts)
        print(hdr)
        for nm in PROBES:
            row = f"   {nm:<18}"
            for x, t in pts:
                f = visible_fraction(x, t, S, A, PROBES)[nm]
                row += f"  {f:13.4f}"
            print(row)

        print("\n  (T5a) is the time direction radial in theta?  "
              "|sin angle(A, d_t A)|")
        for t, s in radial_structure(S, A):
            print(f"        t={t:.2f}   {s:.3e}")

        print("\n  (T5b) information vs radial metric along the ray")
        print("        r        K(r)=KL(p_r||nu)     dK/dr        r*g_rr"
              "     rel.err")
        for r, K, dK, rg in radial_information(S, A):
            rel = abs(dK - rg) / max(abs(rg), 1e-30)
            print(f"     {r:8.3f} {K:15.6e} {dK:13.6e} {rg:13.6e} {rel:10.2e}")
        print()


if __name__ == "__main__":
    main()
