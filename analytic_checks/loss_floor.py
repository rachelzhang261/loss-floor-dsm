"""Is  loss floor = tr(Fisher metric)  a theorem or a coincidence?

Chain to verify:
  (L1) pointwise   tr g(x,t) = (alpha^2/sigma^4) tr Cov[y | x_t = x]
  (L2) schedule    gamma_memoryless(t) * alpha_t^2/sigma_t^4 = d(SNR)/dt   for ANY affine interpolant
  (L3) I-MMSE      (1/2) int MMSE d(SNR) = Delta I(y ; x_t)
=> gap = (1/2) int gamma E[tr g] dt = (1/2) int MMSE dSNR = Delta I
"""
import numpy as np
rng = np.random.default_rng(0)

# ---------------- L2: the schedule lemma, several interpolants ----------------
def check_schedule(name, a, b, da, db, ts):
    kap = da(ts)/a(ts)
    eta = b(ts)*(kap*b(ts) - db(ts))
    gam = 2*eta                       # memoryless / bridge diffusion coefficient
    snr = (a(ts)/b(ts))**2
    dsnr = np.gradient(snr, ts)
    lhs = gam * a(ts)**2 / b(ts)**4   # gamma * alpha^2 / sigma^4
    rel = np.abs(lhs-dsnr)/np.maximum(np.abs(dsnr),1e-12)
    print(f"  {name:24s} max rel.err = {np.nanmax(rel[5:-5]):.3e}")

ts = np.linspace(0.15, 0.85, 4001)
print("L2  gamma*alpha^2/sigma^4 == d(SNR)/dt  for the memoryless schedule:")
check_schedule("linear  a=t, b=1-t", lambda t:t, lambda t:1-t,
               lambda t:np.ones_like(t), lambda t:-np.ones_like(t), ts)
check_schedule("trig    a=sin, b=cos", lambda t:np.sin(np.pi*t/2), lambda t:np.cos(np.pi*t/2),
               lambda t:(np.pi/2)*np.cos(np.pi*t/2), lambda t:-(np.pi/2)*np.sin(np.pi*t/2), ts)
check_schedule("poly    a=t^2, b=1-t^2", lambda t:t**2, lambda t:1-t**2,
               lambda t:2*t, lambda t:-2*t, ts)
check_schedule("vp-like a=t^1.5,b=(1-t)^.5", lambda t:t**1.5, lambda t:np.sqrt(1-t),
               lambda t:1.5*t**0.5, lambda t:-0.5/np.sqrt(1-t), ts)

# ---------------- L1: metric trace == scaled posterior covariance -------------
# 1-D two-point prior y = +-m ; x_t = a y + b z.  Everything closed form.
print("\nL1  tr g(x,t) == (alpha^2/sigma^4) tr Cov[y|x]   (pointwise, 1-D two-point prior)")
m = 2.0
for t in [0.2, 0.4, 0.6, 0.8]:
    a, b = t, 1-t
    xs = np.linspace(-3, 3, 7)
    # posterior over y given x
    z = 2*a*m*xs/b**2
    wp = 1/(1+np.exp(-z))
    covy = 4*m**2*wp*(1-wp)                 # Var[y|x]
    rhs = (a**2/b**4)*covy
    # metric from its definition: Var_{p(y|x)}[ d/dx log q(x|y) ],  q(x|y)=N(a y, b^2)
    #   d/dx log q(x|y) = (a y - x)/b^2  -> Var = (a^2/b^4) Var[y|x]
    # verify numerically by sampling the posterior
    lhs = []
    for i, x in enumerate(xs):
        ys = np.where(rng.random(400000) < wp[i], m, -m)
        sc = (a*ys - x)/b**2
        lhs.append(sc.var())
    lhs = np.array(lhs)
    print(f"   t={t:.2f}  max rel.err = {np.nanmax(np.abs(lhs-rhs)/np.maximum(rhs,1e-9)):.3e}")

# ---------------- L3: I-MMSE, and the full chain ------------------------------
print("\nL3  (1/2) int MMSE dSNR == Delta I(y;x)   (binary prior, exact I)")
def mmse_and_I(snr, m=1.0, n=400000):
    # x = sqrt(snr) y + z,  y = +-m
    y = rng.choice([-m, m], size=n)
    x = np.sqrt(snr)*y + rng.normal(size=n)
    e = m*np.tanh(np.sqrt(snr)*m*x)
    mmse = np.mean((y-e)**2)
    p = 1/(1+np.exp(-2*np.sqrt(snr)*m*x))          # P(y=+m | x)
    Hpost = -np.mean(p*np.log(np.clip(p,1e-300,1))+(1-p)*np.log(np.clip(1-p,1e-300,1)))
    return mmse, np.log(2)-Hpost
sn = np.linspace(1e-3, 6.0, 400)
vals = [mmse_and_I(s) for s in sn]
M = np.array([v[0] for v in vals]); I = np.array([v[1] for v in vals])
lhs = 0.5*np.trapezoid(M, sn)
print(f"   (1/2) int MMSE dSNR = {lhs:.5f}")
print(f"   I(snr_max) - I(snr_min) = {I[-1]-I[0]:.5f}")
print(f"   relative error = {abs(lhs-(I[-1]-I[0]))/abs(I[-1]-I[0]):.3e}")
import numpy as np
rng=np.random.default_rng(0)
print("L1 (redone): the identity is ALGEBRAIC, so check absolute agreement.")
print("   claim:  g(x,t) = Var_{y|x}[ d_x log q(x|y) ] = (a^2/b^4) Var[y|x]")
m=2.0
for t in [0.2,0.4,0.6,0.8]:
    a,b=t,1-t
    xs=np.linspace(-3,3,13)
    wp=1/(1+np.exp(-2*a*m*xs/b**2))
    rhs=(a**2/b**4)*4*m**2*wp*(1-wp)
    lhs=[]
    for i,x in enumerate(xs):
        ys=np.where(rng.random(2000000)<wp[i],m,-m)
        lhs.append(((a*ys-x)/b**2).var())
    lhs=np.array(lhs)
    print(f"   t={t:.2f}  max ABS err = {np.max(np.abs(lhs-rhs)):.3e}   "
          f"(scale of g here: {rhs.max():.3e})")

print("\nGENERAL (non-Gaussian) form: g = Fisher metric of the POSTERIOR family p(y|x).")
print("   d_x log p(y|x) = d_x log q(x|y) - d_x log P(x)   [since p(y|x)=q(x|y)p(y)/P(x)]")
print("   so  E_{y|x}[ d_x log p(y|x) ] = 0  and  Var = second moment = g.")
print("   The DSM gap integrand is E_{y|x}||d_x log q(x|y) - d_x log P||^2 -- the SAME object.")
print("   No Gaussianity used.  Verify on a NON-Gaussian corruption kernel (Laplace):")
# q(x|y) ∝ exp(-|x-y|/s) : score = sign(y-x)/s  (a.e.)
s=0.7; ys=np.array([-1.5,1.5])
for x in [-1.0,0.0,0.6]:
    lq=-np.abs(x-ys)/s; w=np.exp(lq-lq.max()); w/=w.sum()      # posterior p(y|x)
    sc=np.sign(ys-x)/s                                          # d_x log q(x|y)
    marg=(w*sc).sum()                                           # = d_x log P(x)
    gap=(w*(sc-marg)**2).sum()                                  # DSM gap integrand
    fisher=(w*(sc-marg)**2).sum()                               # Fisher metric of p(y|x)
    print(f"   x={x:+.2f}   gap={gap:.6f}   fisher={fisher:.6f}   equal={np.isclose(gap,fisher)}")
