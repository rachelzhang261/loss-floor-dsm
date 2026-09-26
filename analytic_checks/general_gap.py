"""GENERAL (non-affine) proof check of the gap identity.

Claim chain, for ANY corruption diffusion dx = f(x,t)dt + sqrt(D(t)) dW and ANY data:

 (A)  E_{P_t}[tr g]  =  E_y J(q_t(.|y))  -  J(P_t)          [pure algebra, any kernel]
 (B)  dI(y;x_t)/dt   = -(D/2) [ E_y J(q_t(.|y)) - J(P_t) ]  [de Bruijn for a diffusion]
 => gap_w = (1/2) int w(t) E[tr g] dt = - int (w/D) dI/dt dt ;  w = D  =>  gap = Delta I.

Here f(x) = -a x^3 is NONLINEAR, so nothing is affine/Gaussian.
Verified by solving the Fokker-Planck equation on a grid.
"""
import numpy as np

L, N = 12.0, 3001
x = np.linspace(-L, L, N); dx = x[1]-x[0]
D = 0.6
a = 0.35
f = -a * x**3                      # NONLINEAR drift

ys = np.array([-2.0, -0.5, 1.8])   # data atoms
py = np.array([0.3, 0.45, 0.25])

def fp_step(p, dt):
    flux = f * p
    dflux = (np.roll(flux,-1) - np.roll(flux,1)) / (2*dx)
    lap = (np.roll(p,-1) - 2*p + np.roll(p,1)) / dx**2
    p = p + dt * (-dflux + 0.5*D*lap)
    p[0] = p[-1] = 0.0
    p = np.maximum(p, 0.0)
    return p / (p.sum()*dx)

def J(p):
    dp = (np.roll(p,-1) - np.roll(p,1)) / (2*dx)
    m = p > 1e-12
    return np.sum(np.where(m, dp**2/np.maximum(p,1e-300), 0.0)) * dx

def H(p):
    m = p > 1e-300
    return -np.sum(np.where(m, p*np.log(np.maximum(p,1e-300)), 0.0)) * dx

s0 = 0.25
qs = [np.exp(-0.5*(x-y)**2/s0**2)/(np.sqrt(2*np.pi)*s0) for y in ys]
qs = [q/(q.sum()*dx) for q in qs]

dt, nsteps, every = 2e-5, 30000, 1500
rec = []
for n in range(nsteps+1):
    if n % every == 0:
        P = sum(w*q for w, q in zip(py, qs))
        Jq = sum(w*J(q) for w, q in zip(py, qs))
        Hq = sum(w*H(q) for w, q in zip(py, qs))
        rec.append((n*dt, H(P)-Hq, Jq - J(P)))
    qs = [fp_step(q, dt) for q in qs]

rec = np.array(rec)
t, I, gapint = rec[:,0], rec[:,1], rec[:,2]
dIdt = np.gradient(I, t)
pred = -0.5*D*gapint
print("Nonlinear drift f(x) = -0.35 x^3, D = 0.6, three data atoms.\n")
print(f"{'t':>8s} {'I(y;x_t)':>10s} {'dI/dt':>12s} {'-(D/2)*E[tr g]':>16s} {'rel.err':>9s}")
for i in range(2, len(t)-2, 3):
    r = abs(dIdt[i]-pred[i])/max(abs(pred[i]),1e-12)
    print(f"{t[i]:8.4f} {I[i]:10.4f} {dIdt[i]:12.5f} {pred[i]:16.5f} {r:9.2e}")
k = slice(2, len(t)-2)
print(f"\n  median relative error over the interior: "
      f"{np.median(np.abs(dIdt[k]-pred[k])/np.abs(pred[k])):.3e}")
print(f"  gap with w=D:   -int (dI/dt) dt = {-np.trapezoid(dIdt[k], t[k]):.5f}")
print(f"                  I(t_0) - I(t_1) = {I[k][0]-I[k][-1]:.5f}")
