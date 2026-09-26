"""Regenerates fig_lossfloor_identity.pdf (paper Fig. 2, App. D.3).

numpy + matplotlib only.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
OUT = "./"

# loss floor identity:  (1/2) int MMSE dlam  ==  Delta h(x_lam)
mus = np.array([-1.5, 1.5]); ws = np.array([0.5, 0.5]); sy = 0.4
def px(x, lam):
    m = np.sqrt(lam)*mus[None,:]; v = lam*sy**2+1.0
    return (ws[None,:]*np.exp(-0.5*(x[:,None]-m)**2/v)/np.sqrt(2*np.pi*v)).sum(1)
def h_of(lam, L=60.0, n=120001):
    x = np.linspace(-L,L,n); p = px(x,lam)
    return -np.trapezoid(np.where(p>0, p*np.log(np.maximum(p,1e-300)), 0.0), x)
def mmse_of(lam, L=60.0, n=120001):
    x = np.linspace(-L,L,n); m = np.sqrt(lam)*mus[None,:]; v = lam*sy**2+1.0
    comp = ws[None,:]*np.exp(-0.5*(x[:,None]-m)**2/v)/np.sqrt(2*np.pi*v)
    p = comp.sum(1); r = comp/np.maximum(p[:,None],1e-300)
    vk = sy**2/(1+lam*sy**2); mk = (mus[None,:]/sy**2+np.sqrt(lam)*x[:,None])*vk
    Ey = (r*mk).sum(1); Ey2 = (r*(vk+mk**2)).sum(1)
    return np.trapezoid(p*(Ey2-Ey**2), x)

lams = np.linspace(0.05, 40.0, 400)
M = np.array([mmse_of(l) for l in lams])
cum = 0.5*np.concatenate([[0], np.cumsum(np.diff(lams)*0.5*(M[1:]+M[:-1]))])
dh = np.array([h_of(l) for l in lams]); dh -= dh[0]

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
ax[0].plot(lams, M, lw=1.6, color="C0")
ax[0].set_xlabel(r"SNR $\lambda$"); ax[0].set_ylabel(r"$\mathbb{E}_{P}[\mathrm{tr}\,g]\,/\,\lambda$-units")
ax[0].set_title(r"metric trace $=$ MMSE", fontsize=9)
ax[0].set_xscale("log")
ax[1].plot(lams, cum, lw=2.4, color="C0", label=r"$\frac{1}{2}\int \mathrm{MMSE}\,d\lambda$")
ax[1].plot(lams, dh, lw=1.2, ls="--", color="C3", label=r"$\Delta h(x_\lambda)$")
ax[1].set_xlabel(r"SNR $\lambda$"); ax[1].set_ylabel("nats")
ax[1].set_title(r"loss floor $=$ entropy change", fontsize=9)
ax[1].legend(frameon=False, fontsize=8)
err = np.max(np.abs(cum-dh))/max(np.max(np.abs(dh)),1e-12)
ax[1].text(0.05,0.85, f"max rel. dev. {err:.1e}", transform=ax[1].transAxes, fontsize=7)
fig.savefig(OUT+"fig_lossfloor_identity.pdf"); plt.close(fig)
print("fig_lossfloor_identity.pdf written, max relative deviation", err)
