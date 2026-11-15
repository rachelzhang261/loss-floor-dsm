"""Illustrate raw DSM and excess on two different SNR ranges."""
import numpy as np
mus=np.array([-1.5,1.5]); sy=0.4

def stats(lam_lo, lam_hi, c, n=4000):
    """Return (raw DSM loss, excess) for a model whose score is c * exact score."""
    lams=np.linspace(lam_lo,lam_hi,n)
    raw=[]; exc=[]
    for lam in lams:
        x=np.linspace(-25,25,4001)
        m=np.sqrt(lam)*mus[None,:]; v=lam*sy**2+1.0
        comp=0.5*np.exp(-0.5*(x[:,None]-m)**2/v)/np.sqrt(2*np.pi*v)
        p=comp.sum(1); r=comp/np.maximum(p[:,None],1e-300)
        vk=sy**2/(1+lam*sy**2); mk=(mus[None,:]/sy**2+np.sqrt(lam)*x[:,None])*vk
        Ey=(r*mk).sum(1); Ey2=(r*(vk+mk**2)).sum(1)
        mmse=np.trapezoid(p*(Ey2-Ey**2),x)          # = floor integrand
        # In z = sqrt(SNR)*y + eps coordinates, the marginal score is
        # sqrt(SNR)*E[y|z] - z. The measure dSNR/SNR converts its squared
        # error to the same weighting as the integrated MMSE term.
        score = np.sqrt(lam)*Ey - x
        excess=(1-c)**2*np.trapezoid(p*score**2,x)/lam
        raw.append(mmse+excess); exc.append(excess)
    return 0.5*np.trapezoid(raw,lams), 0.5*np.trapezoid(exc,lams), 0.5*np.trapezoid(np.array(raw)-np.array(exc),lams)

print("Two schedules (SNR ranges), two models of DIFFERENT true quality.\n")
print(f"{'schedule':>22s} {'model':>7s} {'RAW loss':>10s} {'floor':>9s} {'EXCESS':>9s}")
rows={}

for name,(lo,hi) in {"A: SNR in [0.05, 8]":(0.05,8.0), "B: SNR in [0.05, 30]":(0.05,30.0)}.items():
    for c,label in [(0.97,"good"), (0.90,"bad")]:
        raw,exc,floor=stats(lo,hi,c)
        rows[(name,label)]=(raw,floor,exc)
        print(f"{name:>22s} {label:>7s} {raw:10.4f} {floor:9.4f} {exc:9.4f}")

print("""
Read the RAW column: the *good* model on schedule B reports a LARGER loss
(worse-looking) than the *bad* model on schedule A. The SNR range changes
both integrals. Compare models on the same range for a common objective.""")
a=rows[("B: SNR in [0.05, 30]","good")][0]; b=rows[("A: SNR in [0.05, 8]","bad")][0]
print(f"\n  raw(good,B)={a:.4f} > raw(bad,A)={b:.4f} ?  {a>b}   <-- inversion")
ea=rows[("B: SNR in [0.05, 30]","good")][2]; eb=rows[("A: SNR in [0.05, 8]","bad")][2]
print(f"  excess(good,B)={ea:.4f} < excess(bad,A)={eb:.4f} ?  {ea<eb}")
