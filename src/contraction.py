"""
[P1] TAUX DE CONTRACTION a priori -> rend la certification "anytime" QUANTITATIVE.

L'iteration scatter/connect est affine V <- M V + c. Par passivite, rho(M) < 1, et
   ||V(k) - V*||_K  ~  C rho^k     ->     demi-largeur(k)  <=  C' rho^k + plancher.
On (i) verifie que le point fixe = KCL pour TOUT alpha (la borne ne depend pas d'alpha),
(ii) mesure rho(alpha) et trouve l'alpha* le plus rapide, (iii) verifie l'enveloppe
C rho^k + plancher, (iv) regarde rho(n).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

np.seterr(over='ignore', divide='ignore', invalid='ignore')
SIG_IN = 10.0
KREF = (1.0/6.0)*np.array([[ 4.,-1.,-2.,-1.],[-1., 4.,-1.,-2.],
                           [-2.,-1., 4.,-1.],[-1.,-2.,-1., 4.]])
sig = lambda x, y: SIG_IN if (0.25 <= x <= 0.75 and 0.25 <= y <= 0.75) else 1.0

def assemble(n, coeff):
    npn=n+1; nid=lambda i,j: j*npn+i; rows,cols,vals=[],[],[]
    for cj in range(n):
        yc=(cj+0.5)/n
        for ci in range(n):
            xc=(ci+0.5)/n; cc=coeff(xc,yc)
            loc=[nid(ci,cj),nid(ci+1,cj),nid(ci+1,cj+1),nid(ci,cj+1)]
            for a in range(4):
                for b in range(4):
                    rows.append(loc[a]); cols.append(loc[b]); vals.append(cc*KREF[a,b])
    return sp.csr_matrix((vals,(rows,cols)),shape=(npn*npn,npn*npn)), nid

def graph(K):
    Kc=K.tocoo(); m=(Kc.row!=Kc.col)&(Kc.data<0)
    s,d,g=Kc.row[m],Kc.col[m],-Kc.data[m]
    key={(a,b):e for e,(a,b) in enumerate(zip(s,d))}
    rev=np.array([key[(b,a)] for a,b in zip(s,d)])
    return s,d,g,rev

def _dir(K,npn,nid,orient):
    N=npn*npn; u=np.zeros(N); fx={}
    n=npn-1
    for k in range(npn):
        if orient=='LR': fx[nid(0,k)]=0.0; fx[nid(n,k)]=1.0
        else:            fx[nid(k,0)]=0.0; fx[nid(k,n)]=1.0
    fi=np.array(sorted(fx)); u[fi]=[fx[k] for k in fi]
    fr=np.setdiff1d(np.arange(N),fi); u[fr]=spla.spsolve(K[fr][:,fr].tocsc(),-K[fr][:,fi]@u[fi])
    return u, fx

def run(n, coeff, orient, alpha, nit):
    K,nid=assemble(n,coeff); npn=n+1; Nn=npn*npn
    s,d,g,rev=graph(K); Vstar,fx=_dir(K,npn,nid,orient)
    is_dir=np.zeros(Nn,bool); phi=np.zeros(Nn)
    for k,v in fx.items(): is_dir[k]=True; phi[k]=v
    den=np.bincount(s,weights=g,minlength=Nn); inc=np.zeros(len(s))
    V=np.zeros(Nn); V[is_dir]=phi[is_dir]
    eK=np.empty(nit); Grun=np.empty(nit)
    for it in range(nit):
        num=np.bincount(s,weights=g*inc,minlength=Nn)
        Vn=np.where(den>0,2.0*num/np.where(den>0,den,1.0),0.0); Vn[is_dir]=phi[is_dir]
        refl=Vn[s]-inc; inc=alpha*refl[rev]+(1.0-alpha)*refl; V=Vn
        e=V-Vstar; eK[it]=np.sqrt(max(e@(K@e),0.0)); Grun[it]=float(V@(K@V))
    res=np.max(np.abs((K@V)[~is_dir]))
    return eK, Grun, res

def rho_fit(eK):
    """Taux de contraction = pente de log(eK) sur la zone log-lineaire PROPRE
    (apres le transitoire initial, au-dessus du plancher machine)."""
    e=np.asarray(eK); e0=e[0]; efloor=max(e.min(), 1e-13); N=len(e)
    mask=(e < 0.3*e0) & (e > 1e3*efloor)              # zone geometrique nette
    if mask.sum() < 20:                                # repli 1 : au-dessus du plancher
        mask=(e < 0.5*e0) & (e > 10*efloor)
    if mask.sum() < 20:                                # repli 2 : fenetre fixe (cas tres lent)
        mask=np.zeros(N,bool); mask[int(0.3*N):int(0.95*N)]=True
    x=np.flatnonzero(mask); y=np.log(e[x]+1e-300)
    return float(np.exp(np.polyfit(x, y, 1)[0]))

# ----------------------------------------------------------------------
# (i) le point fixe = KCL pour tout alpha (residu -> 0)
# ----------------------------------------------------------------------
print("=== (i) Independance de la borne vis-a-vis d'alpha (residu KCL) ===")
for a in [0.2, 0.35, 0.5, 0.65, 0.8]:
    _,_,res = run(32, sig, 'LR', a, 4000)
    print(f"  alpha={a:.2f}  residu KCL = {res:.2e}")

# ----------------------------------------------------------------------
# (ii) rho(alpha) et alpha* optimal, pour les deux reseaux
# ----------------------------------------------------------------------
print("\n=== (ii) Taux de contraction rho(alpha), n=32 ===")
alphas=np.linspace(0.15,0.97,18)
rho_phi=[rho_fit(run(32, sig, 'LR', a, 12000)[0]) for a in alphas]
rho_psi=[rho_fit(run(32, lambda x,y:1/sig(x,y), 'BT', a, 12000)[0]) for a in alphas]
rho_phi=np.array(rho_phi); rho_psi=np.array(rho_psi)
ap=alphas[np.argmin(rho_phi)]; aps=alphas[np.argmin(rho_psi)]
print(f"  reseau phi : alpha* = {ap:.2f}  (rho_min = {rho_phi.min():.4f})  vs rho(0.5)={rho_phi[np.argmin(np.abs(alphas-0.5))]:.4f}")
print(f"  reseau psi : alpha* = {aps:.2f}  (rho_min = {rho_psi.min():.4f})  vs rho(0.5)={rho_psi[np.argmin(np.abs(alphas-0.5))]:.4f}")

# ----------------------------------------------------------------------
# (iii) enveloppe  demi-largeur(k) <= C rho^k + plancher  (alpha=0.5 vs alpha*)
# ----------------------------------------------------------------------
print("\n=== (iii) Enveloppe C rho^k + plancher ===")
def halfwidth_hist(alpha, nit):
    _,Gup,_=run(32, sig, 'LR', alpha, nit)
    _,Rup,_=run(32, lambda x,y:1/sig(x,y), 'BT', alpha, nit)
    return 0.5*(Gup-1.0/Rup)
NIT=8000
hw05=halfwidth_hist(0.5, NIT)
hwopt=halfwidth_hist(float(ap), NIT)
floor=min(hw05[-1], hwopt[-1])
print(f"  plancher (n=32) ~ {floor:.3e} ; demi-largeur initiale {hw05[0]:.3e}")

# ----------------------------------------------------------------------
# (iv) rho(n) a alpha=0.5
# ----------------------------------------------------------------------
print("\n=== (iv) rho(n) a alpha=0.5 (reseau psi) ===")
ns=[16,24,32,48,64]
rho_n=[rho_fit(run(nn, lambda x,y:1/sig(x,y),'BT',0.5, 30000)[0]) for nn in ns]
for nn,r in zip(ns,rho_n):
    print(f"  n={nn:3d}  rho={r:.4f}  (1-rho={1-r:.2e})")

# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
ax[0].plot(alphas, rho_phi, 'o-', color='C3', label='reseau phi')
ax[0].plot(alphas, rho_psi, 's-', color='C0', label='reseau psi')
ax[0].axvline(ap, color='C3', ls=':', lw=1); ax[0].axvline(aps, color='C0', ls=':', lw=1)
ax[0].set_xlabel('alpha (coeff. de transmission)'); ax[0].set_ylabel('taux de contraction rho')
ax[0].set_title('(ii) rho(alpha) : un alpha* optimal'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

ks=np.arange(NIT)
ax[1].semilogy(ks, hw05-floor+1e-12, color='C1', label=f'alpha=0.5')
ax[1].semilogy(ks, hwopt-floor+1e-12, color='C2', label=f'alpha*={ap:.2f}')
C=hw05[0]; ax[1].semilogy(ks, C*rho_phi[np.argmin(np.abs(alphas-0.5))]**ks, 'k--', lw=1,
                          label='enveloppe C rho^k')
ax[1].set_xlabel('iteration k'); ax[1].set_ylabel('half-width - floor')
ax[1].set_title('(iii) decroissance geometrique + speedup alpha*'); ax[1].legend(fontsize=8)
ax[1].grid(alpha=.3, which='both'); ax[1].set_ylim(1e-6, None)

ax[2].plot(ns, [1-r for r in rho_n], 'o-', color='C4')
ax[2].set_xlabel('n'); ax[2].set_ylabel('1 - rho'); ax[2].set_yscale('log'); ax[2].set_xscale('log')
ax[2].set_title('(iv) 1-rho ~ O(1/n^?) (cout des iterations)'); ax[2].grid(alpha=.3, which='both')
p=np.polyfit(np.log(ns), np.log([1-r for r in rho_n]),1)[0]
ax[2].text(0.05,0.1,f'pente ~ {p:.2f}', transform=ax[2].transAxes)

fig.suptitle('Contraction rate: the anytime certification becomes quantitative', fontsize=11)
fig.tight_layout(); fig.savefig('contraction_result.png', dpi=130)
print(f"\n1-rho ~ O(n^{p:.2f})\nFigure -> contraction_result.png")
