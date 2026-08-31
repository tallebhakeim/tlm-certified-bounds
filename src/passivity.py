"""
PASSIVITE & CERTIFICATION "ANYTIME" — le differenciateur propre au TLM.

These :
  A chaque iteration scatter/connect, l'itere V(k) respecte EXACTEMENT les conditions
  de Dirichlet (les noeuds-source sont epingles). Par le principe du minimum d'energie,
  pour TOUT champ v a CL de Dirichlet correctes :   v^T K v  >=  G_discret  >=  G_vrai.
  Donc :
     G_upper(k) = V_phi(k)^T K_sigma V_phi(k)  >=  G_vrai          (borne sup. valide A TOUT k)
     G_lower(k) = 1 / [V_psi(k)^T K_rho V_psi(k)] <= G_vrai        (borne inf. valide A TOUT k)
  -> [G_lower(k), G_upper(k)] CONTIENT G_vrai des la 1ere iteration (certification "anytime").
  La PASSIVITE (scattering dissipatif) fait DECROITRE l'energie de l'itere de facon
  monotone -> l'intervalle se RESSERRE monotonement vers l'intervalle certifie final.

On verifie numeriquement : (i) G_lo(k) <= G_vrai <= G_up(k) a TOUTE iteration (aucune
violation) ; (ii) monotonie ; (iii) l'erreur en norme d'energie ||V(k)-V*||_K decroit
monotone (contraction passive = fonction de Lyapunov).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

np.seterr(over='ignore', divide='ignore', invalid='ignore')
SIG_IN = 10.0; ALPHA = 0.5
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

def direct_solution(K, npn, nid, orient):
    """Solution stationnaire exacte V* (pour la norme d'erreur de Lyapunov)."""
    N=npn*npn; u=np.zeros(N); fx={}
    npn_=int(round(N**0.5))
    for k in range(npn):
        if orient=='LR': fx[nid(0,k)]=0.0; fx[nid(npn-1,k)]=1.0
        else:            fx[nid(k,0)]=0.0; fx[nid(k,npn-1)]=1.0
    fi=np.array(sorted(fx)); u[fi]=[fx[k] for k in fi]
    fr=np.setdiff1d(np.arange(N),fi)
    u[fr]=spla.spsolve(K[fr][:,fr].tocsc(), -K[fr][:,fi]@u[fi])
    return u

def tlm_history(n, coeff, orient, n_iter):
    """Renvoie G_running(k) = V(k)^T K V(k) et err_K(k)=||V(k)-V*||_K, pour chaque k."""
    K, nid = assemble(n, coeff); npn=n+1; Nn=npn*npn
    s,d,g,rev = graph(K)
    phi=np.zeros(Nn); is_dir=np.zeros(Nn,bool)
    for k in range(npn):
        if orient=='LR':
            is_dir[nid(0,k)]=True; phi[nid(0,k)]=0.0; is_dir[nid(n,k)]=True; phi[nid(n,k)]=1.0
        else:
            is_dir[nid(k,0)]=True; phi[nid(k,0)]=0.0; is_dir[nid(k,n)]=True; phi[nid(k,n)]=1.0
    Vstar = direct_solution(K, npn, nid, orient)
    den=np.bincount(s,weights=g,minlength=Nn); inc=np.zeros(len(s))
    V=np.zeros(Nn); V[is_dir]=phi[is_dir]
    Grun=np.empty(n_iter); errK=np.empty(n_iter)
    for it in range(n_iter):
        num=np.bincount(s,weights=g*inc,minlength=Nn)
        Vn=np.where(den>0, 2.0*num/np.where(den>0,den,1.0), 0.0); Vn[is_dir]=phi[is_dir]
        refl=Vn[s]-inc; inc=ALPHA*refl[rev]+(1.0-ALPHA)*refl; V=Vn
        Grun[it]=float(V@(K@V))
        e=V-Vstar; errK[it]=float(np.sqrt(max(e@(K@e),0.0)))
    return Grun, errK

# ----------------------------------------------------------------------
n = 32; NIT = 9000
print("Reference fine pour G_vrai...")
Kf,nidf=assemble(256, sig); Vf=direct_solution(Kf,257,nidf,'LR'); G_up_fine=float(Vf@(Kf@Vf))
Krf,nidrf=assemble(256, lambda x,y:1.0/sig(x,y)); Vrf=direct_solution(Krf,257,nidrf,'BT')
G_lo_fine=1.0/float(Vrf@(Krf@Vrf)); G_true=0.5*(G_up_fine+G_lo_fine)
print(f"G_vrai ~ {G_true:.6f}\n")

print(f"Reseaux TLM n={n}, {NIT} iterations...")
Gup_k, eU = tlm_history(n, sig, 'LR', NIT)                 # reseau phi -> borne sup courante
Rup_k, eL = tlm_history(n, lambda x,y:1.0/sig(x,y), 'BT', NIT)
Glo_k = 1.0/Rup_k                                          # reseau psi -> borne inf courante

# --- (i) garantie "anytime" : aucune violation ---
viol_up = np.sum(Gup_k < G_true - 1e-9)
viol_lo = np.sum(Glo_k > G_true + 1e-9)
print(f"(i)  Violations borne sup (G_up(k) < G_vrai) : {viol_up} / {NIT}")
print(f"     Violations borne inf (G_lo(k) > G_vrai) : {viol_lo} / {NIT}")
print(f"     -> certification ANYTIME {'OK (0 violation)' if viol_up==0 and viol_lo==0 else 'ECHEC'}")

# --- (ii) monotonie ---
mono_up = np.all(np.diff(Gup_k) <= 1e-12); mono_lo = np.all(np.diff(Glo_k) >= -1e-12)
print(f"(ii) G_up(k) non-croissante : {mono_up} ;  G_lo(k) non-decroissante : {mono_lo}")

# --- (iii) Lyapunov : ||V(k)-V*||_K decroit monotone ---
mono_eU = np.all(np.diff(eU) <= 1e-12)
print(f"(iii) ||V(k)-V*||_K (reseau phi) monotone decroissante : {mono_eU}")

# nombre d'iterations pour atteindre une demi-largeur cible
for tgt in [1e-2, 1e-3, 5e-4]:
    half=0.5*(Gup_k-Glo_k); kk=np.argmax(half<=tgt) if np.any(half<=tgt) else -1
    print(f"     demi-largeur <= {tgt:.0e} atteinte a l'iteration {kk}")

# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
ks=np.arange(NIT)
ax[0].fill_between(ks, Glo_k, Gup_k, color='gray', alpha=0.25, label='certified interval (anytime)')
ax[0].plot(ks, Gup_k, color='C3', label='G_upper(k) — reseau phi')
ax[0].plot(ks, Glo_k, color='C0', label='G_lower(k) — reseau psi')
ax[0].axhline(G_true, color='k', ls='--', lw=1, label='G_true')
ax[0].set_xlim(0, 2500); ax[0].set_ylim(G_true-0.04, G_true+0.06)
ax[0].set_xlabel('iteration scatter/connect'); ax[0].set_ylabel('G')
ax[0].set_title('Anytime certification: G_true always bracketed'); ax[0].legend(fontsize=8)

ax[1].semilogy(ks, 0.5*(Gup_k-Glo_k), color='C2')
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('certified half-width')
ax[1].set_title('Resserrement monotone (passivite)'); ax[1].grid(True, which='both', alpha=0.3)

ax[2].semilogy(ks, eU/eU[0], color='C4', label='reseau phi')
ax[2].semilogy(ks, eL/eL[0], color='C1', label='reseau psi')
ax[2].set_xlabel('iteration'); ax[2].set_ylabel('||V(k)-V*||_K (norm.)')
ax[2].set_title('Lyapunov : contraction passive (monotone)'); ax[2].legend(fontsize=8)
ax[2].grid(True, which='both', alpha=0.3)

fig.suptitle('Passivity -> anytime certification: guaranteed bounds at every TLM iteration',
             fontsize=11)
fig.tight_layout(); fig.savefig('passivity_result.png', dpi=130)
print("\nFigure -> passivity_result.png")
