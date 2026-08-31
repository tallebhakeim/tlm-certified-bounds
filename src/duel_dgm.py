"""
DUEL : certification TLM-duale (grille cartesienne) vs DGM dual (maillage triangle,
P1 sup / RT0 inf, le solveur de HT) sur LE MEME probleme.

Probleme commun : carre unite, inclusion carree [0.25,0.75]^2 de sigma=10 (fond 1),
conductance G entre bord gauche (phi=0) et droit (phi=1), haut/bas isoles (Neumann).
Les deux methodes encadrent G ; on compare la DEMI-LARGEUR CERTIFIEE vs DDL et vs CPU,
et on verifie que les deux encadrements contiennent la meme reference.

Note d'equite : l'inclusion est alignee sur les axes -> exacte pour la grille cartesienne ;
pour le maillage triangle DGM, sigma est affecte par centroide (l'interface suit la
triangulation). C'est une difference de representation geometrique, signalee.
"""

import os, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# The reference solver is the author's dual discrete-geometric code, which is the
# subject of a separate publication and is not redistributed here. Point DGM_PATH at
# it, or set it through the environment, to reproduce Table V and Figure 5. Every
# other script in this repository is self-contained.
DGM_PATH = os.environ.get("DGM_PATH", os.path.expanduser("~/Documents/Dual_discret_method/dgm"))
sys.path.insert(0, DGM_PATH)
try:
    from dgm import assemble_primal, energy_lower_2d
    from dgm.meshing import rectangular_domain
except ImportError as exc:
    raise SystemExit(
        "The dual discrete-geometric reference solver was not found in\n"
        f"  {DGM_PATH}\n"
        "Set DGM_PATH to its location. This script is the only one that needs it;\n"
        "the certified TLM bracket itself has no such dependency."
    ) from exc

np.seterr(over='ignore', divide='ignore', invalid='ignore')
SIG_IN = 10.0

# Deux champs : (1) carre aligne = TERRAIN DU CARTESIEN (exact) -> biais geometrique ;
#               (2) sigma LISSE gaussien = duel EQUITABLE (aucune interface).
def sig_square(x, y):
    return SIG_IN if (0.25 <= x <= 0.75 and 0.25 <= y <= 0.75) else 1.0
def sig_smooth(x, y):
    return 1.0 + 9.0 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * 0.15 ** 2))

# ====================================================================
# Methode A : TLM-duale / cartesien (bornes = etats stationnaires des 2 reseaux TLM)
# ====================================================================
KREF = (1.0/6.0) * np.array([[ 4.,-1.,-2.,-1.],[-1., 4.,-1.,-2.],
                             [-2.,-1., 4.,-1.],[-1.,-2.,-1., 4.]])
def _assemble_cart(n, coeff):
    npn = n+1; nid = lambda i,j: j*npn+i
    rows, cols, vals = [], [], []
    for cj in range(n):
        yc=(cj+0.5)/n
        for ci in range(n):
            xc=(ci+0.5)/n; cc=coeff(xc,yc)
            loc=[nid(ci,cj),nid(ci+1,cj),nid(ci+1,cj+1),nid(ci,cj+1)]
            for a in range(4):
                for b in range(4):
                    rows.append(loc[a]); cols.append(loc[b]); vals.append(cc*KREF[a,b])
    return sp.csr_matrix((vals,(rows,cols)), shape=(npn*npn,npn*npn)), nid

def _solve_cart(K, npn, nid, orient):
    N=npn*npn; u=np.zeros(N); fx={}
    for k in range(npn):
        if orient=='LR': fx[nid(0,k)]=0.0; fx[nid(npn-1,k)]=1.0
        else:            fx[nid(k,0)]=0.0; fx[nid(k,npn-1)]=1.0
    fi=np.array(sorted(fx)); u[fi]=[fx[k] for k in fi]
    fr=np.setdiff1d(np.arange(N),fi)
    u[fr]=spla.spsolve(K[fr][:,fr].tocsc(), -K[fr][:,fi]@u[fi])
    return u

def cart_bounds(n, sfield):
    Ks,nid=_assemble_cart(n, sfield)
    phi=_solve_cart(Ks,n+1,nid,'LR'); Gup=float(phi@(Ks@phi))
    Kr,nid=_assemble_cart(n, lambda x,y: 1.0/sfield(x,y))
    psi=_solve_cart(Kr,n+1,nid,'BT'); Glo=1.0/float(psi@(Kr@psi))
    return Glo, Gup, (n+1)**2

# ====================================================================
# Methode B : DGM dual (triangle, P1 sup / RT0 inf) -- le solveur de HT
# ====================================================================
def dgm_bounds(h, sfield):
    mesh, _, dom = rectangular_domain((0,0,1,1), [], h=h, seed=1)
    cen = mesh.points[mesh.tris].mean(1)
    sig = np.array([sfield(x,y) for x,y in cen])
    P = mesh.points
    bc = {}
    for i in dom:
        if abs(P[i,0])     < 1e-7: bc[int(i)] = 0.0     # gauche
        elif abs(P[i,0]-1) < 1e-7: bc[int(i)] = 1.0     # droite
    K = assemble_primal(mesh, sig)
    v = np.zeros(mesh.np); fi = np.array(sorted(bc)); v[fi]=[bc[k] for k in fi]
    fr = np.setdiff1d(np.arange(mesh.np), fi)
    v[fr] = spla.spsolve(K[fr][:,fr].tocsc(), -K[fr][:,fi]@v[fi])
    Gup = float(v@(K@v))                      # borne sup. (P1)
    Glo = 2.0*energy_lower_2d(mesh, sig, bc)  # borne inf. (RT0, energy_lower=1/2 G)
    return Glo, Gup, mesh.np

# ====================================================================
# Reference et duel, pour chaque champ
# ====================================================================
def timed(fn, arg, sfield, ref):
    """ref = (lo, hi) intervalle CERTIFIE de reference, produit par l'AUTRE methode."""
    t0=time.perf_counter(); Glo,Gup,ndof=fn(arg, sfield); dt=time.perf_counter()-t0
    return dict(ndof=ndof, Glo=Glo, Gup=Gup, half=0.5*(Gup-Glo), cpu=dt,
                compat=(Glo <= ref[1] and Gup >= ref[0]))

CASES = {'carre (domicile cartesien)': sig_square, 'lisse (duel equitable)': sig_smooth}
NS = [16,32,64,128,256]; HS = [1/14, 1/20, 1/28, 1/40, 1/56]
results = {}
for name, sf in CASES.items():
    # REFERENCE NON CIRCULAIRE : chaque methode est confrontee a l'encadrement CERTIFIE
    # produit par l'AUTRE methode sur maillage fin. Prendre le milieu du bracket TLM fin
    # comme "vraie valeur" reviendrait a juger la DGM avec la reponse de la TLM, et le
    # milieu n'est de toute facon couvert par aucune garantie : seul l'intervalle l'est.
    tlo, tup, _ = cart_bounds(512, sf)          # reference certifiee TLM
    dlo, dup, _ = dgm_bounds(1/80, sf)          # reference certifiee DGM
    inter = (max(tlo, dlo), min(tup, dup))
    print(f"\n=== {name} ===")
    print(f"  reference certifiee TLM (n=512)  : [{tlo:.6f}, {tup:.6f}]")
    print(f"  reference certifiee DGM (h=1/80) : [{dlo:.6f}, {dup:.6f}]")
    if inter[0] <= inter[1]:
        print(f"  intersection                     : [{inter[0]:.6f}, {inter[1]:.6f}]  "
              f"-> les deux methodes sont MUTUELLEMENT COMPATIBLES")
    else:
        print(f"  intersection VIDE : [{inter[0]:.6f} > {inter[1]:.6f}]  "
              f"-> INCOMPATIBLES, les deux ne peuvent pas encadrer le meme probleme")
    A = [timed(cart_bounds, n, sf, (dlo, dup)) for n in NS]   # TLM juge par la DGM fine
    B = [timed(dgm_bounds, h, sf, (tlo, tup)) for h in HS]    # DGM jugee par la TLM fine
    print(f"{'methode':>10} {'ndof':>7} {'G_lower':>10} {'G_upper':>10} "
          f"{'demi-larg':>11} {'cpu[s]':>8} {'compat?':>8}")
    for r in A: print(f"{'TLM-cart':>10} {r['ndof']:>7} {r['Glo']:>10.5f} {r['Gup']:>10.5f} "
                      f"{r['half']:>11.3e} {r['cpu']:>8.3f} {str(r['compat']):>8}")
    for r in B: print(f"{'DGM-tri':>10} {r['ndof']:>7} {r['Glo']:>10.5f} {r['Gup']:>10.5f} "
                      f"{r['half']:>11.3e} {r['cpu']:>8.3f} {str(r['compat']):>8}")
    results[name] = (A, B)

# ---- figure 2x2 ----
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
for row, (name, (A, B)) in enumerate(results.items()):
    ax[row,0].loglog([r['ndof'] for r in A],[r['half'] for r in A],'o-',color='C0',label='TLM-dual (cartesien)')
    ax[row,0].loglog([r['ndof'] for r in B],[r['half'] for r in B],'s-',color='C3',label='DGM-dual (triangle, HT)')
    ax[row,0].set_xlabel('# DOF'); ax[row,0].set_ylabel('certified half-width')
    ax[row,0].set_title(f'{name} — vs DDL'); ax[row,0].grid(True,which='both',alpha=.3); ax[row,0].legend(fontsize=8)
    ax[row,1].loglog([r['cpu'] for r in A],[r['half'] for r in A],'o-',color='C0',label='TLM-dual')
    ax[row,1].loglog([r['cpu'] for r in B],[r['half'] for r in B],'s-',color='C3',label='DGM-dual (HT)')
    ax[row,1].set_xlabel('CPU [s]'); ax[row,1].set_ylabel('certified half-width')
    ax[row,1].set_title(f'{name} — vs CPU'); ax[row,1].grid(True,which='both',alpha=.3); ax[row,1].legend(fontsize=8)
fig.suptitle('Duel: dual-TLM (Cartesian) vs dual-DGM (triangular) certification', fontsize=12)
fig.tight_layout(); out='duel_result.png'
fig.savefig(out, dpi=130); print(f"\nFigure -> {out}")
