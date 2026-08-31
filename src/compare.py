"""
Banc de comparaison de la certification TLM-duale (bornes complementaires garanties).

Les bornes sont calculees par les solves conforme/equilibre = etats stationnaires des
deux reseaux TLM (equivalence prouvee dans tlm_transient.py a 1e-9). On utilise ici les
solves EF, plus rapides, pour les balayages.

Trois etudes :
  (1) BALAYAGE DE CONTRASTE  : largeur certifiee vs ratio sigma_in/sigma_out (1..1000).
  (2) GEOMETRIE              : taux de convergence O(h^p) du gap (carre / damier / coin L).
  (3) INDICE D'EFFICACITE    : eta = (demi-largeur certifiee) / |erreur vraie|  -> qualite
                               de l'estimateur (eta ~ 1 = borne fine ; eta >> 1 = laxe).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

np.seterr(over='ignore', divide='ignore', invalid='ignore')
KREF = (1.0 / 6.0) * np.array([[ 4., -1., -2., -1.], [-1.,  4., -1., -2.],
                               [-2., -1.,  4., -1.], [-1., -2., -1.,  4.]])

# ----------------------------------------------------------------------
# Champs de conductivite (xc,yc dans [0,1]) parametres par le contraste c=sigma_in
# ----------------------------------------------------------------------
def field_carre(c):
    return lambda x, y: c if (0.25 <= x <= 0.75 and 0.25 <= y <= 0.75) else 1.0

def field_damier(c, k=4):
    def f(x, y):
        return c if ((int(x * k) + int(y * k)) % 2 == 0) else 1.0
    return f

def field_L(c):
    # conducteur en L (fort) : tout sauf l'encoche haut-droite -> coin rentrant en (0.5,0.5)
    def f(x, y):
        return 1.0 if (x >= 0.5 and y >= 0.5) else c
    return f

# ----------------------------------------------------------------------
def assemble(n, coeff):
    npn = n + 1
    nid = lambda i, j: j * npn + i
    rows, cols, vals = [], [], []
    for cj in range(n):
        yc = (cj + 0.5) / n
        for ci in range(n):
            xc = (ci + 0.5) / n
            cc = coeff(xc, yc)
            loc = [nid(ci, cj), nid(ci + 1, cj), nid(ci + 1, cj + 1), nid(ci, cj + 1)]
            for a in range(4):
                for b in range(4):
                    rows.append(loc[a]); cols.append(loc[b]); vals.append(cc * KREF[a, b])
    return sp.csr_matrix((vals, (rows, cols)), shape=(npn * npn, npn * npn)), nid

def _solve(K, npn, nid, orient):
    Ntot = npn * npn; u = np.zeros(Ntot); fixed = {}
    for k in range(npn):
        if orient == 'LR':
            fixed[nid(0, k)] = 0.0; fixed[nid(npn - 1, k)] = 1.0
        else:
            fixed[nid(k, 0)] = 0.0; fixed[nid(k, npn - 1)] = 1.0
    fidx = np.array(sorted(fixed)); u[fidx] = [fixed[k] for k in fidx]
    free = np.setdiff1d(np.arange(Ntot), fidx)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), -K[free][:, fidx] @ u[fidx])
    return u

def bounds(n, coeff):
    """Retourne (G_lower, G_upper) = bornes complementaires garanties."""
    Ks, nid = assemble(n, coeff)
    phi = _solve(Ks, n + 1, nid, 'LR'); G_up = float(phi @ (Ks @ phi))
    Kr, nid = assemble(n, lambda x, y: 1.0 / coeff(x, y))
    psi = _solve(Kr, n + 1, nid, 'BT'); G_lo = 1.0 / float(psi @ (Kr @ psi))
    return G_lo, G_up

# ----------------------------------------------------------------------
# (1) BALAYAGE DE CONTRASTE
# ----------------------------------------------------------------------
print("=== (1) Balayage de contraste (carre, n=128) ===")
contrasts = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000.0])
relwidth = []
for c in contrasts:
    Glo, Gup = bounds(128, field_carre(c))
    relwidth.append((Gup - Glo) / (0.5 * (Gup + Glo)))
    print(f"  sigma_in/out={c:7.0f}  G in [{Glo:.5f}, {Gup:.5f}]  largeur rel={relwidth[-1]:.3e}")
relwidth = np.array(relwidth)

# ----------------------------------------------------------------------
# (2) GEOMETRIE : convergence du gap
# ----------------------------------------------------------------------
print("\n=== (2) Convergence du gap par geometrie (contraste=100) ===")
geoms = {'carre': field_carre(100), 'damier': field_damier(100), 'coin L': field_L(100)}
ns = [16, 32, 64, 128, 256]
conv = {}
for name, f in geoms.items():
    gp = []
    for n in ns:
        Glo, Gup = bounds(n, f); gp.append(Gup - Glo)
    p = np.polyfit(np.log(1.0 / np.array(ns)), np.log(gp), 1)[0]
    conv[name] = (np.array(gp), p)
    print(f"  {name:8s} : gap ~ O(h^{p:.2f})   (gap@n=256 = {gp[-1]:.2e})")

# ----------------------------------------------------------------------
# (3) INDICE D'EFFICACITE  eta = gap / |erreur vraie d'une borne|  (doit ~ 1-2 = fin)
#     + bonus : le MILIEU des deux bornes est superconvergent.
# ----------------------------------------------------------------------
print("\n=== (3) Indice d'efficacite eta (carre, contraste=100) ===")
fcar = field_carre(100)
Glo_ref, Gup_ref = bounds(512, fcar); G_true = 0.5 * (Glo_ref + Gup_ref)
print(f"  G_true (ref n=512) = {G_true:.6f}")
etas, err_mids, err_bnds = [], [], []
for n in ns:
    Glo, Gup = bounds(n, fcar)
    W = Gup - Glo
    err_bnd = max(abs(Gup - G_true), abs(Glo - G_true))   # erreur de la pire borne
    err_mid = abs(0.5 * (Gup + Glo) - G_true)             # erreur du milieu (superconv.)
    eta = W / err_bnd if err_bnd > 0 else np.nan
    etas.append(eta); err_bnds.append(err_bnd); err_mids.append(err_mid)
    print(f"  n={n:3d}  gap={W:.3e}  err_borne={err_bnd:.3e}  eta={eta:.2f}"
          f"   |  err_milieu={err_mid:.2e}  (x{err_bnd/err_mid:.0f} plus precis)")

# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
ax[0].loglog(contrasts, relwidth, 'o-', color='C4')
ax[0].set_xlabel('contrast sigma_in / sigma_out'); ax[0].set_ylabel('relative certified width')
ax[0].set_title('(1) Certification cost vs contrast'); ax[0].grid(True, which='both', alpha=0.3)

hs = 1.0 / np.array(ns)
for name, (gp, p) in conv.items():
    ax[1].loglog(hs, gp, 'o-', label=f'{name} : O(h^{p:.2f})')
ax[1].set_xlabel('h = 1/n'); ax[1].set_ylabel('certified width (gap)')
ax[1].set_title('(2) Convergence par geometrie'); ax[1].legend(fontsize=8)
ax[1].grid(True, which='both', alpha=0.3)

ax[2].semilogx(ns, etas, 'o-', color='C2', label='eta = gap / bound error')
ax[2].axhline(1.0, color='k', ls=':', lw=1)
ax[2].axhline(2.0, color='k', ls='--', lw=1, label='eta=2 (sharp bound)')
ax[2].set_xlabel('n'); ax[2].set_ylabel('indice d\'efficacite eta')
ax[2].set_title('(3) Bound sharpness (eta ~ 2 = sharp)')
ax[2].set_ylim(0, 4); ax[2].legend(fontsize=8); ax[2].grid(True, alpha=0.3)

fig.suptitle('Benchmark - guaranteed-bound dual-TLM certification', fontsize=11)
fig.tight_layout()
out = 'compare_result.png'
fig.savefig(out, dpi=130); print(f"\nFigure -> {out}")
