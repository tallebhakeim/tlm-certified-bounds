"""
PoC 2D — certification a bornes garanties via couple dual (analogue des 2 reseaux TLM).

Lecon du 1D : la certification complementaire ne MORD qu'en >=2D (en 1D, pas de
liberte rotationnelle -> conforme == equilibre -> pas de gap). On teste donc en 2D.

Probleme : conduction stationnaire dans le carre unite, milieu HETEROGENE
(inclusion CARREE [0.25,0.75]^2 sigma=SIG_IN, fond sigma=1 ; carre aligne sur la grille
pour n multiple de 4 => geometrie IDENTIQUE a toute resolution => bornes EMBOITEES
autour d'un G_vrai fixe). On encadre la CONDUCTANCE effective G entre gauche et droite.

Couple dual (Prager / Thomson-Dirichlet) = analogue des 2 reseaux TLM duaux :

  Reseau PHI (potentiel, "capacitif", noeud=tension) :
      phi : Dirichlet 0|1 gauche|droite, insule haut/bas
      principe de Dirichlet  ->  G_upper = \int sigma |grad phi|^2   (BORNE SUP. de G)

  Reseau PSI (flux/fonction de courant, "inductif", noeud=flux) :
      psi : Dirichlet 0|1 bas|haut avec resistivite rho=1/sigma, insule gauche/droite
      principe de Thomson    ->  R_upper = \int rho |grad psi|^2  (BORNE SUP. de R)
                                  G_lower = 1 / R_upper            (BORNE INF. de G)

Resultat attendu :  G_lower <= G_vrai <= G_upper , gap -> 0 sous raffinement.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SIG_IN = 10.0          # conductivite de l'inclusion

# Matrice de raideur Q1 de reference (carre, \int grad Ni . grad Nj), independante de h en 2D
KREF = (1.0 / 6.0) * np.array([[ 4., -1., -2., -1.],
                               [-1.,  4., -1., -2.],
                               [-2., -1.,  4., -1.],
                               [-1., -2., -1.,  4.]])

def sigma_cell(xc, yc):
    """sigma au centre de cellule : inclusion carree alignee [0.25,0.75]^2."""
    return SIG_IN if (0.25 <= xc <= 0.75 and 0.25 <= yc <= 0.75) else 1.0

def assemble(n, coeff):
    """Assemble K = sum_cell coeff_c * KREF sur grille n x n cellules (Q1).
    coeff(xc,yc) donne le coefficient par cellule. Retourne (K, node_id)."""
    npn = n + 1
    nid = lambda i, j: j * npn + i           # i=colonne(x), j=ligne(y)
    rows, cols, vals = [], [], []
    for cj in range(n):
        yc = (cj + 0.5) / n
        for ci in range(n):
            xc = (ci + 0.5) / n
            c = coeff(xc, yc)
            loc = [nid(ci, cj), nid(ci + 1, cj), nid(ci + 1, cj + 1), nid(ci, cj + 1)]
            for a in range(4):
                for b in range(4):
                    rows.append(loc[a]); cols.append(loc[b]); vals.append(c * KREF[a, b])
    K = sp.csr_matrix((vals, (rows, cols)), shape=(npn * npn, npn * npn))
    return K, nid

def solve_dirichlet(K, npn, nid, fixed):
    """Resout avec valeurs imposees 'fixed' = {node: val}. Retourne u complet."""
    Ntot = npn * npn
    u = np.zeros(Ntot)
    fixed_idx = np.array(sorted(fixed))
    u[fixed_idx] = [fixed[k] for k in fixed_idx]
    free = np.setdiff1d(np.arange(Ntot), fixed_idx)
    Kff = K[free][:, free].tocsc()
    rhs = -K[free][:, fixed_idx] @ u[fixed_idx]
    u[free] = spla.spsolve(Kff, rhs)
    res = np.linalg.norm(Kff @ u[free] - rhs) / (np.linalg.norm(rhs) + 1e-300)
    assert res < 1e-8, f"solve non converge (residu={res:.2e})"
    return u

def conductance_bounds(n):
    npn = n + 1
    # --- Reseau PHI : potentiel, conductivite sigma, BC gauche/droite ---
    Ksig, nid = assemble(n, sigma_cell)
    fphi = {}
    for j in range(npn):
        fphi[nid(0, j)] = 0.0          # bord gauche phi=0
        fphi[nid(n, j)] = 1.0          # bord droit  phi=1
    phi = solve_dirichlet(Ksig, npn, nid, fphi)
    G_upper = phi @ (Ksig @ phi)       # \int sigma |grad phi|^2

    # --- Reseau PSI : flux, resistivite rho=1/sigma, BC bas/haut ---
    Krho, nid = assemble(n, lambda xc, yc: 1.0 / sigma_cell(xc, yc))
    fpsi = {}
    for i in range(npn):
        fpsi[nid(i, 0)] = 0.0          # bord bas  psi=0
        fpsi[nid(i, n)] = 1.0          # bord haut psi=1
    psi = solve_dirichlet(Krho, npn, nid, fpsi)
    R_upper = psi @ (Krho @ psi)       # \int rho |grad psi|^2
    G_lower = 1.0 / R_upper
    return G_lower, G_upper

# ----------------------------------------------------------------------
np.seterr(over='ignore', divide='ignore', invalid='ignore')   # warnings overflow benins
ns = [8, 16, 32, 64, 128, 256]
print("Calcul de la reference fine (n=512)...")
Glo_ref, Gup_ref = conductance_bounds(512)
G_ref = 0.5 * (Glo_ref + Gup_ref)      # milieu de l'intervalle fin = estimation de G_vrai
print(f"G_ref ~ {G_ref:.8f}  (intervalle certifie fin [{Glo_ref:.8f}, {Gup_ref:.8f}], "
      f"largeur {Gup_ref - Glo_ref:.2e})\n")

print(f"{'n':>5} {'G_lower':>13} {'G_ref':>13} {'G_upper':>13} "
      f"{'gap':>12} {'contient G?':>12}")
gaps, hs = [], []
for n in ns:
    Glo, Gup = conductance_bounds(n)
    assert Glo <= Gup + 1e-12, "bracket intra-grille viole !"   # doit TOUJOURS tenir
    ok = (Glo <= G_ref + 1e-9) and (G_ref <= Gup + 1e-9)        # emboitement
    print(f"{n:>5} {Glo:>13.8f} {G_ref:>13.8f} {Gup:>13.8f} "
          f"{Gup - Glo:>12.3e} {str(ok):>12}")
    gaps.append(Gup - Glo); hs.append(1.0 / n)

gaps = np.array(gaps); hs = np.array(hs)
p = np.polyfit(np.log(hs), np.log(gaps), 1)[0]
print(f"\nOrdre de convergence du gap  ~  O(h^{p:.2f})")

# ----------------------------------------------------------------------
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
bnds = [conductance_bounds(n) for n in ns]
Glos = [b[0] for b in bnds]; Gups = [b[1] for b in bnds]
ax[0].axhline(G_ref, color='k', ls='--', lw=1, label='G_true (fine ref.)')
ax[0].plot(ns, Glos, 'o-', color='C0', label='G_lower (reseau PSI / flux)')
ax[0].plot(ns, Gups, 's-', color='C3', label='G_upper (reseau PHI / potentiel)')
ax[0].fill_between(ns, Glos, Gups, color='gray', alpha=0.2, label='certified interval')
ax[0].set_xscale('log', base=2); ax[0].set_xlabel('n (cells per side)')
ax[0].set_ylabel('conductance G'); ax[0].set_title('Guaranteed 2D bounds (inclusion sigma=10)')
ax[0].legend(fontsize=8)
ax[1].loglog(hs, gaps, 'o-', color='C2', label=f'gap ~ O(h^{p:.2f})')
ax[1].loglog(hs, gaps[0] * (hs / hs[0]), 'k--', lw=1, label='pente 1 (ref)')
ax[1].set_xlabel('h = 1/n'); ax[1].set_ylabel('certified interval width')
ax[1].set_title('Gap convergence (certified error)')
ax[1].legend(fontsize=8)
fig.suptitle('2D PoC: guaranteed-bound certification via the dual pair (TLM analogue)',
             fontsize=11)
fig.tight_layout()
out = 'poc_2d_result.png'
fig.savefig(out, dpi=130)
print(f"\nFigure -> {out}")
