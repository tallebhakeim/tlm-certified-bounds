"""Figure 6 : l'erreur geometrique est invisible pour le certificat.

Trois panneaux, tous construits a partir des sorties des scripts de calcul, aucune
valeur saisie a la main :
  (a) largeur de pad et entrefer VUS par des grilles uniformes vs conformes (calcul
      geometrique direct, pas de solve) ;
  (b) les brackets certifies correspondants : 5 intervalles DISJOINTS sur grilles
      uniformes (/tmp/ic_pkg_uniform.npz), 5 intervalles EMBOITES sur grilles
      conformes (/tmp/ic_pkg_refine.npz) ;
  (c) meme contraste sur l'inclusion cubique 3D (/tmp/tlm3d_incl.npz).

Prerequis : python3 ic_package_refine.py  et  python3 tlm3d.py
Lancer    : python3 geometric_error.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ic_package_refine import (XBP, ZBP, conforming_nodes, uniform_nodes,
                               pad_bb, sub_bb, gap)

PAD_W = float(pad_bb[1, 0]-pad_bb[0, 0])

U = np.load('/tmp/ic_pkg_uniform.npz')
C = np.load('/tmp/ic_pkg_refine.npz')
T = np.load('/tmp/tlm3d_incl.npz')


def seen(DX, DZ, conforming):
    """Largeur de pad et entrefer effectivement representes par la grille."""
    nodes = conforming_nodes if conforming else uniform_nodes
    xn = nodes(XBP, DX); zn = nodes(ZBP, DZ)
    xc = 0.5*(xn[:-1]+xn[1:]); zc = 0.5*(zn[:-1]+zn[1:])
    mx = (xc >= pad_bb[0, 0]) & (xc <= pad_bb[1, 0])
    i0, i1 = np.nonzero(mx)[0][[0, -1]]
    w = float(xn[i1+1]-xn[i0])
    if conforming:
        mz = (zc >= pad_bb[0, 2]) & (zc <= pad_bb[1, 2])
        k0 = np.nonzero(mz)[0][0]
    else:
        k0 = int(np.argmin(np.abs(zc-pad_bb[:, 2].mean())))
    ks = np.nonzero((zc >= sub_bb[0, 2]) & (zc <= sub_bb[1, 2]))[0]
    g = float(zn[k0]-zn[ks.max()+1])
    return w, g


ug = list(zip(U['DX'], U['DZ']))
cg = list(zip(C['DX'], C['DZ']))
uw = np.array([seen(a, b, False) for a, b in ug])
cw = np.array([seen(a, b, True) for a, b in cg])

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

# ---- (a) geometrie vue ----
a = ax[0]
a.plot(U['hx'], 100*(uw[:, 0]-PAD_W)/PAD_W, 'o-', color='C3', label='pad width, uniform grid')
a.plot(U['hx'], 100*(uw[:, 1]-gap)/gap, 's--', color='C1', label='pad-substrate gap, uniform grid')
a.plot(C['hx'], 100*(cw[:, 0]-PAD_W)/PAD_W+0.09, 'o-', color='C0',
       label='pad width, conforming grid (offset +0.09 for legibility)')
a.plot(C['hx'], 100*(cw[:, 1]-gap)/gap, 's--', color='C2', label='gap, conforming grid')
a.axhline(0, color='k', lw=.8)
a.set_xlabel('grid step h [mm]'); a.set_ylabel('error on the represented geometry [%]')
a.set_title('(a) What the grid actually sees\nCAD pad 4.000 mm, gap 1.075 mm', fontsize=10)
a.grid(alpha=.3); a.legend(fontsize=7.5)
a.invert_xaxis()

# ---- (b) brackets 3D capacite ----
b = ax[1]
ulo, uup = U['C_lo']*1e15, U['C_up']*1e15
clo, cup = C['C_lo']*1e15, C['C_up']*1e15
for i, (lo, up) in enumerate(zip(ulo, uup)):
    b.plot([i-0.16, i-0.16], [lo, up], color='C3', lw=7, solid_capstyle='butt',
           label='uniform grid (disjoint)' if i == 0 else None, alpha=.85)
for i, (lo, up) in enumerate(zip(clo, cup)):
    b.plot([i+0.16, i+0.16], [lo, up], color='C0', lw=7, solid_capstyle='butt',
           label='conforming grid (nested)' if i == 0 else None, alpha=.85)
b.axhspan(clo.max(), cup.min(), color='C0', alpha=.12)
b.text(-0.62, 0.5*(clo.max()+cup.min()), 'certified\nenclosure', fontsize=7.5,
       color='C0', ha='center', va='center')
b.set_xticks(range(5)); b.set_xticklabels([f'g{i+1}' for i in range(5)])
b.set_xlabel('grid, coarse to fine'); b.set_ylabel('pad-substrate capacitance [fF]')
b.set_title('(b) Same solver, same physics\n5 rigorous brackets each', fontsize=10)
b.grid(alpha=.3, axis='y'); b.legend(fontsize=8, loc='lower left')
b.set_ylim(min(ulo.min(), clo.min())-30, max(uup.max(), cup.max())+95)
b.annotate('empty intersection:\nat most one can be right',
           xy=(2-0.16, uup[2]+4), xytext=(2.55, 1175), fontsize=7.5, color='C3',
           ha='left', arrowprops=dict(arrowstyle='->', color='C3', lw=.9))
b.set_xlim(-0.95, 4.6)

# ---- (c) inclusion cubique 3D ----
c = ax[2]
for i, (n, lo, up, e) in enumerate(zip(T['n_bad'], T['lo_bad'], T['up_bad'], T['edge_bad'])):
    c.plot([i-0.16, i-0.16], [lo, up], color='C3', lw=7, solid_capstyle='butt',
           label='non-conforming n (disjoint)' if i == 0 else None, alpha=.85)
    c.text(i-0.16, lo-0.030, f'edge\n{e:.3f}', fontsize=7, color='C3', ha='center', va='top')
for i, (n, lo, up, e) in enumerate(zip(T['n_ok'], T['lo_ok'], T['up_ok'], T['edge_ok'])):
    c.plot([i+0.16, i+0.16], [lo, up], color='C0', lw=7, solid_capstyle='butt',
           label='conforming n (nested)' if i == 0 else None, alpha=.85)
    c.text(i+0.16, up+0.008, f'edge\n{e:.3f}', fontsize=7, color='C0', ha='center', va='bottom')
c.set_ylim(min(T['lo_bad'].min(), T['lo_ok'].min())-0.075,
           max(T['up_bad'].max(), T['up_ok'].max())+0.045)
c.set_xticks(range(3))
c.set_xticklabels([f"n={a}/{b}" for a, b in zip(T['n_bad'], T['n_ok'])])
c.set_xlabel('grid'); c.set_ylabel('conductance G')
c.set_title('(c) 3-D cubic inclusion, exact edge 0.400\nsame failure, same cause', fontsize=10)
c.grid(alpha=.3, axis='y'); c.legend(fontsize=8, loc='upper right')

fig.suptitle('Prager-Synge bounds the discretisation error of the geometry it is given: '
             'it does not see the geometric error', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('geometric_error_result.png', dpi=120)
print('Figure -> geometric_error_result.png')
print(f"uniform : intersection [{ulo.max():.2f}, {uup.min():.2f}] fF -> "
      f"{'NON VIDE' if ulo.max() < uup.min() else 'VIDE'}")
print(f"conforme: intersection [{clo.max():.2f}, {cup.min():.2f}] fF -> "
      f"{'NON VIDE' if clo.max() < cup.min() else 'VIDE'}")
