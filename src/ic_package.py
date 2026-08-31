"""
Certified 3-D capacitance extraction on a real IC package (dual-TLM), corrected figure.

HISTORY / WARNING. The first version of this script voxelised the package on a UNIFORM
Cartesian grid whose planes were unrelated to the geometry. The die pad (4.000 mm wide,
0.020 mm thick, 1.075 mm above the ground plane) was then snapped to whatever cells
happened to contain it: its width was seen as 3.655 mm and the gap as 1.1225 mm on the
grid used for the figure, i.e. -8.6 % on the pad width and +4.4 % on the gap. The
published bracket [829.85, 919.97] fF was therefore a rigorous enclosure of the WRONG
capacitor, and it does not contain the correct value. Prager-Synge bounds the
DISCRETISATION error of a given geometry; it does not cover GEOMETRIC error.

This version delegates the physics to ic_package_refine.solve(), which builds
geometry-conforming grids (cell faces aligned with every material interface, so the
voxel model reproduces the CAD exactly at every refinement level). Reference result,
finest grid: C in [968.51, 1037.37] fF.

Run:  python3 ic_package.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ic_package_refine import solve, B, pad_bb, sub_bb, gap, A_pad, C_pp

DX, DZ = 0.25, 0.07                      # finest grid of the refinement study
CACHE = '/tmp/ic_pkg_fields.npz'         # the finest solve costs ~250 s; cache it
try:
    r = {k: v for k, v in np.load(CACHE, allow_pickle=True).items()}
    r = {k: (v.item() if v.ndim == 0 else v) for k, v in r.items()}
    print(f"fields loaded from {CACHE}")
except (FileNotFoundError, OSError):
    r = solve(DX, DZ, verbose=True, return_fields=True)
    np.savez(CACHE, **r)
    print(f"fields cached to {CACHE}")
C_lo, C_up = r['C_lo'], r['C_up']
mid = 0.5*(C_lo+C_up); hw = (C_up-C_lo)/(C_up+C_lo)*100
print(f"CERTIFIED bracket: C in [{C_lo*1e15:.2f}, {C_up*1e15:.2f}] fF ; "
      f"midpoint {mid*1e15:.2f} fF ; half-width {hw:.2f}%")
print(f"pad {r['pad_w']:.6f} x {r['pad_t']:.6f} mm, gap {r['gap_eff']:.6f} mm "
      f"(CAD 4.000000 / 0.020000 / {gap:.6f}) -> geometry exact")
print(f"parallel-plate estimate without fringing: {C_pp*1e15:.1f} fF "
      f"(x{mid/C_pp:.2f} below the certified midpoint)")

xn, yn, zn = r['xn'], r['yn'], r['zn']
x0, x1, z0, z1 = xn[0], xn[-1], zn[0], zn[-1]

fig = plt.figure(figsize=(16.5, 5.0))

# (a) top view of the real package (QFP)
axA = fig.add_subplot(1, 3, 1)
lead_ids = [i for i in range(len(B)) if i not in (0, 2, 4)]
sb = sub_bb
axA.add_patch(Rectangle((sb[0, 0], sb[0, 1]), sb[1, 0]-sb[0, 0], sb[1, 1]-sb[0, 1],
                        fc='#d9c200', ec='k', lw=.4, zorder=0))
for i in lead_ids:
    bb = B[i].bounds
    axA.add_patch(Rectangle((bb[0, 0], bb[0, 1]), bb[1, 0]-bb[0, 0], bb[1, 1]-bb[0, 1],
                            fc='#b8b8b8', ec='0.4', lw=.2, zorder=1))
axA.add_patch(Rectangle((pad_bb[0, 0], pad_bb[0, 1]), pad_bb[1, 0]-pad_bb[0, 0],
                        pad_bb[1, 1]-pad_bb[0, 1], fc='#c0392b', ec='k', lw=.6, zorder=2))
axA.text(5, 5, 'die pad\n(signal)', color='w', ha='center', va='center',
         fontsize=8, fontweight='bold', zorder=3)
axA.text(5, 11.2, 'leads', color='0.3', ha='center', fontsize=8)
axA.text(10.4, -1.3, 'substrate\n(ground)', color='0.45', ha='center', fontsize=7.5)
for xb in (pad_bb[0, 0], pad_bb[1, 0]):
    axA.axvline(xb, color='k', lw=.4, ls=':', alpha=.6, zorder=4)
axA.set_xlim(-2.2, 12.2); axA.set_ylim(-2.2, 12.2); axA.set_aspect('equal')
axA.set_xticks([]); axA.set_yticks([])
axA.set_title('(a) IC package top view (QFP)\nsubstrate / leads / central die pad', fontsize=10)

# (b) xz cross-section through the pad centre
jmid = int(np.argmin(np.abs(0.5*(yn[:-1]+yn[1:])-pad_bb[:, 1].mean())))
U = r['u'][:, jmid, :]                                   # (nz+1, nx+1) = (z, x)
axB = fig.add_subplot(1, 3, 2)
im = axB.pcolormesh(xn, zn, U, cmap='turbo', vmin=0, vmax=1, shading='auto')
axB.contour(xn, zn, U, levels=np.linspace(0.05, 0.95, 12), colors='k',
            linewidths=.35, alpha=.45)
axB.add_patch(Rectangle((pad_bb[0, 0], pad_bb[0, 2]), pad_bb[1, 0]-pad_bb[0, 0],
                        max(pad_bb[1, 2]-pad_bb[0, 2], 0.03), ec='r', fc='r', lw=1.5))
axB.axhspan(sub_bb[0, 2], sub_bb[1, 2], color='0.25', alpha=.85)
axB.text(9.2, sub_bb[:, 2].mean(), 'substrate (V=0)', fontsize=8, color='w',
         va='center', ha='center')
axB.text(5, pad_bb[1, 2]+0.18, 'pad (V=1)', fontsize=8, color='r', ha='center',
         fontweight='bold')
axB.set_xlim(x0, x1); axB.set_ylim(-1.1, 1.05)
axB.set_xlabel('x [mm]'); axB.set_ylabel('z [mm]')
axB.set_title('(b) xz cross-section (pad centre)\npotential and fringing field '
              '(mould eps_r = 4)', fontsize=10)
fig.colorbar(im, ax=axB, shrink=.75, label='potential u')

# (c) refinement history: the five nested certified brackets
axC = fig.add_subplot(1, 3, 3)
H = np.load('/tmp/ic_pkg_refine.npz')
cells = H['ncell']; lo = H['C_lo']*1e15; up = H['C_up']*1e15
axC.fill_between(cells, lo, up, color='C3', alpha=.25, step=None,
                 label='certified bracket')
axC.plot(cells, up, 'o-', color='C3', lw=1.4, ms=4, label='upper (conforming Q1)')
axC.plot(cells, lo, 's-', color='C0', lw=1.4, ms=4, label='lower (equilibrated RT0)')
axC.plot(cells, 0.5*(lo+up), 'k--', lw=1.0, label='midpoint')
axC.axhline(C_pp*1e15, color='0.5', lw=1.2, ls=':',
            label=f'parallel-plate, no fringing {C_pp*1e15:.0f} fF')
axC.set_xscale('log')
axC.set_xlabel('number of cells'); axC.set_ylabel('pad-substrate capacitance [fF]')
axC.set_title(f'(c) Nested certified brackets (dual-TLM)\n'
              f'finest: C in [{C_lo*1e15:.0f}, {C_up*1e15:.0f}] fF  '
              f'(half-width {hw:.1f}%)', fontsize=10)
axC.grid(True, which='both', alpha=.25)
axC.legend(fontsize=7.5, loc='lower right')
axC.annotate(f'half-width\n{(up[0]-lo[0])/(up[0]+lo[0])*100:.1f}%',
             xy=(cells[0], 0.5*(lo[0]+up[0])), xytext=(cells[0]*1.15, 1090),
             fontsize=7.5, color='0.35', ha='left')
axC.annotate(f'{hw:.1f}%', xy=(cells[-1], 0.5*(lo[-1]+up[-1])),
             xytext=(cells[-1]*0.55, 900), fontsize=7.5, color='0.35')

fig.suptitle('Certified 3-D capacitance extraction on a real IC package: '
             'dual-TLM two-sided bracket on the pad-substrate capacitance', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('ic_package_result.png', dpi=120)
print('Figure -> ic_package_result.png')
