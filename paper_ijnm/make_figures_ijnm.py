"""
Figures of the IJNM manuscript, drawn from results_ijnm.json / results_ijnm.npz
(written by compute_all.py). Every figure distinguishes the certified interval from the
midpoint. Output: figures/Fig1..Fig7 as PNG (300 dpi) and TIFF (Wiley asks for TIFF/EPS).

Run:  python3 make_figures_ijnm.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, 'figures'); os.makedirs(FIGD, exist_ok=True)
R = json.load(open(os.path.join(HERE, 'results_ijnm.json')))
Z = np.load(os.path.join(HERE, 'results_ijnm.npz'))
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
                     'legend.fontsize': 7.2, 'figure.dpi': 110, 'savefig.dpi': 300})
CUP, CLO, CMID, CBAND = '#c0392b', '#1f77b4', 'k', '#c0392b'


def save(fig, name):
    fig.savefig(os.path.join(FIGD, name+'.png'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGD, name+'.tif'), bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig); print('->', name)


# ------------------------------------------------------------------ Fig 1: principle
fig, ax = plt.subplots(figsize=(7.6, 4.3)); ax.set_xlim(0, 10); ax.set_ylim(-0.4, 6); ax.axis('off')


def box(x, y, w, h, text, fc, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08', fc=fc, ec='0.3', lw=0.9))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal')


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=12, color='0.25', lw=1.0))


box(1.6, 5.05, 6.8, 0.8, 'Elliptic problem on a box assembly: div(σ grad u) = 0\nquantity of interest G = ∫σ|grad u|²', '#eeeeee', 7.4)
box(0.2, 3.3, 4.5, 1.1, 'Potential network\nnode variable: potential, pinned on electrodes\nsteady state: conforming potential v', '#fde3e3', 7.0)
box(5.3, 3.3, 4.5, 1.1, 'Flux network\nnode variable: stream function (2-D), face flux (3-D)\nsteady state: equilibrated flux j', '#e3ecfd', 7.0)
arrow(4.0, 5.05, 2.4, 4.42); arrow(6.0, 5.05, 7.6, 4.42)
box(0.2, 1.75, 4.5, 0.9, 'Dirichlet principle\nG(v) = ∫σ|grad v|²  ≥  G', '#fde3e3', 7.4)
box(5.3, 1.75, 4.5, 0.9, 'Thomson principle\nG(j) = 1 / ∫σ⁻¹|j|²  ≤  G', '#e3ecfd', 7.4)
arrow(2.4, 3.3, 2.4, 2.67); arrow(7.6, 3.3, 7.6, 2.67)
box(0.8, 0.25, 8.4, 1.0, 'Guaranteed enclosure  G(j) ≤ G ≤ G(v),  width = ∫σ⁻¹|j − σ grad v|²  (hypercircle identity)\nvalid for every admissible pair, hence at every iteration', '#e8f5e9', 7.2, True)
arrow(2.4, 1.75, 3.4, 1.27); arrow(7.6, 1.75, 6.6, 1.27)
ax.text(0.15, -0.25, 'What the enclosure does not cover: the difference between the box assembly the grid represents and the intended one (geometric error).',
        fontsize=6.8, color='0.3', style='italic')
save(fig, 'Fig1_principle')

# ------------------------------------------------------------------ Fig 2: 2-D validation + effectivity
v = R['val2d']; rows = v['rows']
ns = np.array([r['n'] for r in rows]); lo = np.array([r['lo'] for r in rows]); up = np.array([r['up'] for r in rows])
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
ax[0].fill_between(ns, lo, up, color=CBAND, alpha=.18, label='certified interval')
ax[0].plot(ns, up, 'o-', color=CUP, ms=4, label='upper bound (potential network)')
ax[0].plot(ns, lo, 's-', color=CLO, ms=4, label='lower bound (flux network)')
ax[0].plot(ns, 0.5*(lo+up), 'k--', lw=1, label='midpoint (not certified)')
ax[0].axhspan(v['ref_lo'], v['ref_up'], color='0.4', alpha=.5, label='reference interval, n = 512')
ax[0].set_xscale('log', base=2); ax[0].set_xlabel('cells per side n'); ax[0].set_ylabel('conductance G')
ax[0].set_ylim(1.52, 1.57); ax[0].legend(loc='upper right'); ax[0].set_title('(a) Nested enclosures, inclusion σ = 10')
hs = 1/ns; gap = up-lo
ax[1].loglog(hs, gap, 'o-', color='C2', label=f'certified width, O(h^{v["rate"]:.2f})')
ax[1].loglog(hs, gap[0]*(hs/hs[0]), 'k:', lw=1, label='slope 1')
ax[1].loglog(hs, gap[0]*(hs/hs[0])**2, 'k--', lw=1, label='slope 2')
ax[1].set_xlabel('grid step h'); ax[1].set_ylabel('width of the certified interval'); ax[1].legend(); ax[1].set_title('(b) Convergence of the width')
e = R['effectivity']['rows']; n2 = [r['n'] for r in e]
ax[2].loglog(n2, [r['halfwidth'] for r in e], 'o-', color='C2', label='certified half-width')
ax[2].loglog(n2, [r['err_bound'] for r in e], 's--', color=CUP, label='true error of the worse bound')
ax[2].loglog(n2, [r['err_mid'] for r in e], '^:', color='k', label='true error of the midpoint')
ax[2].set_xlabel('cells per side n'); ax[2].set_ylabel('error'); ax[2].legend(); ax[2].set_title('(c) Sharpness, inclusion σ = 100')
fig.tight_layout(); save(fig, 'Fig2_validation2d')

# ------------------------------------------------------------------ Fig 3: anytime
A = R['anytime']; Gu = Z['any_Gup']; Gl = Z['any_Glo']; k = np.arange(len(Gu))
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
ax[0].fill_between(k, Gl, Gu, color=CBAND, alpha=.18, label='interval at iteration k')
ax[0].plot(k, Gu, color=CUP, lw=1.2, label='upper bound, potential network')
ax[0].plot(k, Gl, color=CLO, lw=1.2, label='lower bound, flux network')
ax[0].axhline(A['steady_up'], color=CUP, ls='--', lw=.8); ax[0].axhline(A['steady_lo'], color=CLO, ls='--', lw=.8)
ax[0].axhspan(v['ref_lo'], v['ref_up'], color='0.4', alpha=.6, label='reference interval, n = 512')
ax[0].set_xlim(0, 3000); ax[0].set_ylim(v['ref_lo']-0.05, v['ref_lo']+0.08)
ax[0].set_xlabel('scatter/connect iteration k'); ax[0].set_ylabel('conductance G'); ax[0].legend(loc='upper right')
ax[0].set_title('(a) Enclosure at every iteration, n = 32')
ax[1].semilogy(k, 0.5*(Gu-Gl), color='C2', label='half-width at iteration k')
ax[1].axhline(0.5*(A['steady_up']-A['steady_lo']), color='k', ls='--', lw=.9, label='steady-state half-width (grid floor)')
ax[1].set_xlabel('iteration k'); ax[1].set_ylabel('certified half-width'); ax[1].legend(); ax[1].set_title('(b) Monotone tightening')
ax[2].semilogy(k, Z['any_eU']/Z['any_eU'][0], color=CUP, label='potential network')
ax[2].semilogy(k, Z['any_eL']/Z['any_eL'][0], color=CLO, label='flux network')
ax[2].set_xlabel('iteration k'); ax[2].set_ylabel('energy-norm distance to steady state (normalised)'); ax[2].legend(); ax[2].set_title('(c) Convergence of each network')
fig.tight_layout(); save(fig, 'Fig3_anytime')

# ------------------------------------------------------------------ Fig 4: contrast and geometry
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
c = R['contrast']
ax[0].loglog([r['contrast'] for r in c], [r['relwidth'] for r in c], 'o-', color='C4')
ax[0].set_xlabel('contrast σ_in / σ_out'); ax[0].set_ylabel('relative certified width'); ax[0].set_title('(a) Contrast sweep, n = 128')
ax[0].grid(True, which='both', alpha=.3)
for name, mk in [('square', 'o'), ('L', 's'), ('checkerboard', '^')]:
    g = R['geometry'][name]; rr = g['rows']
    ax[1].loglog([1/r['n'] for r in rr], [r['gap'] for r in rr], mk+'-', label=f'{name}: O(h^{g["rate"]:.2f})')
ax[1].set_xlabel('grid step h'); ax[1].set_ylabel('width of the certified interval'); ax[1].legend(); ax[1].set_title('(b) Geometry sweep, contrast 100')
ax[1].grid(True, which='both', alpha=.3)
fig.tight_layout(); save(fig, 'Fig4_contrast_geometry')

# ------------------------------------------------------------------ Fig 5: benchmark
if 'error' not in R['benchmark']:
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for col, (name, lab) in enumerate([('smooth', 'smooth coefficient'), ('square', 'aligned square inclusion')]):
        b = R['benchmark'][name]
        T = b['tlm']; D = b['dgm']
        ax[col].loglog([r['dof'] for r in T], [r['half'] for r in T], 'o-', color=CUP, label='Cartesian dual pair (this work)')
        ax[col].loglog([r['dof'] for r in D], [r['half'] for r in D], 's-', color=CLO, label='triangular dual pair (reference solver)')
        bad = [r for r in D if not r['compat']]
        if bad:
            ax[col].loglog([r['dof'] for r in bad], [r['half'] for r in bad], 'x', color='k', ms=9, mew=1.6,
                           label='incompatible with the other certificate')
        ax[col].set_xlabel('degrees of freedom'); ax[col].set_ylabel('certified half-width')
        ax[col].set_title(f'({"ab"[col]}) {lab}'); ax[col].grid(True, which='both', alpha=.3); ax[col].legend(loc='upper right')
    fig.tight_layout(); save(fig, 'Fig5_benchmark')

# ------------------------------------------------------------------ Fig 6: geometric error
P = R['package']; cf = P['conforming']; un = P['uniform']; I3 = R['incl3d']
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
hu = [r['hx'] for r in un]; hc = [r['hx'] for r in cf]
ax[0].plot(hu, [r['pad_w'] for r in un], 'o-', color='C1', label='pad width, uniform grid')
ax[0].plot(hc, [r['pad_w'] for r in cf], 's-', color='C2', label='pad width, conforming grid')
ax[0].axhline(P['pad_w_cad'], color='k', ls=':', lw=1, label='CAD value')
ax0b = ax[0].twinx()
ax0b.plot(hu, [r['gap_eff'] for r in un], 'o--', color='C1', alpha=.6, label='gap, uniform grid')
ax0b.plot(hc, [r['gap_eff'] for r in cf], 's--', color='C2', alpha=.6, label='gap, conforming grid')
ax0b.axhline(P['gap_mm'], color='k', ls=':', lw=1); ax0b.set_ylabel('represented gap (mm)')
ax[0].set_xlabel('grid step h (mm)'); ax[0].set_ylabel('represented pad width (mm)'); ax[0].invert_xaxis(); ax[0].set_ylim(3.45, 4.08)
h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = ax0b.get_legend_handles_labels(); ax[0].legend(h1+h2, l1+l2, fontsize=6.2, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
ax[0].set_title('(a) Geometry seen by the grid')
for i, r in enumerate(un):
    ax[1].plot([i, i], [r['C_lo']*1e15, r['C_up']*1e15], color='C1', lw=4, solid_capstyle='butt', label='uniform grids' if i == 0 else None)
for i, r in enumerate(cf):
    ax[1].plot([i+0.25, i+0.25], [r['C_lo']*1e15, r['C_up']*1e15], color='C2', lw=4, solid_capstyle='butt', label='conforming grids' if i == 0 else None)
ax[1].axhspan(P['inter_conf'][0], P['inter_conf'][1], color='C2', alpha=.15, label='intersection, conforming')
ax[1].set_xticks(range(5)); ax[1].set_xticklabels([f'grid {i+1}' for i in range(5)])
ax[1].set_ylabel('pad-to-substrate capacitance (fF)'); ax[1].legend(); ax[1].set_title('(b) Package: five enclosures per grid family')
for i, r in enumerate(I3['nonconforming']):
    ax[2].plot([i, i], [r['lo'], r['up']], color='C1', lw=4, solid_capstyle='butt', label='non-conforming n = 8, 12, 16' if i == 0 else None)
for i, r in enumerate(I3['conforming']):
    ax[2].plot([i+0.25, i+0.25], [r['lo'], r['up']], color='C2', lw=4, solid_capstyle='butt', label='conforming n = 10, 20, 30' if i == 0 else None)
ax[2].axhspan(I3['inter_ok'][0], I3['inter_ok'][1], color='C2', alpha=.15, label='intersection, conforming')
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(['coarse', 'medium', 'fine']); ax[2].set_ylabel('conductance G')
ax[2].legend(); ax[2].set_title('(c) Cubic inclusion: same contrast')
fig.tight_layout(); save(fig, 'Fig6_geometric_error')

# ------------------------------------------------------------------ Fig 7: package
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import ic_package_refine as icp
r = icp.solve(0.50, 0.14, verbose=False, return_fields=True)
xn, yn, zn = r['xn'], r['yn'], r['zn']; B = icp.B; pad_bb = icp.pad_bb; sub_bb = icp.sub_bb
fig = plt.figure(figsize=(10.5, 3.3))
axA = fig.add_subplot(1, 3, 1)
axA.add_patch(Rectangle((sub_bb[0, 0], sub_bb[0, 1]), sub_bb[1, 0]-sub_bb[0, 0], sub_bb[1, 1]-sub_bb[0, 1], fc='#d9c200', ec='k', lw=.4, zorder=0))
for i in [i for i in range(len(B)) if i not in (0, 2, 4)]:
    bb = B[i].bounds
    axA.add_patch(Rectangle((bb[0, 0], bb[0, 1]), bb[1, 0]-bb[0, 0], bb[1, 1]-bb[0, 1], fc='#b8b8b8', ec='0.4', lw=.2, zorder=1))
axA.add_patch(Rectangle((pad_bb[0, 0], pad_bb[0, 1]), pad_bb[1, 0]-pad_bb[0, 0], pad_bb[1, 1]-pad_bb[0, 1], fc='#c0392b', ec='k', lw=.6, zorder=2))
axA.text(5, 5, 'die pad', color='w', ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)
axA.text(5, 11.2, 'leads (not modelled)', color='0.3', ha='center', fontsize=7)
axA.text(10.4, -1.3, 'substrate', color='0.45', ha='center', fontsize=7)
axA.set_xlim(-2.2, 12.2); axA.set_ylim(-2.2, 12.2); axA.set_aspect('equal'); axA.set_xticks([]); axA.set_yticks([])
axA.set_title('(a) Package, top view')
jmid = int(np.argmin(np.abs(0.5*(yn[:-1]+yn[1:])-pad_bb[:, 1].mean())))
U = r['u'][:, jmid, :]
axB = fig.add_subplot(1, 3, 2)
im = axB.pcolormesh(xn, zn, U, cmap='viridis', vmin=0, vmax=1, shading='auto')
axB.contour(xn, zn, U, levels=np.linspace(0.05, 0.95, 12), colors='w', linewidths=.3, alpha=.5)
axB.add_patch(Rectangle((pad_bb[0, 0], pad_bb[0, 2]), pad_bb[1, 0]-pad_bb[0, 0], max(pad_bb[1, 2]-pad_bb[0, 2], 0.03), ec='r', fc='r', lw=1.2))
axB.axhspan(sub_bb[0, 2], sub_bb[1, 2], color='0.25', alpha=.85)
axB.text(9.2, sub_bb[:, 2].mean(), 'substrate, u = 0', fontsize=7, color='w', va='center', ha='center')
axB.text(5, pad_bb[1, 2]+0.18, 'pad, u = 1', fontsize=7, color='r', ha='center', fontweight='bold')
axB.set_xlim(xn[0], xn[-1]); axB.set_ylim(-1.1, 1.05); axB.set_xlabel('x (mm)'); axB.set_ylabel('z (mm)')
axB.set_title('(b) Potential, section through the pad centre'); fig.colorbar(im, ax=axB, shrink=.8, label='u')
axC = fig.add_subplot(1, 3, 3)
cells = np.array([q['ncell'] for q in cf]); lo = np.array([q['C_lo'] for q in cf])*1e15; up = np.array([q['C_up'] for q in cf])*1e15
axC.fill_between(cells, lo, up, color=CBAND, alpha=.18, label='certified interval')
axC.plot(cells, up, 'o-', color=CUP, ms=4, label='upper bound (conforming trilinear)')
axC.plot(cells, lo, 's-', color=CLO, ms=4, label='lower bound (equilibrated face flux)')
axC.plot(cells, 0.5*(lo+up), 'k--', lw=1, label='midpoint (not certified)')
axC.axhline(P['C_pp_fF'], color='0.5', lw=1, ls=':', label=f'parallel plate, no fringing: {P["C_pp_fF"]:.0f} fF')
axC.set_xscale('log'); axC.set_xlabel('number of cells'); axC.set_ylabel('pad-to-substrate capacitance (fF)')
axC.set_title('(c) Enclosures on conforming grids'); axC.grid(True, which='both', alpha=.25); axC.legend(loc='lower right')
fig.tight_layout(); save(fig, 'Fig7_package')
print('all figures written to', FIGD)
