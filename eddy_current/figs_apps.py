"""Figs. 3-7 from the cached results (tlm_anytime.npz, ndt_results.npz, ndt_detect_fine.npz, mit_results.npz)."""
import numpy as np, matplotlib.pyplot as plt, os
plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
C2 = 0.5*(1 + np.sqrt(2)); T = 2e-3

# ---- Fig. 3 anytime ----
d = np.load('tlm_anytime.npz'); rows, ref, gal = d['rows'], d['ref'], d['gal']
fig, ax = plt.subplots(figsize=(3.5, 2.3))
ax.axhspan(gal[1]*1e6, gal[2]*1e6, color='0.85', lw=0, label='direct Galerkin, same grid')
ax.axhline(ref[1]*1e6, color='k', ls='--', lw=.7); ax.axhline(ref[2]*1e6, color='k', ls='--', lw=.7, label='reference bracket ($h/4$)')
for spp, col, dx in ((16, 'C0', -0.12), (32, 'C1', 0), (64, 'C2', 0.12)):
    r = rows[rows[:, 0] == spp]; p = r[:, 1]
    ax.errorbar(p*(1 + dx*0.3), r[:, 2]*1e6, yerr=[(r[:, 2] - r[:, 3])*1e6, (r[:, 4] - r[:, 2])*1e6], fmt='o', ms=2.5, color=col, lw=.8, capsize=1.5, label=f'TLM march, {spp} steps/period')
ax.set_xscale('log'); ax.set_xlabel('period $p$ of the time march'); ax.set_ylabel(r'$L$ ($\mu$H/m)')
ax.legend(fontsize=5.5, loc='upper right'); ax.grid(alpha=.3, which='both', lw=.4)
fig.tight_layout(pad=.3); fig.savefig('fig3_anytime.pdf'); fig.savefig('fig3_anytime.png', dpi=200); plt.close(fig)

# ---- Fig. 4 NDT refinement ----
d = np.load('ndt_results.npz', allow_pickle=True); R = d['refine']
fig, ax = plt.subplots(2, 1, figsize=(3.5, 3.4), sharex=True)
for f, col, dx in ((1e3, 'C0', .93), (5e3, 'C1', 1.0), (20e3, 'C3', 1.07)):
    r = R[R[:, 0] == f]; h = r[:, 1]*1e3
    for k, (i, lo, up) in enumerate(((3, 4, 5), (6, 7, 8))):
        mid = 0.5*(r[-1, lo] + r[-1, up])
        ax[k].errorbar(h*dx, r[:, i]/mid, yerr=[(r[:, i] - r[:, lo])/mid, (r[:, up] - r[:, i])/mid], fmt='o', ms=2.5, color=col, capsize=2, lw=.9, label=f'{f/1e3:g} kHz')
for k, lab in enumerate(['$L/L_{\\rm mid}$', '$R/R_{\\rm mid}$']):
    ax[k].axhline(1, color='k', lw=.5); ax[k].set_ylabel(lab); ax[k].grid(alpha=.3, which='both', lw=.4)
ax[0].set_ylim(0.93, 1.03); ax[1].set_ylim(0.55, 1.45)
ax[1].set_xscale('log'); ax[1].set_xticks([0.05, 0.1, 0.2, 0.4]); ax[1].set_xticklabels(['0.05', '0.1', '0.2', '0.4']); ax[1].minorticks_off(); ax[1].set_xlabel('$h$ (mm)'); ax[0].legend(fontsize=6, ncol=3, loc='lower left')
ax[0].set_title('(a) inductance', fontsize=8); ax[1].set_title('(b) resistance', fontsize=8)
fig.tight_layout(pad=.3); fig.savefig('fig4_ndt_refine.pdf'); fig.savefig('fig4_ndt_refine.png', dpi=200); plt.close(fig)

# ---- Fig. 5 impedance plane ----
tr = d['traj']; X0 = 2*np.pi*5e3*float(d['L0_5000'])
fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.4), gridspec_kw=dict(height_ratios=[1, 1.3]))
for typ, col, lab in ((0, 'C0', 'lift-off 0.25 to 1.5 mm'), (1, 'C3', 'wall loss 10 to 50 % of $T$')):
    t = tr[tr[:, 0] == typ]
    ax[0].plot(t[:, 2]/X0, t[:, 3]/X0, 'o-', color=col, ms=3, lw=.9, label=lab)
    if typ == 1:
        for z in t:
            ax[1].add_patch(plt.Circle((z[2]/X0, z[3]/X0), z[4]/X0, fc=col, alpha=.25, ec=col, lw=.5))
t = tr[tr[:, 0] == 1]; ax[1].plot(t[:, 2]/X0, t[:, 3]/X0, 'o-', color='C3', ms=2.5, lw=.8)
t0 = tr[(tr[:, 0] == 0) & (np.isclose(tr[:, 1], 0.5e-3))][0]
ax[1].plot(t0[2]/X0, t0[3]/X0, 's', color='k', ms=3.5); ax[1].add_patch(plt.Circle((t0[2]/X0, t0[3]/X0), t0[4]/X0, fc='k', alpha=.2, lw=0))
ax[1].annotate('sound plate', (t0[2]/X0, t0[3]/X0), xytext=(8, -12), textcoords='offset points', fontsize=6.5)
tt = np.vstack([tr[tr[:, 0] == 1][:, 2:4], t0[None, 2:4]])/X0; m = 0.004
ax[1].set_xlim(tt[:, 0].min() - m, tt[:, 0].max() + m); ax[1].set_ylim(tt[:, 1].min() - m, tt[:, 1].max() + m)
ax[1].set_aspect('equal', adjustable='box')
for a in ax: a.grid(alpha=.3, lw=.4); a.set_xlabel(r'$R/X_0$'); a.set_ylabel(r'$\omega L/X_0$')
ax[0].legend(fontsize=6); ax[0].set_title('(a) trajectories', fontsize=8); ax[1].set_title('(b) wall-loss trajectory with certified disks', fontsize=8)
fig.tight_layout(pad=.3); fig.savefig('fig5_ndt_plane.pdf'); fig.savefig('fig5_ndt_plane.png', dpi=200); plt.close(fig)

# ---- Fig. 6 detection ----
if os.path.exists('ndt_detect_fine.npz'):
    D = np.load('ndt_detect_fine.npz')['rows']
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for hf, col in ((0.4e-3, 'C7'), (0.2e-3, 'C0'), (0.1e-3, 'C1'), (0.05e-3, 'C2')):
        r = D[(D[:, 0] == 5e3) & np.isclose(D[:, 1], hf)]
        ax.semilogy(100*r[:, 2]/T, r[:, 4] + r[:, 5], '-', color=col, lw=1.2, label=f'threshold, $h$ = {hf*1e3:g} mm')
    r = D[(D[:, 0] == 5e3) & np.isclose(D[:, 1], 0.05e-3)]
    ax.semilogy(100*r[:, 2]/T, r[:, 3], 'ko', ms=3, label='impedance change')
    ax.set_xlabel('far-side wall loss (% of $T$)'); ax.set_ylabel(r'$|\tilde Z_d-\tilde Z_0|$, $\rho_d+\rho_0$ ($\Omega$/m)')
    ax.legend(fontsize=6); ax.grid(alpha=.3, which='both', lw=.4)
    fig.tight_layout(pad=.3); fig.savefig('fig6_ndt_detect.pdf'); fig.savefig('fig6_ndt_detect.png', dpi=200); plt.close(fig)

# ---- Fig. 7 MIT ----
if os.path.exists('mit_results.npz'):
    M = np.load('mit_results.npz'); hs = sorted({float(k.split('_')[-1]) for k in M.files if k.startswith('Z0_')}, reverse=True)
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 2.0))
    for a, hf in zip(ax, (hs[0], hs[-1])):
        Z0, r0, Z1, r1 = M[f'Z0_{hf}'], M[f'dZ0_{hf}'], M[f'Z1_1_{hf}'], M[f'dZ1_1_{hf}']
        rat = np.abs(Z1 - Z0)/(r0 + r1); rat = np.where(np.triu(np.ones_like(rat)) > 0, rat, np.nan)
        im = a.imshow(np.log10(rat), cmap='RdBu', vmin=-2, vmax=2, origin='upper')
        a.set_xticks(range(8)); a.set_yticks(range(8)); a.set_xticklabels(range(1, 9), fontsize=5.5); a.set_yticklabels(range(1, 9), fontsize=5.5)
        a.set_title(f'({"ab"[list(ax).index(a)]}) $h$ = {hf*1e3:g} mm\n{int(np.nansum(rat > 1))}/36 certified', fontsize=6.5)
    cb = fig.colorbar(im, ax=ax, shrink=.85); cb.set_label(r'$\log_{10}$ ratio', fontsize=6.5); cb.ax.tick_params(labelsize=5.5)
    fig.savefig('fig7_mit.pdf', bbox_inches='tight'); fig.savefig('fig7_mit.png', dpi=200, bbox_inches='tight'); plt.close(fig)
print('figures written')
