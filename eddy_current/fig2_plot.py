"""Fig. 2 from verification.npz (produced by fig_verification.py)."""
import numpy as np, matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
d = np.load('verification.npz'); res, HS, AD = d['res'], d['HS'], d['AD']
hw = (res[..., 2] - res[..., 1])/2/res[..., 0]; eff = (res[..., 2] - res[..., 1])/2/np.abs(res[..., 6] - res[..., 0])
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.2))
cols = plt.cm.viridis(np.linspace(0, .85, len(HS)))
for k in range(len(HS)):
    ax[0].loglog(AD, res[k, :, 2]/res[k, :, 0] - 1, '-', color=cols[k], lw=1, label=f'$h$ = {HS[k]*1e3:.3g} mm')
    ax[0].loglog(AD, 1 - res[k, :, 1]/res[k, :, 0], '--', color=cols[k], lw=1)
ax[0].set_xlabel(r'$d/\delta$'); ax[0].set_ylabel(r'$L_+/L-1$ (—), $1-L_-/L$ (- -)')
ax[0].legend(fontsize=5.5, loc='lower right', ncol=1); ax[0].set_title('(a) margins of the bracket on $L$', fontsize=8); ax[0].set_ylim(1e-5, 3)
for m, mk in zip([0, 6, 12], 'os^'):
    ax[1].loglog(HS*1e3, hw[:, m], mk + '-', ms=3.5, lw=.8, label=f'$d/\\delta$ = {AD[m]:.2g}')
ax[1].loglog([0.0625, 0.25], [2e-4, 3.2e-3], 'k--', lw=.7); ax[1].text(0.09, 1.2e-3, '$h^2$', fontsize=7)
ax[1].set_xlabel('$h$ (mm)'); ax[1].set_ylabel('relative half-width on $L$'); ax[1].legend(fontsize=6); ax[1].set_title('(b) convergence', fontsize=8)
for m, mk in zip([0, 6, 12], 'os^'):
    ax[2].semilogx(HS*1e3, eff[:, m], mk + '-', ms=3.5, lw=.8)
ax[2].axhline(0.5*(1 + np.sqrt(2)), color='k', lw=.6, ls=':'); ax[2].text(0.07, 0.6, '$c=(1+\\sqrt{2})/2$', fontsize=6.5)
ax[2].set_ylim(0, 8.5); ax[2].set_xlabel('$h$ (mm)'); ax[2].set_ylabel('half-width / true error'); ax[2].set_title('(c) effectivity index', fontsize=8)
for a in ax: a.grid(alpha=.3, which='both', lw=.4)
fig.tight_layout(pad=.3); fig.savefig('fig2_verification.pdf'); fig.savefig('fig2_verification.png', dpi=200)
print("min margin upper", (res[..., 2]/res[..., 0] - 1).min(), "min margin lower", (1 - res[..., 1]/res[..., 0]).min())
