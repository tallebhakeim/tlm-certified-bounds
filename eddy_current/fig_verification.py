"""Fig. 2: verification on the exact layered benchmark (bracket vs frequency, convergence, effectivity)."""
import sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import eddy_certified2d as ec
from exact_slab import slab_exact
plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
MU0 = ec.MU0; NU0 = 1/MU0; SIG = 3.5e7; H, T = 10e-3, 2e-3; yP0, yP1 = 2e-3, 4e-3; yC0, yC1 = 5e-3, 5.5e-3
Jc = 1e6; W = 4e-3; I = Jc*(yC1 - yC0)*W
layers = [(0, yP0, NU0, 0, 0), (yP0, yP1, NU0, SIG, 0), (yP1, yC0, NU0, 0, 0), (yC0, yC1, NU0, 0, Jc), (yC1, H, NU0, 0, 0)]
HS = [1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5]; AD = np.geomspace(0.1, 10, 13)
res = np.zeros((len(HS), len(AD), 8))
for k, hmax in enumerate(HS):
    xs = ec.conforming_axis([0, W], hmax); ys = ec.conforming_axis([0, yP0, yP1, yC0, yC1, H], hmax)
    g = ec.Grid(xs, ys, bc="NNDD"); Y = np.broadcast_to(g.yc[:, None], (g.ny, g.nx))
    nu = np.full((g.ny, g.nx), NU0); sg = np.where((Y > yP0) & (Y < yP1), SIG, 0.); J = np.where((Y > yC0) & (Y < yC1), Jc, 0.)
    KMf = ec.assemble(g, nu, sg, J)
    for m, ad in enumerate(AD):
        w = 2*(ad/T)**2/(MU0*SIG); lx, _ = slab_exact(layers, w, W); Lx = lx.real/I**2; Rx = -w*lx.imag/I**2
        c = ec.certify(g, nu, sg, J, w, I, KMf=KMf)
        res[k, m] = [Lx, c['L_lo'], c['L_up'], Rx, c['R_lo'], c['R_up'], c['L'], c['eta']]
np.savez('verification.npz', res=res, HS=HS, AD=AD)
viol = np.sum((res[..., 1] > res[..., 0]*(1 + 1e-12)) | (res[..., 2] < res[..., 0]*(1 - 1e-12)) | (res[..., 4] > res[..., 3]) | (res[..., 5] < res[..., 3]))
print("violations:", viol, "of", 2*res.shape[0]*res.shape[1])
hw = (res[..., 2] - res[..., 1])/2/res[..., 0]; eff = (res[..., 2] - res[..., 1])/2/np.abs(res[..., 6] - res[..., 0])
for m in [0, 6, 12]:
    p = np.polyfit(np.log(HS[2:]), np.log(hw[2:, m]), 1)[0]; print(f"a/delta={AD[m]:.2f}: rates {p:.2f}, eff {eff[:, m]}")
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.2))
cols = plt.cm.viridis(np.linspace(0, .85, len(HS)))
for k in range(len(HS)):
    ax[0].fill_between(AD, res[k, :, 1]/res[k, :, 0], res[k, :, 2]/res[k, :, 0], color=cols[k], alpha=.35, lw=0,
                       label=f'$h$ = {HS[k]*1e3:.3g} mm')
ax[0].axhline(1, color='k', lw=.8); ax[0].set_xscale('log'); ax[0].set_ylim(0.8, 1.15)
ax[0].set_xlabel(r'$d/\delta$'); ax[0].set_ylabel(r'$[L_-, L_+]/L_{\rm exact}$'); ax[0].legend(fontsize=6, loc='lower left'); ax[0].set_title('(a) certified bracket on $L$', fontsize=8)
for m, mk in zip([0, 6, 12], 'os^'):
    ax[1].loglog(np.array(HS)*1e3, hw[:, m], mk + '-', ms=3.5, lw=.8, label=f'$d/\\delta$ = {AD[m]:.2g}')
ax[1].loglog([0.0625, 0.25], [2e-4, 3.2e-3], 'k--', lw=.7); ax[1].text(0.09, 1.2e-3, '$h^2$', fontsize=7)
ax[1].set_xlabel('$h$ (mm)'); ax[1].set_ylabel('relative half-width on $L$'); ax[1].legend(fontsize=6); ax[1].set_title('(b) convergence', fontsize=8)
for m, mk in zip([0, 6, 12], 'os^'):
    ax[2].semilogx(np.array(HS)*1e3, eff[:, m], mk + '-', ms=3.5, lw=.8)
ax[2].axhline(1, color='k', lw=.6); ax[2].set_ylim(0, 8); ax[2].set_xlabel('$h$ (mm)'); ax[2].set_ylabel(r'half-width / true error'); ax[2].set_title('(c) effectivity index', fontsize=8)
for a in ax: a.grid(alpha=.3, which='both', lw=.4)
fig.tight_layout(pad=.3); fig.savefig('fig2_verification.pdf'); fig.savefig('fig2_verification.png', dpi=200)
