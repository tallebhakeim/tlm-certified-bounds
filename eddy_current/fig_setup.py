"""Fig. 1: the three configurations (exact layered benchmark, NDT probe, 8-coil MIT array)."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw=dict(width_ratios=[0.8, 1.9, 1.3]))
# (a) layered benchmark (mm)
a = ax[0]
a.add_patch(Rectangle((0, 0), 4, 10, fc='none', ec='k', lw=1))
a.add_patch(Rectangle((0, 2), 4, 2, fc='#9ab', ec='k', lw=.6)); a.text(2, 3, r'plate $\sigma$', ha='center', va='center')
a.add_patch(Rectangle((0, 5), 4, .5, fc='#d55', ec='k', lw=.6)); a.text(2, 7.2, 'current\nlayer $J$', ha='center', va='center', fontsize=7)
a.text(2, -0.8, r'$A=0$', ha='center'); a.text(2, 10.3, r'$A=0$', ha='center')
a.text(-0.6, 8.5, r'$\partial_n A=0$', rotation=90, ha='center', va='center', fontsize=7); a.text(4.6, 8.5, r'$\partial_n A=0$', rotation=90, ha='center', va='center', fontsize=7)
a.set_xlim(-1.3, 5.3); a.set_ylim(-1.5, 11); a.set_aspect('equal'); a.axis('off'); a.set_title('(a) layered benchmark', fontsize=8)
# (b) NDT probe (mm)
b = ax[1]; T = 2; lo = .5
b.add_patch(Rectangle((-12, 0), 24, T, fc='#9ab', ec='k', lw=.6)); b.text(-11.5, 1, 'Al plate', va='center', fontsize=7)
b.add_patch(Rectangle((-5, 0), 10, .8, fc='w', ec='#c33', lw=.8, hatch='////')); b.text(0, -1.2, 'far-side wall loss $d$', ha='center', color='#c33', fontsize=7)
for xc, s in ((2.5, r'$\odot$'), (-2.5, r'$\otimes$')):
    b.add_patch(Rectangle((xc - .75, T + lo), 1.5, 1.5, fc='#d55', ec='k', lw=.6)); b.text(xc, T + lo + .75, s, ha='center', va='center', fontsize=9)
b.annotate('lift-off', xy=(3.3, T + .25), xytext=(6.5, T + 2.6), fontsize=7, arrowprops=dict(arrowstyle='->', lw=.6))
b.annotate('', xy=(-12.8, 0), xytext=(-12.8, T), arrowprops=dict(arrowstyle='<->', lw=.6)); b.text(-13.2, 1, '$T$', ha='right', va='center')
b.set_xlim(-14, 12.5); b.set_ylim(-2.2, 5.5); b.set_aspect('equal'); b.axis('off'); b.set_title('(b) eddy-current NDT probe', fontsize=8)
# (c) MIT ring
c = ax[2]; RC = 40; S = 16
c.add_patch(Rectangle((-60, -60), 120, 120, fc='none', ec='0.6', ls='--', lw=.6))
for k in range(8):
    th = 2*np.pi*k/8; cx, cy = RC*np.cos(th), RC*np.sin(th); tx, ty = -np.sin(th), np.cos(th)
    for sg in (1, -1): c.add_patch(Rectangle((cx + sg*tx*S/2 - 1.5, cy + sg*ty*S/2 - 1.5), 3, 3, fc='#d55', ec='k', lw=.4))
    c.text(1.3*cx, 1.3*cy, str(k + 1), ha='center', va='center', fontsize=7)
c.add_patch(Rectangle((-15, -15), 30, 30, fc='none', ec='#467', lw=.7, ls='--')); c.text(0, -19.5, 'incl. 1 (30 mm)', ha='center', fontsize=5.5)
c.add_patch(Rectangle((2, -4), 20, 20, fc='#9ab', ec='k', lw=.5, alpha=.8)); c.add_patch(Rectangle((7, 1), 10, 10, fc='#678', ec='k', lw=.5))
c.text(12, 18, 'incl. 2, 3', ha='center', fontsize=5.5)
c.set_xlim(-66, 66); c.set_ylim(-66, 66); c.set_aspect('equal'); c.axis('off'); c.set_title('(c) 8-coil MIT array', fontsize=8)
fig.tight_layout(pad=.3); fig.savefig('fig1_setup.pdf'); fig.savefig('fig1_setup.png', dpi=200)
