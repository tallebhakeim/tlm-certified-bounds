"""Verification of the polarisation-based certificate for MUTUAL impedances (exact layered case).
Two current layers (coils 1, 2) around a conducting plate; exact Z11, Z22, Z12 from the closed form."""
import sys, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import eddy_certified2d as ec
from exact_slab import slab_exact
MU0 = ec.MU0; NU0 = 1/MU0; SIG = 3.5e7; W = 4e-3
H = 12e-3; P0, P1 = 5e-3, 7e-3; C1 = (2e-3, 2.5e-3); C2 = (9e-3, 9.5e-3); Jc = 1e6
I1 = Jc*(C1[1]-C1[0])*W; I2 = Jc*(C2[1]-C2[0])*W
def layers(j1, j2):
    return [(0, C1[0], NU0, 0, 0), (C1[0], C1[1], NU0, 0, j1), (C1[1], P0, NU0, 0, 0), (P0, P1, NU0, SIG, 0),
            (P1, C2[0], NU0, 0, 0), (C2[0], C2[1], NU0, 0, j2), (C2[1], H, NU0, 0, 0)]
fails = 0
for ad in [0.3, 1, 3]:
    w = 2*(ad/(P1-P0))**2/(MU0*SIG)
    l11 = slab_exact(layers(Jc, 0), w, W)[0]; l22 = slab_exact(layers(0, Jc), w, W)[0]
    lp = slab_exact(layers(Jc, Jc), w, W)[0]; l12 = (lp - l11 - l22)/2
    Zx = 1j*w*np.array([[l11/I1**2, l12/(I1*I2)], [l12/(I1*I2), l22/I2**2]])
    for hmax in [5e-4, 2.5e-4, 1.25e-4]:
        xs = ec.conforming_axis([0, W], hmax); ys = ec.conforming_axis([0, *C1, P0, P1, *C2, H], hmax)
        g = ec.Grid(xs, ys, bc="NNDD"); Y = np.broadcast_to(g.yc[:, None], (g.ny, g.nx))
        nu = np.full((g.ny, g.nx), NU0); sg = np.where((Y > P0) & (Y < P1), SIG, 0.)
        J1 = np.where((Y > C1[0]) & (Y < C1[1]), Jc, 0.); J2 = np.where((Y > C2[0]) & (Y < C2[1]), Jc, 0.)
        Z, dZ, dX, _ = ec.impedance_matrix(g, nu, sg, [J1, J2], w, [I1, I2])
        ok = np.all(np.abs(Z - Zx) <= dZ) and np.all(np.abs(Z.imag - Zx.imag) <= dX)
        fails += not ok
        print(f"a/d={ad:3.1f} h={hmax:.2e}  |Z12-Z12x|={abs(Z[0,1]-Zx[0,1]):.3e} <= dZ12={dZ[0,1]:.3e}   "
              f"X12 err {abs(Z[0,1].imag-Zx[0,1].imag):.3e} <= {dX[0,1]:.3e}   rel.radius Z12 {dZ[0,1]/abs(Zx[0,1]):.2e}  {'OK' if ok else 'FAIL'}")
print("TOTAL FAILURES:", fails)
