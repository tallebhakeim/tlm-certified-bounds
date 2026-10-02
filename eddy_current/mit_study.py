"""Industrial MIT application: 8-coil array, certified impedance matrix (self + mutual via
polarisation / Cauchy-Schwarz), certified detection of a metallic inclusion entry by entry.
An entry Z_ij 'certifies' the inclusion when the discs |Z - Z_h| <= dZ of the empty and the
loaded configurations are disjoint. Geometry-conforming grids."""
import sys, time, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import eddy_certified2d as ec
MU0 = ec.MU0; NU0 = 1/MU0
NC = 8; RC = 40e-3; S = 16e-3; CS = 3e-3       # coils: ring radius, conductor spacing, conductor size
BOX = 120e-3; FINE = 60e-3; F = 1e3; W = 2*np.pi*F; SIG = 3.5e7
INCL = [((0.0, 0.0), 30e-3), ((12e-3, 6e-3), 20e-3), ((12e-3, 6e-3), 10e-3)]


def graded(a, b, h0, ratio=1.25, hc=8e-3):
    s = np.sign(b - a); L = abs(b - a); pts = [0.0]; h = h0
    while pts[-1] + h < L - 1e-12: pts.append(pts[-1] + h); h = min(h*ratio, hc)
    if L - pts[-1] < 0.5*h and len(pts) > 1: pts[-1] = L
    else: pts.append(L)
    return a + s*np.array(pts)


def conductors():
    out = []
    for k in range(NC):
        th = 2*np.pi*k/NC; cx, cy = RC*np.cos(th), RC*np.sin(th); tx, ty = -np.sin(th), np.cos(th)
        for sgn in (+1, -1):
            out.append((k, sgn, round((cx + sgn*tx*S/2)*1e5)/1e5, round((cy + sgn*ty*S/2)*1e5)/1e5))
    return out


def build(hf, incl=None):
    bx, by = {-FINE, FINE}, {-FINE, FINE}
    for _, _, x, y in conductors(): bx |= {x - CS/2, x + CS/2}; by |= {y - CS/2, y + CS/2}
    if incl: (ix, iy), a = incl; bx |= {ix - a/2, ix + a/2}; by |= {iy - a/2, iy + a/2}
    xs = np.concatenate([graded(-FINE, -BOX, hf)[::-1][:-1], ec.conforming_axis(sorted(bx), hf), graded(FINE, BOX, hf)[1:]])
    ys = np.concatenate([graded(-FINE, -BOX, hf)[::-1][:-1], ec.conforming_axis(sorted(by), hf), graded(FINE, BOX, hf)[1:]])
    g = ec.Grid(xs, ys, "DDDD"); X = np.broadcast_to(g.xc[None, :], (g.ny, g.nx)); Y = np.broadcast_to(g.yc[:, None], (g.ny, g.nx))
    nu = np.full((g.ny, g.nx), NU0); sg = np.zeros((g.ny, g.nx))
    if incl: (ix, iy), a = incl; sg[(np.abs(X - ix) < a/2) & (np.abs(Y - iy) < a/2)] = SIG
    Js = [np.zeros((g.ny, g.nx)) for _ in range(NC)]
    for k, sgn, x, y in conductors():
        Js[k][(np.abs(X - x) < CS/2) & (np.abs(Y - y) < CS/2)] = sgn/CS**2
    return g, nu, sg, Js


res = {}
for hf in [1.0e-3, 0.5e-3, 0.25e-3]:
    t0 = time.time()
    g, nu, sg, Js = build(hf); Z0, dZ0, dX0, _ = ec.impedance_matrix(g, nu, sg, Js, W, [1.0]*NC)
    res[f'Z0_{hf}'] = Z0; res[f'dZ0_{hf}'] = dZ0
    print(f"hf={hf*1e3:.2f} mm N={g.N}: empty array, self L {Z0[0,0].imag/W*1e6:.4f} uH/m +- {dX0[0,0]/W*1e6:.1e}; "
          f"|Z01|={abs(Z0[0,1]):.3e} r={dZ0[0,1]:.1e}; |Z04|={abs(Z0[0,4]):.3e} r={dZ0[0,4]:.1e}", flush=True)
    for m, inc in enumerate(INCL):
        g, nu, sg, Js = build(hf, inc); Z1, dZ1, dX1, _ = ec.impedance_matrix(g, nu, sg, Js, W, [1.0]*NC)
        iu = np.triu_indices(NC); cert = np.abs(Z1 - Z0)[iu] > (dZ1 + dZ0)[iu]
        ratio = (np.abs(Z1 - Z0)/(dZ1 + dZ0))[iu]
        res[f'Z1_{m}_{hf}'] = Z1; res[f'dZ1_{m}_{hf}'] = dZ1
        print(f"   inclusion {m} {inc}: {cert.sum()}/{len(cert)} entries certify it; max |dZ|/(r0+r1) = {ratio.max():.2f}; "
              f"self entries {np.sum(cert[[i for i,(a,b) in enumerate(zip(*iu)) if a==b]])}/8", flush=True)
    print(f"   ({time.time()-t0:.0f} s)", flush=True)
np.savez('mit_results.npz', **res)
