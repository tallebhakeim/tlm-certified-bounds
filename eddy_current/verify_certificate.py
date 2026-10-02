"""Verification of Theorem 1 (guaranteed impedance bounds) on the exact layered benchmark.

T1  static limit w = 0: bracket = classical Prager-Synge, contains the exact L;
T2  all frequencies (a/delta 0.1 .. 10) and all grids: exact L and R inside the certified bracket;
    bracket half-width shrinks as O(h^2) (second order in eta);
T3  anytime: for NON-converged approximations (Galerkin + random perturbation, truncated
    iterations) the bracket built with the corrected output still contains the exact value.
"""
import sys, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import eddy_certified2d as ec
from exact_slab import slab_exact

MU0 = ec.MU0; NU0 = 1/MU0; SIG = 3.5e7          # aluminium plate
H, T = 10e-3, 2e-3                             # box height, plate thickness
yP0, yP1 = 2e-3, 2e-3 + T                      # plate
yC0, yC1 = 5e-3, 5.5e-3                        # current layer (coil)
Jc = 1e6; W = 4e-3; I = Jc*(yC1 - yC0)*W
layers = [(0, yP0, NU0, 0, 0), (yP0, yP1, NU0, SIG, 0), (yP1, yC0, NU0, 0, 0), (yC0, yC1, NU0, 0, Jc), (yC1, H, NU0, 0, 0)]
fails = 0


def run(hmax, w, perturb=0.0, seed=0):
    xs = ec.conforming_axis([0, W], hmax); ys = ec.conforming_axis([0, yP0, yP1, yC0, yC1, H], hmax)
    g = ec.Grid(xs, ys, bc="NNDD")
    Y = np.broadcast_to(g.yc[:, None], (g.ny, g.nx))
    nu = np.full((g.ny, g.nx), NU0); sg = np.where((Y > yP0) & (Y < yP1), SIG, 0.0)
    J = np.where((Y > yC0) & (Y < yC1), Jc, 0.0)
    KMf = ec.assemble(g, nu, sg, J)
    At = None
    if perturb:
        A0 = ec.solve(g, *KMf, w); rng = np.random.default_rng(seed)
        At = A0*(1 + perturb*(rng.standard_normal(g.N) + 1j*rng.standard_normal(g.N)))
        At[g.dirichlet_nodes()] = 0
    return ec.certify(g, nu, sg, J, w, I, At=At, KMf=KMf)


print("T1 static limit (w -> 0)")
l0, _ = slab_exact([(a, b, c, 0, e) for a, b, c, d, e in layers], 1.0, W)
L0 = l0.real/I**2
for hmax in [1e-3, 5e-4, 2.5e-4]:
    xs = ec.conforming_axis([0, W], hmax); ys = ec.conforming_axis([0, yP0, yP1, yC0, yC1, H], hmax)
    g = ec.Grid(xs, ys, bc="NNDD"); Y = np.broadcast_to(g.yc[:, None], (g.ny, g.nx))
    nu = np.full((g.ny, g.nx), NU0); sg = np.zeros((g.ny, g.nx)); J = np.where((Y > yC0) & (Y < yC1), Jc, 0.0)
    c = ec.certify(g, nu, sg, J, 0.0, I)
    ok = c['L_lo'] <= L0 <= c['L_up']; fails += not ok
    print(f"  h={hmax:.1e}: L in [{c['L_lo']:.6e}, {c['L_up']:.6e}]  exact {L0:.6e}  {'OK' if ok else 'FAIL'}  eq.res {c['eq_res']:.1e}")

print("\nT2 frequency sweep x refinement (exact inside bracket?)")
print("  a/delta   hmax     L_lo/L    L_up/L    R_lo/R    R_up/R   half-width L %  eff. index")
rows = []
for ad in [0.1, 0.5, 1, 2, 5, 10]:
    w = 2*(ad/T)**2/(MU0*SIG)
    lex, _ = slab_exact(layers, w, W); Lx = lex.real/I**2; Rx = -w*lex.imag/I**2
    for hmax in [1e-3, 5e-4, 2.5e-4, 1.25e-4]:
        c = run(hmax, w)
        okL = c['L_lo'] <= Lx <= c['L_up']; okR = c['R_lo'] <= Rx <= c['R_up']; fails += (not okL) + (not okR)
        eff = (c['L_up'] - c['L_lo'])/2/abs(c['L'] - Lx) if abs(c['L'] - Lx) > 0 else np.inf
        rows.append((ad, hmax, (c['L_up'] - c['L_lo'])/2/Lx))
        print(f"  {ad:5.1f}   {hmax:.2e}  {c['L_lo']/Lx:.6f}  {c['L_up']/Lx:.6f}  {c['R_lo']/Rx:8.4f}  {c['R_up']/Rx:8.4f}   {100*(c['L_up']-c['L_lo'])/2/Lx:9.4f}   {eff:7.2f}  {'OK' if okL and okR else 'FAIL'}")
rows = np.array(rows)
for ad in [0.1, 1, 10]:
    r = rows[rows[:, 0] == ad]; p = np.polyfit(np.log(r[:, 1]), np.log(r[:, 2]), 1)[0]
    print(f"  a/delta={ad}: half-width ~ h^{p:.2f}")

print("\nT3 anytime: perturbed (non-Galerkin) approximations")
for ad in [0.5, 2, 5]:
    w = 2*(ad/T)**2/(MU0*SIG); lex, _ = slab_exact(layers, w, W); Lx = lex.real/I**2; Rx = -w*lex.imag/I**2
    for pert in [1e-3, 1e-2, 1e-1]:
        viol = 0
        for s in range(5):
            c = run(5e-4, w, perturb=pert, seed=s)
            viol += not (c['L_lo'] <= Lx <= c['L_up'] and c['R_lo'] <= Rx <= c['R_up'])
        fails += viol
        print(f"  a/delta={ad}, perturbation {pert:.0e}: half-width L {100*(c['L_up']-c['L_lo'])/2/Lx:8.3f} %  violations {viol}/5")
print(f"\nTOTAL FAILURES: {fails}")
