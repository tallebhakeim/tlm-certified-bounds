"""Anytime certificate along the passive time-domain march (Section IV).

Nodal capacitances C_i = (lumped sigma mass) are advanced with the trapezoidal rule, which is
exactly the TLM open-circuit stub (Bergeron / Dommel companion) model of a capacitor: a stub of
impedance Z_i = dt/(2 C_i) and round-trip delay dt. The resistive (nu) network is solved at each
step (connect step). The march is passive: the discrete energy is non-increasing for the free
response. After each period p the phasor At_p is demodulated from the midpoint states and
CERTIFIED by Theorem 2 (valid for any At, converged or not): the bracket on L contains the exact
value at every period and shrinks to the floor set by the grid and the time step.
"""
import time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from ndt_geometry import *

f = 5e3; w = 2*np.pi*f; hf = 0.2e-3
g, nu, sg, J = build(hf)
K, M, fv = ec.assemble(g, nu, sg, J)
ML = sp.diags(np.asarray(M.sum(axis=1)).ravel()).tocsr()          # lumped capacitances (stubs)
D = g.dirichlet_nodes(); fr = np.setdiff1d(np.arange(g.N), D)
Kf, Mf, ff = K[fr][:, fr], ML[fr][:, fr], fv[fr]
# fine-grid Galerkin reference (for display only; the certificate does not use it)
cref, _ = certify(0.05e-3, f)
cG = ec.certify(g, nu, sg, J, w, I_COIL, KMf=(K, M, fv))
print(f"reference (hf=0.05 mm) L in [{cref['L_lo']*1e6:.5f}, {cref['L_up']*1e6:.5f}] uH/m")
print(f"Galerkin on this grid     L in [{cG['L_lo']*1e6:.5f}, {cG['L_up']*1e6:.5f}] uH/m")

rows = []
for spp in [16, 32, 64]:
    T = 1/f; dt = T/spp
    lu = spla.splu((Mf/dt + Kf/2).tocsc()); B = (Mf/dt - Kf/2).tocsr()
    A = np.zeros(len(fr)); t0 = time.time()
    for p in range(1, 61):
        acc = np.zeros(len(fr), complex)
        for k in range(spp):
            n = (p - 1)*spp + k; tn, tn1 = n*dt, (n + 1)*dt
            Anew = lu.solve(B@A + 0.5*(np.cos(w*tn) + np.cos(w*tn1))*ff)
            acc += 0.5*(A + Anew)*np.exp(-1j*w*(tn + 0.5*dt))*dt
            A = Anew
        if p in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 45, 60):
            At = np.zeros(g.N, complex); At[fr] = (2/T)*acc
            c = ec.certify(g, nu, sg, J, w, I_COIL, At=At, KMf=(K, M, fv))
            inside = c['L_lo'] <= cref['L_lo'] and cref['L_up'] <= c['L_up'] or (c['L_lo'] <= cref['L'] <= c['L_up'])
            rows.append((spp, p, c['L'], c['L_lo'], c['L_up'], c['R'], c['dZ'], c['eta']))
            print(f"steps/period={spp:3d} period {p:4d}: L in [{c['L_lo']*1e6:9.5f},{c['L_up']*1e6:9.5f}]  "
                  f"half-width {100*(c['L_up']-c['L_lo'])/2/cref['L']:8.3f} %  contains ref: {inside}", flush=True)
    print(f"  ({time.time()-t0:.0f} s)")
np.savez('tlm_anytime.npz', rows=np.array(rows), ref=[cref['L'], cref['L_lo'], cref['L_up'], cref['R'], cref['dZ']],
         gal=[cG['L'], cG['L_lo'], cG['L_up'], cG['R'], cG['dZ']])
