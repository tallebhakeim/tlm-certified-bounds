"""NDT study: refinement table, certified detection threshold of far-side wall loss vs mesh and
frequency, impedance-plane trajectories (lift-off and wall loss) with certified discs.
Results cached in ndt_results.npz for the figures and tables."""
import time, numpy as np
from ndt_geometry import *

out = {}
# ---- 1. refinement study, sound plate, three frequencies ----
HF = [0.4e-3, 0.2e-3, 0.1e-3, 0.05e-3]; FREQ = [1e3, 5e3, 20e3]
w_air = None
rows = []
for f in FREQ:
    for hf in HF:
        t0 = time.time(); c, g = certify(hf, f); dt = time.time() - t0
        rows.append((f, hf, g.N, c['L'], c['L_lo'], c['L_up'], c['R'], c['R_lo'], c['R_up'], c['eta_flux'], c['osc'], c['eq_res'], dt))
        print(f"f={f/1e3:5.1f} kHz hf={hf*1e3:.3f} mm  N={g.N:6d}  L in [{c['L_lo']*1e6:.5f},{c['L_up']*1e6:.5f}] uH/m "
              f"R in [{c['R_lo']*1e3:.4f},{c['R_up']*1e3:.4f}] mOhm/m  osc/eta={c['osc']/c['eta_flux']:.1e}  eq={c['eq_res']:.0e}  {dt:.1f}s", flush=True)
out['refine'] = np.array(rows)
# air reference (no plate) for normalisation X0 = w L0
for f in FREQ:
    c, g = certify(0.1e-3, f, sig=0.0); out[f'L0_{int(f)}'] = c['L']
print("L0 (air) =", out['L0_1000'])

# ---- 2. far-side wall loss sweep: certified detection per mesh and frequency ----
LOSS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6])*T
det = []
for f in FREQ:
    w = 2*np.pi*f
    for hf in HF[:3]:
        c0, _ = certify(hf, f)
        for d in LOSS:
            cd, _ = certify(hf, f, loss=d)
            Zd = cd['R'] + 1j*w*cd['L']; Z0 = c0['R'] + 1j*w*c0['L']
            sep_Z = abs(Zd - Z0) > cd['dZ'] + c0['dZ']
            sep_L = (cd['L_lo'] > c0['L_up']) or (cd['L_up'] < c0['L_lo'])
            det.append((f, hf, d, Zd.real, Zd.imag, Z0.real, Z0.imag, cd['dZ'], c0['dZ'], cd['L_lo'], cd['L_up'], c0['L_lo'], c0['L_up'], sep_Z, sep_L))
        dd = [r for r in det if r[0] == f and r[1] == hf]
        th = [r[2] for r in dd if r[13]]
        print(f"f={f/1e3:5.1f} kHz hf={hf*1e3:.2f} mm: certified min. wall loss (Z disc) = {100*min(th)/T if th else float('nan'):.0f} %", flush=True)
out['det'] = np.array(det, dtype=float)

# ---- 3. impedance plane at 5 kHz, hf = 0.1 mm: lift-off and wall-loss trajectories ----
f = 5e3; w = 2*np.pi*f; hf = 0.1e-3; traj = []
for lo in [0.25e-3, 0.5e-3, 0.75e-3, 1.0e-3, 1.5e-3]:
    c, _ = certify(hf, f, lift=lo); traj.append((0, lo, c['R'], w*c['L'], c['dZ'], w*(c['L_up'] - c['L'])))
for d in [0.1*T, 0.2*T, 0.3*T, 0.4*T, 0.5*T]:
    c, _ = certify(hf, f, loss=d); traj.append((1, d, c['R'], w*c['L'], c['dZ'], w*(c['L_up'] - c['L'])))
out['traj'] = np.array(traj)
np.savez('ndt_results.npz', **out)
print("saved ndt_results.npz")
