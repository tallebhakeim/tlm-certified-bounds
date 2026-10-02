"""Guaranteed detection threshold of a far-side wall loss: fine sweep (2.5 % steps of T),
four grids, three frequencies. Output ndt_detect_fine.npz: rows
(f, hf, d, |Zd - Z0|, rho_d, rho_0, certified)."""
import numpy as np
from ndt_geometry import *
LOSS = np.arange(1, 25)*0.025*T
rows = []
for f in [1e3, 5e3, 20e3]:
    w = 2*np.pi*f
    for hf in [0.4e-3, 0.2e-3, 0.1e-3, 0.05e-3]:
        c0, _ = certify(hf, f); Z0 = c0['R'] + 1j*w*c0['L']; thr = None
        for d in LOSS:
            cd, _ = certify(hf, f, loss=d); Zd = cd['R'] + 1j*w*cd['L']
            ok = abs(Zd - Z0) > cd['dZ'] + c0['dZ']
            rows.append((f, hf, d, abs(Zd - Z0), cd['dZ'], c0['dZ'], ok))
            if ok and thr is None: thr = d
        print(f"f={f/1e3:4.0f} kHz  h={hf*1e3:.2f} mm: guaranteed detection from {100*thr/T if thr else float('nan'):.1f} % of T", flush=True)
np.savez('ndt_detect_fine.npz', rows=np.array(rows, float))
