# Eddy-current impedance bounds (release 1.1)

Code for the paper *Guaranteed Two-Sided Bounds on Eddy-Current Coil Impedance for Any
Approximate Field, With a Passive TLM Solver and Application to Nondestructive Testing and
Induction Tomography* (H. Talleb, submitted to IEEE Transactions on Magnetics).

`eddy_certified2d.py` implements the 2D time-harmonic A_z problem on geometry-conforming
tensor grids (Q1 elements), the lowest-order Raviart-Thomas equilibrated flux, the estimator
eta, the impedance bound |Z - Z~| <= w c eta^2 / I^2 with c = (1 + sqrt 2)/2 (valid for any
approximate field, with the corrected output 2 l(A~) - b(A~, A~)), and certified mutual
impedances (polarisation / Cauchy-Schwarz).

| Script | Output | Paper |
|---|---|---|
| `verify_certificate.py` | static limit, frequency x grid sweep, perturbed fields | Sec. V |
| `verify_mutual.py` | exact two-coil mutual impedance | Sec. V |
| `fig_verification.py`, `fig2_plot.py` | `verification.npz`, verification figure | Sec. V |
| `ndt_study.py` | `ndt_results.npz` (refinement, impedance plane) | Sec. VI |
| `ndt_detect_fine.py` | `ndt_detect_fine.npz` (guaranteed wall-loss thresholds) | Sec. VI |
| `tlm_anytime.py` | `tlm_anytime.npz` (certificate along the passive TLM march) | Sec. IV |
| `mit_study.py` | `mit_results.npz` (8-coil array) | Sec. VII |
| `fig_setup.py`, `figs_apps.py` | figures | |

`exact_slab.py` gives the closed-form impedance of the layered benchmark and `ndt_geometry.py`
the probe model. Requirements: numpy, scipy, matplotlib. When several scripts run in parallel,
set `OMP_NUM_THREADS=1`.
