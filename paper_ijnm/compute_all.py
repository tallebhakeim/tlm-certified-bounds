"""
Every number of the IJNM manuscript (article 10, TLM certified bounds), recomputed
from scratch and written to results_ijnm.json + results_ijnm.npz.

Self-contained for the two-dimensional parts; the three-dimensional parts import the
reference implementation (tlm-certified-bounds/src) and the benchmark against the dual
discrete-geometric solver imports the DGM engine (optional, skipped if absent).

Run:  python3 compute_all.py          (~35 min on one core, dominated by the package)
"""
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src')
sys.path.insert(0, SRC)
np.seterr(over='ignore', divide='ignore', invalid='ignore')
OUT = {}
NPZ = {}
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f} s] {msg}", flush=True)


# ============================================================ 2-D machinery
KREF = (1.0/6.0)*np.array([[4., -1., -2., -1.], [-1., 4., -1., -2.],
                           [-2., -1., 4., -1.], [-1., -2., -1., 4.]])


def assemble(n, coeff):
    npn = n+1
    nid = lambda i, j: j*npn+i
    rows, cols, vals = [], [], []
    for cj in range(n):
        yc = (cj+0.5)/n
        for ci in range(n):
            xc = (ci+0.5)/n
            cc = coeff(xc, yc)
            loc = [nid(ci, cj), nid(ci+1, cj), nid(ci+1, cj+1), nid(ci, cj+1)]
            for a in range(4):
                for b in range(4):
                    rows.append(loc[a]); cols.append(loc[b]); vals.append(cc*KREF[a, b])
    return sp.csr_matrix((vals, (rows, cols)), shape=(npn*npn, npn*npn)), nid


def solve(K, npn, nid, orient):
    N = npn*npn; u = np.zeros(N); fx = {}
    for k in range(npn):
        if orient == 'LR': fx[nid(0, k)] = 0.0; fx[nid(npn-1, k)] = 1.0
        else:              fx[nid(k, 0)] = 0.0; fx[nid(k, npn-1)] = 1.0
    fi = np.array(sorted(fx)); u[fi] = [fx[k] for k in fi]
    fr = np.setdiff1d(np.arange(N), fi)
    u[fr] = spla.spsolve(K[fr][:, fr].tocsc(), -K[fr][:, fi]@u[fi])
    return u


def bounds2d(n, coeff):
    """Certified bracket on the n x n grid: potential network (upper), stream-function
    network (lower). Returns (G_lo, G_up, dof)."""
    Ks, nid = assemble(n, coeff)
    phi = solve(Ks, n+1, nid, 'LR'); Gup = float(phi@(Ks@phi))
    Kr, nid = assemble(n, lambda x, y: 1.0/coeff(x, y))
    psi = solve(Kr, n+1, nid, 'BT'); Glo = 1.0/float(psi@(Kr@psi))
    return Glo, Gup, (n+1)**2


def field_square(c):
    return lambda x, y: c if (0.25 <= x <= 0.75 and 0.25 <= y <= 0.75) else 1.0


def field_checker(c, k=4):
    return lambda x, y: c if ((int(x*k)+int(y*k)) % 2 == 0) else 1.0


def field_L(c):
    return lambda x, y: 1.0 if (x >= 0.5 and y >= 0.5) else c


def field_smooth(x, y):
    return 1.0+9.0*np.exp(-((x-0.5)**2+(y-0.5)**2)/(2*0.15**2))


def rate(hs, g):
    return float(np.polyfit(np.log(hs), np.log(g), 1)[0])


# ============================================================ 1. 2-D validation
log("2-D validation, square inclusion sigma=10")
sq = field_square(10.0)
Glo_ref, Gup_ref, _ = bounds2d(512, sq)
ns = [8, 16, 32, 64, 128, 256]
rows = []
for n in ns:
    lo, up, dof = bounds2d(n, sq)
    rows.append(dict(n=n, h=1.0/n, dof=dof, lo=lo, up=up, gap=up-lo, mid=0.5*(lo+up),
                     contains_ref=bool(lo <= Gup_ref and up >= Glo_ref)))
OUT['val2d'] = dict(ref_lo=Glo_ref, ref_up=Gup_ref, ref_width=Gup_ref-Glo_ref, rows=rows,
                    rate=rate([r['h'] for r in rows], [r['gap'] for r in rows]))
log(f"  reference [{Glo_ref:.6f}, {Gup_ref:.6f}], rate {OUT['val2d']['rate']:.2f}")

# ============================================================ 2. effectivity (contrast 100)
log("effectivity index, square contrast 100")
f100 = field_square(100.0)
lo5, up5, _ = bounds2d(512, f100); Gt = 0.5*(lo5+up5)
eff = []
for n in [16, 32, 64, 128, 256]:
    lo, up, _ = bounds2d(n, f100)
    e_b = max(abs(up-Gt), abs(lo-Gt)); e_m = abs(0.5*(up+lo)-Gt)
    eff.append(dict(n=n, lo=lo, up=up, halfwidth=0.5*(up-lo), err_bound=e_b, err_mid=e_m,
                    eta=(up-lo)/e_b, eta_half=0.5*(up-lo)/e_b, mid_gain=e_b/e_m))
OUT['effectivity'] = dict(ref_lo=lo5, ref_up=up5, rows=eff)

# ============================================================ 3. contrast sweep
log("contrast sweep n=128")
cs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
con = []
for c in cs:
    lo, up, _ = bounds2d(128, field_square(float(c)))
    con.append(dict(contrast=c, lo=lo, up=up, relwidth=(up-lo)/(0.5*(up+lo))))
OUT['contrast'] = con

# ============================================================ 4. geometry sweep
log("geometry sweep, contrast 100")
geo = {}
for name, f in [('square', field_square(100.)), ('checkerboard', field_checker(100.)),
                ('L', field_L(100.))]:
    rr = []
    for n in [16, 32, 64, 128, 256]:
        lo, up, _ = bounds2d(n, f)
        rr.append(dict(n=n, lo=lo, up=up, gap=up-lo, relhalf=0.5*(up-lo)/(0.5*(up+lo))))
    geo[name] = dict(rows=rr, rate=rate([1/r['n'] for r in rr], [r['gap'] for r in rr]))
OUT['geometry'] = geo

# ============================================================ 5. refinement-difference indicator
# What a practitioner does without a certificate: take the change of the conforming
# answer between two successive grids as the error indicator.
log("refinement-difference indicator vs certified bracket (square, contrast 10)")
ind = []
prev = None
for r in OUT['val2d']['rows']:
    d = None if prev is None else abs(r['up']-prev)
    true_err = abs(r['up']-0.5*(Glo_ref+Gup_ref))
    ind.append(dict(n=r['n'], up=r['up'], diff_prev=d, true_err_up=true_err,
                    cert_halfwidth=0.5*r['gap']))
    prev = r['up']
OUT['indicator'] = ind

# ============================================================ 6. anytime iteration (2-D)
log("anytime iteration n=32, 9000 steps (both networks)")


def graph(K):
    Kc = K.tocoo(); m = (Kc.row != Kc.col) & (Kc.data < 0)
    s, d, g = Kc.row[m], Kc.col[m], -Kc.data[m]
    key = {(a, b): e for e, (a, b) in enumerate(zip(s, d))}
    rev = np.array([key[(b, a)] for a, b in zip(s, d)])
    return s, d, g, rev


def tlm_history(n, coeff, orient, nit, alpha):
    K, nid = assemble(n, coeff); npn = n+1; N = npn*npn
    s, d, g, rev = graph(K)
    assert np.all(K.diagonal() > 0)
    Kc = K.tocoo(); off = Kc.data[Kc.row != Kc.col]
    assert np.all(off <= 1e-14), "stiffness graph is not an M-matrix"
    phi = np.zeros(N); is_dir = np.zeros(N, bool)
    for k in range(npn):
        if orient == 'LR':
            is_dir[nid(0, k)] = True; is_dir[nid(n, k)] = True; phi[nid(n, k)] = 1.0
        else:
            is_dir[nid(k, 0)] = True; is_dir[nid(k, n)] = True; phi[nid(k, n)] = 1.0
    Vstar = solve(K, npn, nid, orient)
    den = np.bincount(s, weights=g, minlength=N); inc = np.zeros(len(s))
    V = np.zeros(N); V[is_dir] = phi[is_dir]
    G = np.empty(nit); E = np.empty(nit); P = np.empty(nit)
    for it in range(nit):
        num = np.bincount(s, weights=g*inc, minlength=N)
        Vn = np.where(den > 0, 2.0*num/np.where(den > 0, den, 1.0), 0.0); Vn[is_dir] = phi[is_dir]
        refl = Vn[s]-inc
        inc = alpha*refl[rev]+(1.0-alpha)*refl
        V = Vn
        G[it] = float(V@(K@V))
        e = V-Vstar; E[it] = float(np.sqrt(max(e@(K@e), 0.0)))
        P[it] = float(np.sum(g*inc*inc))      # pulse energy
    res = float(np.linalg.norm((K@V)[~is_dir]))
    return G, E, P, res, float(Vstar@(K@Vstar))


NIT = 9000; ALPHA = 0.5
Gup_k, eU, PU, resU, GupInf = tlm_history(32, sq, 'LR', NIT, ALPHA)
Rup_k, eL, PL, resL, RupInf = tlm_history(32, lambda x, y: 1.0/sq(x, y), 'BT', NIT, ALPHA)
Glo_k = 1.0/Rup_k
Gt10 = 0.5*(Glo_ref+Gup_ref)
viol_up = int(np.sum(Gup_k < Glo_ref-1e-12)); viol_lo = int(np.sum(Glo_k > Gup_ref+1e-12))
half = 0.5*(Gup_k-Glo_k)
it_tgt = {}
for tgt in [1e-1, 1e-2, 1e-3]:
    k = int(np.argmax(half <= tgt)) if np.any(half <= tgt) else -1
    it_tgt[f"{tgt:.0e}"] = k
OUT['anytime'] = dict(n=32, nit=NIT, alpha=ALPHA, violations_up=viol_up, violations_lo=viol_lo,
                      mono_up=bool(np.all(np.diff(Gup_k) <= 1e-12)),
                      mono_lo=bool(np.all(np.diff(Glo_k) >= -1e-12)),
                      mono_errK_phi=bool(np.all(np.diff(eU) <= 1e-12)),
                      mono_errK_psi=bool(np.all(np.diff(eL) <= 1e-12)),
                      mono_pulse_phi=bool(np.all(np.diff(PU) <= 1e-12)),
                      mono_pulse_psi=bool(np.all(np.diff(PL) <= 1e-12)),
                      kcl_residual_phi=resU, kcl_residual_psi=resL,
                      final_up=float(Gup_k[-1]), final_lo=float(Glo_k[-1]),
                      steady_up=GupInf, steady_lo=1.0/RupInf,
                      iterations_to_halfwidth=it_tgt, first_up=float(Gup_k[0]), first_lo=float(Glo_k[0]))
NPZ.update(any_Gup=Gup_k, any_Glo=Glo_k, any_eU=eU, any_eL=eL, any_PU=PU, any_PL=PL)
log(f"  violations {viol_up}/{viol_lo}, monotone {OUT['anytime']['mono_up']}/{OUT['anytime']['mono_lo']}")

# fixed point independent of alpha, and spectral radius
log("fixed point vs alpha, spectral radius")
fp = []
for a in [0.3, 0.5, 0.7, 0.9, 0.97]:
    G_a, _, _, r_a, Ginf = tlm_history(32, sq, 'LR', 6000, a)
    fp.append(dict(alpha=a, G_final=float(G_a[-1]), G_steady=Ginf, kcl_residual=r_a))
OUT['alpha'] = fp

# ============================================================ 7. three dimensions
log("3-D cube and inclusion (import tlm3d)")
import tlm3d
cube = []
for n in [8, 12]:
    Gup, _, _, _ = tlm3d.primal_upper(n, lambda i, j, k: 1.0)
    Glo = tlm3d.rt0_lower(n, lambda i, j, k: 1.0)
    cube.append(dict(n=n, lo=Glo, up=Gup))


def incl(i, j, k, n):
    c = ((i+0.5)/n, (j+0.5)/n, (k+0.5)/n)
    return 10.0 if all(0.3 <= t <= 0.7 for t in c) else 1.0


def edge_seen(n):
    c = (np.arange(n)+0.5)/n
    return float(((c >= 0.3) & (c <= 0.7)).sum()/n)


ok3, bad3 = [], []
for n in [10, 20, 30]:
    Gup, _, _, _ = tlm3d.primal_upper(n, lambda i, j, k: incl(i, j, k, n))
    Glo = tlm3d.rt0_lower(n, lambda i, j, k: incl(i, j, k, n))
    ok3.append(dict(n=n, edge=edge_seen(n), lo=Glo, up=Gup, gap=Gup-Glo))
for n in [8, 12, 16]:
    Gup, _, _, _ = tlm3d.primal_upper(n, lambda i, j, k: incl(i, j, k, n))
    Glo = tlm3d.rt0_lower(n, lambda i, j, k: incl(i, j, k, n))
    bad3.append(dict(n=n, edge=edge_seen(n), lo=Glo, up=Gup))
OUT['cube3d'] = cube
OUT['incl3d'] = dict(conforming=ok3, rate=rate([1/r['n'] for r in ok3], [r['gap'] for r in ok3]),
                     nonconforming=bad3,
                     inter_ok=[max(r['lo'] for r in ok3), min(r['up'] for r in ok3)],
                     inter_bad=[max(r['lo'] for r in bad3), min(r['up'] for r in bad3)])
# anytime in 3-D (conforming member only)
G3 = tlm3d.tlm_upper_3d(10, lambda i, j, k: incl(i, j, k, 10), 3000, alpha=0.9)
Gup10 = ok3[0]['up']
OUT['anytime3d'] = dict(n=10, nit=3000, alpha=0.9, violations=int(np.sum(G3 < Gup10-1e-12)),
                        mono=bool(np.all(np.diff(G3) <= 1e-12)), final=float(G3[-1]), steady=Gup10)
NPZ['any3d_G'] = G3

# ============================================================ 8. DGM benchmark (optional)
log("benchmark vs dual DGM (optional)")
DGM = None
for cand in ['~/Documents/DGM-engine', '~/Documents/Dual_discret_method/dgm']:
    p = os.path.expanduser(cand)
    if os.path.isdir(os.path.join(p, 'dgm')):
        DGM = p; break
if DGM:
    sys.path.insert(0, DGM)
    try:
        from dgm import assemble_primal, energy_lower_2d
        from dgm.meshing import rectangular_domain

        def dgm_bounds(h, sfield):
            mesh, _, dom = rectangular_domain((0, 0, 1, 1), [], h=h, seed=1)
            cen = mesh.points[mesh.tris].mean(1)
            sig = np.array([sfield(x, y) for x, y in cen])
            P = mesh.points; bc = {}
            for i in dom:
                if abs(P[i, 0]) < 1e-7: bc[int(i)] = 0.0
                elif abs(P[i, 0]-1) < 1e-7: bc[int(i)] = 1.0
            K = assemble_primal(mesh, sig)
            v = np.zeros(mesh.np); fi = np.array(sorted(bc)); v[fi] = [bc[k] for k in fi]
            fr = np.setdiff1d(np.arange(mesh.np), fi)
            v[fr] = spla.spsolve(K[fr][:, fr].tocsc(), -K[fr][:, fi]@v[fi])
            area_in = float(np.sum(0.5*np.abs(np.cross(P[mesh.tris[:, 1]]-P[mesh.tris[:, 0]],
                                                       P[mesh.tris[:, 2]]-P[mesh.tris[:, 0]]))[sig > 5]))
            return float(v@(K@v)), 2.0*energy_lower_2d(mesh, sig, bc), int(mesh.np), area_in

        bench = {}
        for name, sf in [('square', sq), ('smooth', field_smooth)]:
            t0 = time.perf_counter(); tlo, tup, _ = bounds2d(512, sf); tt = time.perf_counter()-t0
            t0 = time.perf_counter(); dup, dlo, _, _ = dgm_bounds(1/80, sf); td = time.perf_counter()-t0
            A, B = [], []
            for n in [16, 32, 64, 128, 256]:
                t0 = time.perf_counter(); lo, up, dof = bounds2d(n, sf); dt = time.perf_counter()-t0
                A.append(dict(n=n, dof=dof, lo=lo, up=up, half=0.5*(up-lo), cpu=dt,
                              compat=bool(lo <= dup and up >= dlo)))
            for h in [1/14, 1/20, 1/28, 1/40, 1/56]:
                t0 = time.perf_counter(); up, lo, dof, ain = dgm_bounds(h, sf); dt = time.perf_counter()-t0
                B.append(dict(h=h, dof=dof, lo=lo, up=up, half=0.5*(up-lo), cpu=dt, area_in=ain,
                              compat=bool(lo <= tup and up >= tlo)))
            bench[name] = dict(tlm_ref=[tlo, tup], tlm_ref_cpu=tt, dgm_ref=[dlo, dup], dgm_ref_cpu=td,
                               inter=[max(tlo, dlo), min(tup, dup)], tlm=A, dgm=B)
        OUT['benchmark'] = bench
    except Exception as exc:
        OUT['benchmark'] = dict(error=str(exc))
        log(f"  DGM benchmark skipped: {exc}")
else:
    OUT['benchmark'] = dict(error='DGM engine not found')

with open(os.path.join(HERE, 'results_ijnm.json'), 'w') as f:
    json.dump(OUT, f, indent=1)
np.savez(os.path.join(HERE, 'results_ijnm.npz'), **NPZ)
log("intermediate results written")

# ============================================================ 9. IC package
log("IC package, conforming and uniform grids (import ic_package_refine)")
import ic_package_refine as icp
OUT['package'] = dict(gap_mm=float(icp.gap), A_pad_mm2=float(icp.A_pad), C_pp_fF=float(icp.C_pp*1e15),
                      pad_w_cad=float(icp.pad_bb[1, 0]-icp.pad_bb[0, 0]),
                      pad_t_cad=float(icp.pad_bb[1, 2]-icp.pad_bb[0, 2]),
                      sub_xy=[float(icp.sub_bb[1, 0]-icp.sub_bb[0, 0]), float(icp.sub_bb[1, 1]-icp.sub_bb[0, 1])],
                      mould_xy=[float(icp.mould_bb[1, 0]-icp.mould_bb[0, 0]), float(icp.mould_bb[1, 1]-icp.mould_bb[0, 1])],
                      epsr=icp.EPSR_MOULD)
conf = [icp.solve(DX, DZ) for DX, DZ in [(1.00, 0.30), (0.70, 0.20), (0.50, 0.14), (0.35, 0.10), (0.25, 0.07)]]
unif = [icp.solve(DX, DZ, conforming=False) for DX, DZ in
        [(0.80, 0.20), (0.53, 0.135), (0.40, 0.10), (0.31, 0.077), (0.25, 0.0625)]]


def strip(r):
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer, bool)) else v)
            for k, v in r.items() if k not in ('u', 'xn', 'yn', 'zn', 'xc', 'yc', 'zc', 'epsr',
                                                'subcell', 'padcell')}


OUT['package']['conforming'] = [strip(r) for r in conf]
OUT['package']['uniform'] = [strip(r) for r in unif]
lo = np.array([r['C_lo'] for r in conf]); up = np.array([r['C_up'] for r in conf])
hs = np.array([r['hx'] for r in conf])
OUT['package']['rate'] = rate(hs, up-lo)
OUT['package']['inter_conf'] = [float(lo.max()*1e15), float(up.min()*1e15)]
ulo = np.array([r['C_lo'] for r in unif]); uup = np.array([r['C_up'] for r in unif])
OUT['package']['inter_unif'] = [float(ulo.max()*1e15), float(uup.min()*1e15)]
cpu = np.array([r['t_up']+r['t_lo'] for r in conf]); nc = np.array([r['ncell'] for r in conf])
OUT['package']['cpu_exponent'] = float(np.polyfit(np.log(nc[1:]), np.log(cpu[1:]), 1)[0])

with open(os.path.join(HERE, 'results_ijnm.json'), 'w') as f:
    json.dump(OUT, f, indent=1)
log("DONE: results_ijnm.json")
