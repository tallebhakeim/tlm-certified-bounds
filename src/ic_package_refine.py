"""
Mesh-refinement study of the certified pad-to-substrate capacitance of the IC package.

Same physics and same two bounds as ic_package.py (conforming Q1-hex upper energy,
equilibrated RT0 face-flux Thomson lower energy), run on a sequence of refined
anisotropic Cartesian grids. Produces the mesh/cost table of the paper and checks
that the successive certified brackets are consistent (non-empty intersection).

GEOMETRY-CONFORMING GRIDS. The package is a Manhattan assembly of axis-aligned boxes,
so its faces define a set of breakpoints in x, y and z. Every grid used here is built
by subdividing the intervals BETWEEN consecutive breakpoints, so that no cell ever
straddles a material interface and the voxel model reproduces the CAD geometry EXACTLY
at every refinement level. A naive uniform grid does not have this property: the die
pad (4.000 mm wide, 0.020 mm thick, 1.075 mm above the ground plane) is then snapped
to whatever cells happen to contain it, its width oscillates by several percent from
one grid to the next, and the resulting "guaranteed" brackets are mutually disjoint,
because each of them certifies a DIFFERENT capacitor. Prager-Synge bounds the
discretisation error of a given geometry; it does not cover geometric error.

Run:  python3 ic_package_refine.py
"""
import time
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh

np.seterr(over='ignore', divide='ignore', invalid='ignore')
EPS0 = 8.8541878128e-12
EPSR_MOULD = 4.0
STL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'model', 'ic_package.stl')

scene = trimesh.load(STL)
B = list(scene.geometry.values())
pad_bb = B[2].bounds.copy()
sub_bb = B[4].bounds.copy()
mould_bb = B[0].bounds.copy()
gap = pad_bb[0, 2] - sub_bb[1, 2]
A_pad = (pad_bb[1, 0]-pad_bb[0, 0])*(pad_bb[1, 1]-pad_bb[0, 1])
C_pp = EPS0*EPSR_MOULD*A_pad*1e-6/(gap*1e-3)


def _kref_components():
    g = 1/np.sqrt(3)
    gp = [(a, b, c) for a in (-g, g) for b in (-g, g) for c in (-g, g)]
    nc = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                   (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], float)
    Kxx = np.zeros((8, 8)); Kyy = np.zeros((8, 8)); Kzz = np.zeros((8, 8))
    for (xi, et, ze) in gp:
        Gx = np.array([0.125*a*(1+b*et)*(1+c*ze) for a, b, c in nc])
        Gy = np.array([0.125*b*(1+a*xi)*(1+c*ze) for a, b, c in nc])
        Gz = np.array([0.125*c*(1+a*xi)*(1+b*et) for a, b, c in nc])
        Kxx += np.outer(Gx, Gx); Kyy += np.outer(Gy, Gy); Kzz += np.outer(Gz, Gz)
    return Kxx, Kyy, Kzz


KXX, KYY, KZZ = _kref_components()
LOC = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]


# ---- geometry breakpoints: every material face of the Manhattan assembly ----
XBP = sorted({sub_bb[0, 0], sub_bb[1, 0], mould_bb[0, 0], mould_bb[1, 0],
              pad_bb[0, 0], pad_bb[1, 0]})
YBP = sorted({sub_bb[0, 1], sub_bb[1, 1], mould_bb[0, 1], mould_bb[1, 1],
              pad_bb[0, 1], pad_bb[1, 1]})
ZBP = sorted({sub_bb[0, 2]-0.05, sub_bb[0, 2], sub_bb[1, 2],
              pad_bb[0, 2], pad_bb[1, 2], pad_bb[1, 2]+0.6})


def conforming_nodes(bp, h):
    """Node coordinates subdividing each interval of `bp` into cells of size <= h."""
    out = [bp[0]]
    for a, b in zip(bp[:-1], bp[1:]):
        n = max(1, int(np.ceil((b-a)/h - 1e-9)))
        out.extend(a+(b-a)*np.arange(1, n+1)/n)
    return np.array(out)


def uniform_nodes(bp, h):
    """Legacy grid: uniform subdivision of the whole span, ignoring the geometry."""
    n = max(1, int(round((bp[-1]-bp[0])/h)))
    return bp[0]+(bp[-1]-bp[0])*np.arange(n+1)/n


def solve(DX, DZ, verbose=True, return_fields=False, conforming=True):
    """Certified bracket [C_lo, C_up] on one anisotropic grid.

    conforming=True : cell faces aligned with every material interface (geometry exact).
    conforming=False: legacy uniform grid, kept to quantify the geometric error it causes.
    """
    nodes_of = conforming_nodes if conforming else uniform_nodes
    xn = nodes_of(XBP, DX); yn = nodes_of(YBP, DX); zn = nodes_of(ZBP, DZ)
    nx = len(xn)-1; ny = len(yn)-1; nz = len(zn)-1
    dxv = np.diff(xn); dyv = np.diff(yn); dzv = np.diff(zn)
    hx = float(dxv.max()); hy = float(dyv.max()); hz = float(dzv.max())
    x0, x1 = xn[0], xn[-1]; y0, y1 = yn[0], yn[-1]; z0, z1 = zn[0], zn[-1]
    xc = 0.5*(xn[:-1]+xn[1:]); yc = 0.5*(yn[:-1]+yn[1:]); zc = 0.5*(zn[:-1]+zn[1:])
    Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')

    def inbb(bb, X, Y, Z):
        return ((X >= bb[0, 0]) & (X <= bb[1, 0]) & (Y >= bb[0, 1]) & (Y <= bb[1, 1]) &
                (Z >= bb[0, 2]) & (Z <= bb[1, 2]))

    subcell = inbb(sub_bb, Xc, Yc, Zc)
    kpad = int(np.argmin(np.abs(zc-pad_bb[:, 2].mean())))
    if conforming:
        padcell = inbb(pad_bb, Xc, Yc, Zc)      # exact: no cell straddles a pad face
    else:                                       # legacy: snap the pad to one z-layer
        padcell = ((Xc >= pad_bb[0, 0]) & (Xc <= pad_bb[1, 0]) &
                   (Yc >= pad_bb[0, 1]) & (Yc <= pad_bb[1, 1]) &
                   (np.arange(nz)[None, None, :] == kpad))
    cond = subcell | padcell
    inmould = ((Xc >= mould_bb[0, 0]) & (Xc <= mould_bb[1, 0]) &
               (Yc >= mould_bb[0, 1]) & (Yc <= mould_bb[1, 1]))
    epsr = np.where(inmould, EPSR_MOULD, 1.0)

    DXc, DYc, DZc = np.meshgrid(dxv, dyv, dzv, indexing='ij')   # per-cell sizes
    npx, npy, npz_ = nx+1, ny+1, nz+1
    nid = lambda i, j, k: (k*npy+j)*npx+i
    N = npx*npy*npz_
    I, J, Kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    I = I.ravel(); J = J.ravel(); Kk = Kk.ravel(); ncell = nx*ny*nz
    ev = epsr[I, J, Kk]
    ax_c = DYc[I, J, Kk]*DZc[I, J, Kk]/(2*DXc[I, J, Kk])
    ay_c = DXc[I, J, Kk]*DZc[I, J, Kk]/(2*DYc[I, J, Kk])
    az_c = DXc[I, J, Kk]*DYc[I, J, Kk]/(2*DZc[I, J, Kk])
    nodes = np.stack([nid(I+dx, J+dy, Kk+dz) for dx, dy, dz in LOC], axis=1)
    rows = np.broadcast_to(nodes[:, :, None], (ncell, 8, 8)).ravel()
    cols = np.broadcast_to(nodes[:, None, :], (ncell, 8, 8)).ravel()
    vals = (ev[:, None, None]*(ax_c[:, None, None]*KXX[None] + ay_c[:, None, None]*KYY[None]
                               + az_c[:, None, None]*KZZ[None])).ravel()
    K = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

    nodeV = np.full(N, np.nan)
    for cellmask, V in [(subcell, 0.0), (padcell, 1.0)]:
        ii, jj, kk = np.nonzero(cellmask)
        for dx, dy, dz in LOC:
            nodeV[nid(ii+dx, jj+dy, kk+dz)] = V
    fi = np.where(~np.isnan(nodeV))[0]
    u = np.zeros(N); u[fi] = nodeV[fi]
    fr = np.setdiff1d(np.arange(N), fi)
    t0 = time.perf_counter()
    u[fr] = spla.spsolve(K[fr][:, fr].tocsc(), -K[fr][:, fi]@u[fi])
    t_up = time.perf_counter()-t0
    C_up = EPS0*float(u@(K@u))*1e-3
    ndof_up = len(fr)

    nxf = (nx+1)*ny*nz; nyf = nx*(ny+1)*nz; nzf = nx*ny*(nz+1)
    xid = lambda i, j, k: (k*ny+j)*(nx+1)+i
    yid = lambda i, j, k: (k*(ny+1)+j)*nx+i
    zid = lambda i, j, k: (k*ny+j)*nx+i
    OY = nxf; OZ = nxf+nyf; NF = nxf+nyf+nzf
    R = np.zeros(NF); pin = np.zeros(NF, bool)
    isc = lambda i, j, k: cond[i, j, k]

    def setface(f, cA, cB, eA, eB, dA, dB, area):
        """Series resistance of the two half-cells sharing the face (non-uniform grid)."""
        if cA and cB: pin[f] = True; R[f] = 1.0                      # inside a conductor
        elif cA:      R[f] = (dB/2)/(eB*area)                        # half-cell on the dielectric side
        elif cB:      R[f] = (dA/2)/(eA*area)
        else:         R[f] = (dA/2)/(eA*area) + (dB/2)/(eB*area)     # dielectric-dielectric

    for k in range(nz):
        for j in range(ny):
            area = dyv[j]*dzv[k]
            for i in range(nx+1):
                f = xid(i, j, k)
                if i == 0 or i == nx: pin[f] = True; R[f] = 1.0; continue
                setface(f, isc(i-1, j, k), isc(i, j, k), epsr[i-1, j, k], epsr[i, j, k],
                        dxv[i-1], dxv[i], area)
    for k in range(nz):
        for j in range(ny+1):
            for i in range(nx):
                f = OY+yid(i, j, k)
                if j == 0 or j == ny: pin[f] = True; R[f] = 1.0; continue
                setface(f, isc(i, j-1, k), isc(i, j, k), epsr[i, j-1, k], epsr[i, j, k],
                        dyv[j-1], dyv[j], dxv[i]*dzv[k])
    for k in range(nz+1):
        for j in range(ny):
            for i in range(nx):
                f = OZ+zid(i, j, k)
                if k == 0 or k == nz: pin[f] = True; R[f] = 1.0; continue
                setface(f, isc(i, j, k-1), isc(i, j, k), epsr[i, j, k-1], epsr[i, j, k],
                        dzv[k-1], dzv[k], dxv[i]*dyv[j])
    keep = ~pin; fidx = -np.ones(NF, int); fidx[keep] = np.arange(keep.sum()); nfk = int(keep.sum())
    cellfaces = lambda i, j, k: [(xid(i+1, j, k), 1), (xid(i, j, k), -1),
                                 (OY+yid(i, j+1, k), 1), (OY+yid(i, j, k), -1),
                                 (OZ+zid(i, j, k+1), 1), (OZ+zid(i, j, k), -1)]
    rr, cc, vv = [], [], []; brow = []; r = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if cond[i, j, k]: continue
                for f, s in cellfaces(i, j, k):
                    if keep[f]: rr.append(r); cc.append(fidx[f]); vv.append(s)
                brow.append(0.0); r += 1
    Q = 1.0; ii, jj, kk = np.nonzero(padcell)
    for i, j, k in zip(ii, jj, kk):
        for f, s in cellfaces(i, j, k):
            if keep[f]: rr.append(r); cc.append(fidx[f]); vv.append(s)
    brow.append(Q); r += 1
    Bm = sp.csr_matrix((vv, (rr, cc)), shape=(r, nfk)); Adiag = sp.diags(R[keep])
    KK = sp.bmat([[Adiag, Bm.T], [Bm, None]], format='csc')
    t0 = time.perf_counter()
    sol = spla.spsolve(KK, np.concatenate([np.zeros(nfk), np.array(brow)]))
    t_lo = time.perf_counter()-t0
    q = sol[:nfk]
    C_lo = EPS0*(Q*Q/float(q@(Adiag@q)))*1e-3

    # geometry actually represented by the voxel model (must equal the CAD values)
    ii, jj, kk = np.nonzero(padcell)
    pad_w = float(xn[ii.max()+1]-xn[ii.min()]); pad_t = float(zn[kk.max()+1]-zn[kk.min()])
    ks = np.nonzero(subcell)[2]
    gap_eff = float(zn[kk.min()]-zn[ks.max()+1])
    out = dict(DX=DX, DZ=DZ, conforming=conforming, nx=nx, ny=ny, nz=nz, ncell=nx*ny*nz, ndof_up=ndof_up,
               ndof_lo=nfk+r, C_lo=C_lo, C_up=C_up, t_up=t_up, t_lo=t_lo,
               hx=hx, hy=hy, hz=hz, npad=int(padcell.sum()),
               pad_w=pad_w, pad_t=pad_t, gap_eff=gap_eff)
    if return_fields:
        out.update(u=u.reshape(npz_, npy, npx), xn=xn, yn=yn, zn=zn, xc=xc, yc=yc, zc=zc,
                   epsr=epsr, subcell=subcell, padcell=padcell, kpad=kpad, C_pp=C_pp)
    if verbose:
        hw = (C_up-C_lo)/(C_up+C_lo)*100
        print(f"  hmax=({hx:.3f},{hy:.3f},{hz:.4f}) mm  {nx}x{ny}x{nz}={out['ncell']:>8d} cells  "
              f"ndof(up/lo)={ndof_up:>7d}/{out['ndof_lo']:>8d}  "
              f"C in [{C_lo*1e15:8.2f}, {C_up*1e15:8.2f}] fF  hw {hw:5.2f}%  "
              f"| pad {pad_w:.3f}x{pad_t:.3f} mm, gap {gap_eff:.4f} mm  "
              f"cpu={t_up+t_lo:7.2f} s", flush=True)
    return out


if __name__ == '__main__':
    print(f"pad-substrate gap = {gap:.3f} mm ; A_pad = {A_pad:.2f} mm^2 ; "
          f"parallel-plate (no fringing) C = {C_pp*1e15:.1f} fF\n")
    grids = [(1.00, 0.30), (0.70, 0.20), (0.50, 0.14), (0.35, 0.10), (0.25, 0.07)]
    res = []
    for DX, DZ in grids:
        res.append(solve(DX, DZ))
    lo = np.array([r['C_lo'] for r in res]); up = np.array([r['C_up'] for r in res])
    print("\nGEOMETRY EXACTNESS (voxel model vs CAD)")
    for n, r in enumerate(res, 1):
        print(f"  grid {n}: pad width {r['pad_w']:.6f} mm (CAD 4.000000), "
              f"thickness {r['pad_t']:.6f} mm (CAD 0.020000), "
              f"gap {r['gap_eff']:.6f} mm (CAD {gap:.6f})")
    print("\nCONSISTENCY OF THE SUCCESSIVE BRACKETS")
    print(f"  max lower bound = {lo.max()*1e15:.2f} fF (grid {int(lo.argmax())+1})")
    print(f"  min upper bound = {up.min()*1e15:.2f} fF (grid {int(up.argmin())+1})")
    print(f"  intersection    = [{lo.max()*1e15:.2f}, {up.min()*1e15:.2f}] fF -> "
          f"{'NON-EMPTY (consistent)' if lo.max() < up.min() else 'EMPTY (INCONSISTENT)'}")
    print(f"  lower monotone increasing: {bool(np.all(np.diff(lo) >= 0))} ; "
          f"upper monotone decreasing: {bool(np.all(np.diff(up) <= 0))}")
    np.savez('/tmp/ic_pkg_refine.npz', **{k: np.array([r[k] for r in res]) for k in res[0]})
    print("\nsaved /tmp/ic_pkg_refine.npz")

    print("\n--- CONTRE-EPREUVE : memes physiques, grilles UNIFORMES non conformes ---")
    ugrids = [(0.80, 0.20), (0.53, 0.135), (0.40, 0.10), (0.31, 0.077), (0.25, 0.0625)]
    ures = [solve(DX, DZ, conforming=False) for DX, DZ in ugrids]
    ulo = np.array([r['C_lo'] for r in ures]); uup = np.array([r['C_up'] for r in ures])
    print(f"  intersection = [{ulo.max()*1e15:.2f}, {uup.min()*1e15:.2f}] fF -> "
          f"{'NON-EMPTY' if ulo.max() < uup.min() else 'EMPTY (brackets disjoints)'}")
    np.savez('/tmp/ic_pkg_uniform.npz', **{k: np.array([r[k] for r in ures]) for k in ures[0]})
    print("saved /tmp/ic_pkg_uniform.npz")
