"""Guaranteed two-sided bounds on the eddy-current coil impedance (2D, A_z formulation).

Model problem (time-harmonic, magneto-quasistatic, out-of-plane vector potential A = A_z):

    -div(nu grad A) + j w sigma A = J      in Omega = [x0,x1] x [y0,y1],
    A = 0 on the sides flagged 'D',  nu dA/dn = 0 on the sides flagged 'N'.

Coil impedance per unit depth (current I carried by the positive part of J):

    Z = j w l(A) / I^2 ,   l(A) = int J A ,   L = Re l(A)/I^2 ,   R = -w Im l(A)/I^2 .

Certificate (see Talleb, TMAG resubmission, Theorem 1). For ANY approximation At in H^1_0
(converged or not) and ANY flux q in H(div) with -div q = Pi0(J - j w sigma At) cell by cell:

    eta = || nu^{-1/2} (q - nu grad At) || + osc ,
    osc^2 = sum_K (hmax_K/pi)^2 nu_K^{-1} || w sigma (At - mean_K At) ||_K^2 ,

and with the corrected output lt = 2 l(At) - b(At, At) (b = bilinear, NOT sesquilinear form):

    l(A) - lt = b(e, e) ,   |b(e, e)| <= |||e|||^2 <= (1 + sqrt 2)/2 * eta^2 ,   e = A - At .

Hence |Z - Z_t| <= w c eta^2 / I^2 with c = (1+sqrt2)/2, and L, R lie in intervals of
half-width c eta^2/I^2 and w c eta^2/I^2 around L_t, R_t. (Re b(e,e) contains the cross term
-2w(sigma e_r, e_i), so the constant c, not 1, is needed for L as well.) For a Galerkin
solution lt = l(At).

Discretisation: Q1 nodal elements on a tensor-product (possibly non-uniform, geometry-
conforming) grid; lowest-order Raviart-Thomas RT[0] fluxes on the same rectangles; all
norms integrated exactly (2x2 Gauss is exact for the degree-2 integrands involved).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MU0 = 4e-7*np.pi
S1 = np.array([[1., -1.], [-1., 1.]])
M1 = np.array([[2., 1.], [1., 2.]])/6.0
GP = np.array([0.5 - 0.5/np.sqrt(3), 0.5 + 0.5/np.sqrt(3)])   # 2-pt Gauss on [0,1], weights 1/2


class Grid:
    """Tensor grid with node coordinates xs (nx+1), ys (ny+1); cells (j, i)."""

    def __init__(self, xs, ys, bc="DDDD"):
        self.xs = np.asarray(xs, float); self.ys = np.asarray(ys, float)
        self.nx = len(xs) - 1; self.ny = len(ys) - 1
        self.hx = np.diff(self.xs); self.hy = np.diff(self.ys)
        self.bc = bc                      # sides in order: left, right, bottom, top
        self.N = (self.nx + 1)*(self.ny + 1)
        self.xc = 0.5*(self.xs[:-1] + self.xs[1:]); self.yc = 0.5*(self.ys[:-1] + self.ys[1:])
        # face numbering: x-faces (normal x) f = j*(nx+1)+i ; y-faces offset OY, f = OY + j*nx + i
        self.OY = (self.nx + 1)*self.ny; self.NF = self.OY + self.nx*(self.ny + 1)

    def nid(self, i, j):
        return j*(self.nx + 1) + i

    def cell_nodes(self):
        """(ny, nx, 4) node ids in order (i,j), (i+1,j), (i,j+1), (i+1,j+1)."""
        I, J = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        n00 = self.nid(I, J); return np.stack([n00, n00 + 1, n00 + self.nx + 1, n00 + self.nx + 2], -1)

    def dirichlet_nodes(self):
        out = set(); nx, ny = self.nx, self.ny
        if self.bc[0] == 'D': out |= {self.nid(0, j) for j in range(ny + 1)}
        if self.bc[1] == 'D': out |= {self.nid(nx, j) for j in range(ny + 1)}
        if self.bc[2] == 'D': out |= {self.nid(i, 0) for i in range(nx + 1)}
        if self.bc[3] == 'D': out |= {self.nid(i, ny) for i in range(nx + 1)}
        return np.array(sorted(out), int)


def conforming_axis(breaks, hmax):
    """Nodes that contain every break point and subdivide each interval with spacing <= hmax."""
    breaks = np.unique(np.asarray(breaks, float)); pts = [breaks[0]]
    for a, b in zip(breaks[:-1], breaks[1:]):
        m = max(1, int(np.ceil((b - a)/hmax - 1e-9))); pts += list(a + (b - a)*np.arange(1, m + 1)/m)
    return np.array(pts)


def assemble(g, nu, sigma, J):
    """Q1 stiffness K (weight nu), consistent mass M (weight sigma), load f = int J phi."""
    cn = g.cell_nodes().reshape(-1, 4)
    hx = np.broadcast_to(g.hx[None, :], (g.ny, g.nx)).ravel(); hy = np.broadcast_to(g.hy[:, None], (g.ny, g.nx)).ravel()
    nu = np.asarray(nu).ravel(); sg = np.asarray(sigma).ravel(); Jc = np.asarray(J).ravel()
    # local 4x4 in node order (i,j),(i+1,j),(i,j+1),(i+1,j+1) = kron(y-factor, x-factor)
    Sx = np.kron(M1, S1); Sy = np.kron(S1, M1); Mm = np.kron(M1, M1)
    ke = nu[:, None, None]*((hy/hx)[:, None, None]*Sx[None] + (hx/hy)[:, None, None]*Sy[None])
    me = (sg*hx*hy)[:, None, None]*Mm[None]
    rows = np.repeat(cn, 4, axis=1).ravel(); cols = np.tile(cn, (1, 4)).ravel()
    K = sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(g.N, g.N))
    M = sp.csr_matrix((me.ravel(), (rows, cols)), shape=(g.N, g.N))
    f = np.zeros(g.N); np.add.at(f, cn.ravel(), np.repeat(Jc*hx*hy/4.0, 4))
    return K, M, f


def solve(g, K, M, f, w):
    """Galerkin solution of (K + j w M) A = f with the Dirichlet sides of g."""
    D = g.dirichlet_nodes(); fr = np.setdiff1d(np.arange(g.N), D)
    A = np.zeros(g.N, complex)
    A[fr] = spla.spsolve((K + 1j*w*M)[fr][:, fr].tocsc(), f[fr].astype(complex))
    return A


def _grads_at_gauss(g, A):
    """Cell-wise values of A, dA/dx, dA/dy at the 2x2 Gauss points: arrays (ny, nx, 2, 2) [eta, xi]."""
    cn = g.cell_nodes(); a00, a10, a01, a11 = (A[cn[..., k]] for k in range(4))
    X, E = GP[None, :], GP[:, None]                    # xi along x, eta along y
    val = (a00[..., None, None]*(1 - X)*(1 - E) + a10[..., None, None]*X*(1 - E)
           + a01[..., None, None]*(1 - X)*E + a11[..., None, None]*X*E)
    dx = ((a10 - a00)[..., None, None]*(1 - E) + (a11 - a01)[..., None, None]*E)/g.hx[None, :, None, None]
    dy = ((a01 - a00)[..., None, None]*(1 - X) + (a11 - a10)[..., None, None]*X)/g.hy[:, None, None, None]
    return val, dx, dy


def equilibrated_flux(g, nu, sigma, J, w, At):
    """RT[0] flux q minimising ||nu^{-1/2}(q - nu grad At)|| s.t. -div q = Pi0(J - j w sigma At).

    Returns face fluxes F (integrated normal flux per face) and the equilibration residual."""
    nx, ny, OY, NF = g.nx, g.ny, g.OY, g.NF
    nu = np.asarray(nu); sigma = np.asarray(sigma); J = np.asarray(J)
    HX, HY = np.meshgrid(g.hx, g.hy)
    cn = g.cell_nodes(); a00, a10, a01, a11 = (At[cn[..., k]] for k in range(4))
    Abar = 0.25*(a00 + a10 + a01 + a11)
    rhs_c = -(J - 1j*w*sigma*Abar)*HX*HY               # F_R - F_L + F_T - F_B = -int_K g
    I, Jj = np.meshgrid(np.arange(nx), np.arange(ny))
    fL = Jj*(nx + 1) + I; fR = fL + 1; fB = OY + Jj*nx + I; fT = fB + nx
    cid = Jj*nx + I
    # RT0 mass (weight 1/nu) and projection b_f = int grad At . psi_f
    cxx = (HX/HY)/nu; cyy = (HY/HX)/nu
    r, c, v = [], [], []
    for (p, q, wgt) in ((fL, fL, 1/3), (fR, fR, 1/3), (fL, fR, 1/6), (fR, fL, 1/6)):
        r.append(p.ravel()); c.append(q.ravel()); v.append((wgt*cxx).ravel())
    for (p, q, wgt) in ((fB, fB, 1/3), (fT, fT, 1/3), (fB, fT, 1/6), (fT, fB, 1/6)):
        r.append(p.ravel()); c.append(q.ravel()); v.append((wgt*cyy).ravel())
    Mq = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(NF, NF))
    bx = 0.25*(a10 - a00 + a11 - a01); by = 0.25*(a01 - a00 + a11 - a10)
    b = np.zeros(NF, complex); np.add.at(b, fL.ravel(), bx.ravel()); np.add.at(b, fR.ravel(), bx.ravel())
    np.add.at(b, fB.ravel(), by.ravel()); np.add.at(b, fT.ravel(), by.ravel())
    D = sp.csr_matrix((np.concatenate([np.ones(nx*ny), -np.ones(nx*ny), np.ones(nx*ny), -np.ones(nx*ny)]),
                       (np.concatenate([cid.ravel()]*4), np.concatenate([fR.ravel(), fL.ravel(), fT.ravel(), fB.ravel()]))),
                      shape=(nx*ny, NF))
    # Neumann sides: q.n = 0 imposed by removing those faces
    pinned = []
    if g.bc[0] == 'N': pinned += [j*(nx + 1) for j in range(ny)]
    if g.bc[1] == 'N': pinned += [j*(nx + 1) + nx for j in range(ny)]
    if g.bc[2] == 'N': pinned += [OY + i for i in range(nx)]
    if g.bc[3] == 'N': pinned += [OY + ny*nx + i for i in range(nx)]
    keep = np.setdiff1d(np.arange(NF), np.array(pinned, int))
    Mk = Mq[keep][:, keep]; Dk = D[:, keep]
    KK = sp.bmat([[Mk, Dk.T], [Dk, None]], format='csc')
    sol = spla.spsolve(KK, np.concatenate([b[keep], rhs_c.ravel()]))
    F = np.zeros(NF, complex); F[keep] = sol[:len(keep)]
    res = np.abs(D@F - rhs_c.ravel()).max()/max(np.abs(rhs_c).max(), 1e-300)
    return F, res


def estimator(g, nu, sigma, J, w, At, F):
    """eta_flux (cell-wise), osc (cell-wise); exact integration."""
    nx, ny, OY = g.nx, g.ny, g.OY
    nu = np.asarray(nu); sigma = np.asarray(sigma)
    I, Jj = np.meshgrid(np.arange(nx), np.arange(ny))
    FL = F[Jj*(nx + 1) + I]; FR = F[Jj*(nx + 1) + I + 1]; FB = F[OY + Jj*nx + I]; FT = F[OY + (Jj + 1)*nx + I]
    val, dx, dy = _grads_at_gauss(g, At)
    X, E = GP[None, :], GP[:, None]
    qx = (FL[..., None, None]*(1 - X) + FR[..., None, None]*X)/g.hy[:, None, None, None]
    qy = (FB[..., None, None]*(1 - E) + FT[..., None, None]*E)/g.hx[None, :, None, None]
    HX, HY = np.meshgrid(g.hx, g.hy)
    nuc = nu[..., None, None]
    dens = (np.abs(qx - nuc*dx)**2 + np.abs(qy - nuc*dy)**2)/nuc
    etaK2 = 0.25*dens.sum(axis=(-1, -2))*HX*HY
    Abar = val.mean(axis=(-1, -2))
    dev2 = 0.25*(np.abs(val - Abar[..., None, None])**2).sum(axis=(-1, -2))*HX*HY
    hmax = np.maximum(HX, HY)
    oscK2 = (hmax/np.pi)**2/nu*(w*sigma)**2*dev2
    return etaK2, oscK2


def output_terms(K, M, f, w, At):
    """Corrected output lt = 2 l(At) - b(At, At) (bilinear) and plain l(At)."""
    lA = f@At
    bAA = At@(K@At) + 1j*w*(At@(M@At))
    return 2*lA - bAA, lA


def certify(g, nu, sigma, J, w, I, At=None, KMf=None):
    """Full certificate for coil current I. Returns dict with L, R brackets and diagnostics."""
    K, M, f = KMf if KMf is not None else assemble(g, nu, sigma, J)
    galerkin = At is None
    if galerkin: At = solve(g, K, M, f, w)
    F, res = equilibrated_flux(g, nu, sigma, J, w, At)
    etaK2, oscK2 = estimator(g, nu, sigma, J, w, At, F)
    eta = np.sqrt(etaK2.sum()) + np.sqrt(oscK2.sum())
    lt, lA = output_terms(K, M, f, w, At)
    Lt = lt.real/I**2; Rt = -w*lt.imag/I**2
    c2 = 0.5*(1 + np.sqrt(2)); dL = c2*eta**2/I**2; dZ = w*dL
    return dict(L=Lt, L_lo=Lt - dL, L_up=Lt + dL, R=Rt, R_lo=Rt - dZ, R_up=Rt + dZ, dZ=dZ,
                eta=eta, eta_flux=np.sqrt(etaK2.sum()), osc=np.sqrt(oscK2.sum()), eq_res=res,
                etaK2=etaK2, A=At, F=F, energy=np.real(np.conj(At)@(K@At)), ndof=g.N)


# ---------------------------------------------------------------------------
#  Linear pieces for mutual impedances (polarisation identity).
#  For two excitations J_i, J_j of the SAME medium, the bilinear form is symmetric, so
#     l_ij = int J_i A_j = ( l(J_i + J_j) - l(J_i - J_j) )/4 ,
#  each term being a compliance output certified by Theorem 2. Galerkin solutions and the
#  optimal RT0 flux are linear in the data, so the residual fields combine linearly.
# ---------------------------------------------------------------------------
def residual_fields(g, nu, sigma, w, At, F):
    """Gauss-point flux residual r = q - nu grad At (x, y) and cell-mean deviation of At."""
    nx, ny, OY = g.nx, g.ny, g.OY
    I, Jj = np.meshgrid(np.arange(nx), np.arange(ny))
    FL = F[Jj*(nx + 1) + I]; FR = F[Jj*(nx + 1) + I + 1]; FB = F[OY + Jj*nx + I]; FT = F[OY + (Jj + 1)*nx + I]
    val, dx, dy = _grads_at_gauss(g, At)
    X, E = GP[None, :], GP[:, None]
    nuc = np.asarray(nu)[..., None, None]
    rx = (FL[..., None, None]*(1 - X) + FR[..., None, None]*X)/g.hy[:, None, None, None] - nuc*dx
    ry = (FB[..., None, None]*(1 - E) + FT[..., None, None]*E)/g.hx[None, :, None, None] - nuc*dy
    dev = val - val.mean(axis=(-1, -2))[..., None, None]
    return rx, ry, dev


def eta_from_residuals(g, nu, sigma, w, rx, ry, dev):
    HX, HY = np.meshgrid(g.hx, g.hy); nu = np.asarray(nu); sigma = np.asarray(sigma)
    ef = 0.25*((np.abs(rx)**2 + np.abs(ry)**2)/nu[..., None, None]).sum(axis=(-1, -2))*HX*HY
    d2 = 0.25*(np.abs(dev)**2).sum(axis=(-1, -2))*HX*HY
    osc = (np.maximum(HX, HY)/np.pi)**2/nu*(w*sigma)**2*d2
    return np.sqrt(ef.sum()) + np.sqrt(osc.sum())


def impedance_matrix(g, nu, sigma, Js, w, currents):
    """Certified impedance matrix of several coils (Galerkin solutions).

    Js: list of cell current-density arrays (one per coil, unit excitation); currents: I_k.
    Returns Z (n x n, per unit depth), dZ (certified radius of each entry, complex disc),
    dX (certified half-width on the reactive part w*L; equal to dZ, kept for compatibility)."""
    ncoil = len(Js); c2 = 0.5*(1 + np.sqrt(2))
    KMs = [assemble(g, nu, sigma, J) for J in Js]
    K, M, _ = KMs[0]; fs = [kmf[2] for kmf in KMs]
    D = g.dirichlet_nodes(); fr = np.setdiff1d(np.arange(g.N), D)
    lu = spla.splu((K + 1j*w*M)[fr][:, fr].tocsc())
    As, Rs = [], []
    for J, f in zip(Js, fs):
        A = np.zeros(g.N, complex); A[fr] = lu.solve(f[fr].astype(complex)); As.append(A)
        F, _ = equilibrated_flux(g, nu, sigma, J, w, A); Rs.append(residual_fields(g, nu, sigma, w, A, F))
    Z = np.zeros((ncoil, ncoil), complex); dZ = np.zeros((ncoil, ncoil)); dX = np.zeros((ncoil, ncoil))
    for i in range(ncoil):
        for j in range(i, ncoil):
            lij = fs[i]@As[j]
            if i == j:
                e2 = eta_from_residuals(g, nu, sigma, w, *Rs[i])**2; eL = eZ = c2*e2
            else:
                ep = eta_from_residuals(g, nu, sigma, w, *(a + b for a, b in zip(Rs[i], Rs[j])))**2
                em = eta_from_residuals(g, nu, sigma, w, *(a - b for a, b in zip(Rs[i], Rs[j])))**2
                ei = eta_from_residuals(g, nu, sigma, w, *Rs[i]); ej = eta_from_residuals(g, nu, sigma, w, *Rs[j])
                # polarisation bound, or Cauchy-Schwarz |b(e_i,e_j)| <= |||e_i||| |||e_j||| <= c2 eta_i eta_j
                eL = eZ = c2*min((ep + em)/4, ei*ej)
            s = currents[i]*currents[j]
            Z[i, j] = Z[j, i] = 1j*w*lij/s
            dZ[i, j] = dZ[j, i] = w*eZ/abs(s); dX[i, j] = dX[j, i] = w*eL/abs(s)
    return Z, dZ, dX, As
