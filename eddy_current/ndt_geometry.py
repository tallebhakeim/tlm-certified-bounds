"""Eddy-current NDT configuration (2D planar): a two-conductor probe above an aluminium plate.
All interfaces are grid lines (geometry-conforming tensor grids), so every grid certifies the
SAME geometric model and brackets from different grids must intersect."""
import sys, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import eddy_certified2d as ec

MU0 = ec.MU0; NU0 = 1/MU0
SIG_AL = 2.5e7            # S/m, aluminium alloy plate
T = 2.0e-3                # plate thickness
XC, CW, CH = 2.5e-3, 1.5e-3, 1.5e-3   # conductor centre offset, width, height
I_COIL = 1.0              # A per conductor (go / return)
BOX = (-40e-3, 40e-3, -30e-3, 40e-3)
FINE_X = 15e-3            # fine band |x| <= FINE_X
DEF_W = 10e-3             # width of the far-side wall-loss region


def graded(a, b, h0, ratio=1.25, hc=4e-3):
    """nodes from a (spacing h0) to b with geometric growth, capped at hc; a, b in any order."""
    s = np.sign(b - a); L = abs(b - a); pts = [0.0]; h = h0
    while pts[-1] + h < L - 1e-12:
        pts.append(pts[-1] + h); h = min(h*ratio, hc)
    if L - pts[-1] < 0.5*h and len(pts) > 1: pts[-1] = L
    else: pts.append(L)
    return a + s*np.array(pts)


def build(hf, lift=0.5e-3, loss=0.0, sig=SIG_AL):
    """Grid + coefficients. loss = far-side wall loss depth (m) over |x| < DEF_W/2."""
    x0, x1, y0, y1 = BOX
    cb = [XC - CW/2, XC + CW/2]
    bx = sorted({-FINE_X, -DEF_W/2, -cb[1], -cb[0], 0.0, cb[0], cb[1], DEF_W/2, FINE_X})
    xs_f = ec.conforming_axis(bx, hf)
    xs = np.concatenate([graded(-FINE_X, x0, hf)[::-1][:-1], xs_f, graded(FINE_X, x1, hf)[1:]])
    yb0, yb1 = -1.0e-3, T + lift + CH + 1.0e-3
    by = sorted({yb0, 0.0, T, T + lift, T + lift + CH, yb1} | ({loss} if loss > 0 else set()))
    ys_f = ec.conforming_axis(by, hf)
    ys = np.concatenate([graded(yb0, y0, hf)[::-1][:-1], ys_f, graded(yb1, y1, hf)[1:]])
    g = ec.Grid(xs, ys, bc="DDDD")
    X = np.broadcast_to(g.xc[None, :], (g.ny, g.nx)); Y = np.broadcast_to(g.yc[:, None], (g.ny, g.nx))
    nu = np.full((g.ny, g.nx), NU0)
    plate = (Y > 0) & (Y < T)
    if loss > 0: plate &= ~((np.abs(X) < DEF_W/2) & (Y < loss))
    sg = np.where(plate, sig, 0.0)
    ycoil = (Y > T + lift) & (Y < T + lift + CH)
    J = np.zeros((g.ny, g.nx)); area = CW*CH
    J[ycoil & (np.abs(X - XC) < CW/2)] = I_COIL/area; J[ycoil & (np.abs(X + XC) < CW/2)] = -I_COIL/area
    return g, nu, sg, J


def certify(hf, f, **kw):
    g, nu, sg, J = build(hf, **kw)
    return ec.certify(g, nu, sg, J, 2*np.pi*f, I_COIL), g
