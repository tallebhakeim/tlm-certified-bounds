"""Exact reference for the layered (1D-reducible) eddy-current benchmark.

Layers in y with constant nu, sigma, J; A(0) = A(H) = 0; -nu A'' + j w sigma A = J.
Returns l = W * int J A dy (W = width in x, Neumann sides) evaluated in closed form.
"""
import numpy as np


def slab_exact(layers, w, W=1.0):
    """layers: list of (y0, y1, nu, sigma, J). Returns (l, A(y) callable)."""
    n = len(layers)
    # unknowns c1, c2 per layer; basis: sigma>0 -> exp(k (y-y0)), exp(-k (y-y0)); sigma=0 -> 1, (y-y0)
    def basis(L, y):
        y0, y1, nu, sg, J = L
        s = y - y0
        if sg > 0:
            k = np.sqrt(1j*w*sg/nu)
            return (np.exp(k*s), np.exp(-k*s), J/(1j*w*sg)), (k*np.exp(k*s), -k*np.exp(-k*s), 0.0)
        return (1.0 + 0j, s + 0j, -J*s**2/(2*nu)), (0j, 1.0 + 0j, -J*s/nu)
    Mat = np.zeros((2*n, 2*n), complex); rhs = np.zeros(2*n, complex); r = 0
    (b1, b2, p), _ = basis(layers[0], layers[0][0]); Mat[r, 0:2] = [b1, b2]; rhs[r] = -p; r += 1
    for m in range(n - 1):
        La, Lb = layers[m], layers[m + 1]; y = La[1]
        (a1, a2, pa), (da1, da2, dpa) = basis(La, y); (c1, c2, pc), (dc1, dc2, dpc) = basis(Lb, y)
        Mat[r, 2*m:2*m + 4] = [a1, a2, -c1, -c2]; rhs[r] = pc - pa; r += 1
        Mat[r, 2*m:2*m + 4] = [La[2]*da1, La[2]*da2, -Lb[2]*dc1, -Lb[2]*dc2]; rhs[r] = Lb[2]*dpc - La[2]*dpa; r += 1
    (b1, b2, p), _ = basis(layers[-1], layers[-1][1]); Mat[r, 2*n - 2:2*n] = [b1, b2]; rhs[r] = -p
    c = np.linalg.solve(Mat, rhs)
    l = 0j
    for m, L in enumerate(layers):
        y0, y1, nu, sg, J = L
        if J == 0: continue
        h = y1 - y0
        if sg > 0:
            k = np.sqrt(1j*w*sg/nu)
            l += J*(c[2*m]*(np.exp(k*h) - 1)/k + c[2*m + 1]*(1 - np.exp(-k*h))/k + J/(1j*w*sg)*h)
        else:
            l += J*(c[2*m]*h + c[2*m + 1]*h**2/2 - J*h**3/(6*nu))
    def A(y):
        y = np.atleast_1d(y); out = np.zeros(len(y), complex)
        for m, L in enumerate(layers):
            sel = (y >= L[0]) & (y <= L[1])
            (b1, b2, p), _ = basis(L, y[sel]); out[sel] = c[2*m]*b1 + c[2*m + 1]*b2 + p
        return out
    return W*l, A
