"""
3D dual-TLM EIT — infrastructure + validation du bracket sur un cube.

Probleme : conduction 3D  -div(sigma grad u)=0  dans le cube unite, conductance G
entre les faces x=0 (u=0) et x=1 (u=1), faces y,z isolees (Neumann).

Borne SUP. : energie nodale Q1-hex (conforme)  G_up = phi^T K phi   (realisable par
             le TLM scatter/connect, anytime ; ici solve direct pour valider le bracket).
Borne INF. : flux equilibre RT0 par FACES (systeme mixte [[A,B^T],[B,0]]),  G_lo.
=> bracket garanti  G_lo <= G_exact <= G_up. On valide qu'il se referme (cube + inclusion).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

np.seterr(over='ignore', divide='ignore', invalid='ignore')

# ---- Q1 hex reference stiffness  Kref8 = sum_gauss w * gradref_i . gradref_j ----
def _kref8():
    g=1/np.sqrt(3); gp=[(-g,-g,-g),(g,-g,-g),(g,g,-g),(-g,g,-g),(-g,-g,g),(g,-g,g),(g,g,g),(-g,g,g)]
    # local node coords on [-1,1]^3 (same order)
    nc=np.array([(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)],float)
    K=np.zeros((8,8))
    for (xi,et,ze) in gp:
        # dN_i/dxi etc for trilinear N_i=1/8 (1+xi*xi_i)(1+et*et_i)(1+ze*ze_i)
        G=np.zeros((8,3))
        for i,(a,b,c) in enumerate(nc):
            G[i,0]=0.125*a*(1+b*et)*(1+c*ze)
            G[i,1]=0.125*b*(1+a*xi)*(1+c*ze)
            G[i,2]=0.125*c*(1+a*xi)*(1+b*et)
        K+=G@G.T
    return K
KREF8=_kref8()   # 8x8, for unit reference cube; physical Ke = sigma*(h/2)*KREF8

def assemble_K(n, sigfun, sig_cells=None):
    """Vectorise. sig_cells: array (n,n,n) [k,j,i] optionnel (sinon sigfun(i,j,k))."""
    npn=n+1; nid=lambda i,j,k:(k*npn+j)*npn+i; h=1.0/n; N=npn**3
    I,J,K=np.meshgrid(np.arange(n),np.arange(n),np.arange(n),indexing='ij')
    I=I.ravel();J=J.ravel();Kk=K.ravel(); nc=n**3
    if sig_cells is not None: sig=sig_cells[Kk,J,I].astype(float)
    else: sig=np.array([sigfun(int(i),int(j),int(k)) for i,j,k in zip(I,J,Kk)],float)
    loc=[(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    nodes=np.stack([nid(I+dx,J+dy,Kk+dz) for dx,dy,dz in loc],axis=1)   # (nc,8)
    Ke=(h/2)*KREF8                                                      # 8x8
    rows=np.broadcast_to(nodes[:,:,None],(nc,8,8)).ravel()
    cols=np.broadcast_to(nodes[:,None,:],(nc,8,8)).ravel()
    vals=(sig[:,None,None]*Ke[None,:,:]).ravel()
    return sp.coo_matrix((vals,(rows,cols)),shape=(N,N)).tocsr(),nid

def primal_upper(n, sigfun, sig_cells=None):
    K,nid=assemble_K(n,sigfun,sig_cells); npn=n+1; N=npn**3
    fix={}
    for k in range(npn):
        for j in range(npn):
            fix[nid(0,j,k)]=0.0; fix[nid(n,j,k)]=1.0
    fi=np.array(sorted(fix)); u=np.zeros(N); u[fi]=[fix[t] for t in fi]
    fr=np.setdiff1d(np.arange(N),fi)
    u[fr]=spla.spsolve(K[fr][:,fr].tocsc(),-K[fr][:,fi]@u[fi])
    return float(u@(K@u)), K, u, npn   # G_up = u^T K u  (V=1)

# ---- RT0 (face-flux) lower bound on the voxel grid ----
def rt0_lower(n, sigfun):
    h=1.0/n; sig=np.array([[[sigfun(i,j,k) for i in range(n)] for j in range(n)] for k in range(n)])
    # face indexing
    nxf=(n+1)*n*n      # x-faces : (i in 0..n, j,k in 0..n-1)
    nyf=n*(n+1)*n
    nzf=n*n*(n+1)
    xid=lambda i,j,k:(k*n+j)*(n+1)+i
    yid=lambda i,j,k:(k*(n+1)+j)*n+i
    zid=lambda i,j,k:(k*n+j)*n+i
    OFFY=nxf; OFFZ=nxf+nyf; NF=nxf+nyf+nzf
    # face resistance R_f = 1/(sigma_f * h)  (cube face area h^2, length h ; harmonic mean sigma)
    Rdiag=np.zeros(NF); pinned=np.zeros(NF,bool); gbnd=np.zeros(NF)
    def hm(a,b): return 2*a*b/(a+b) if (a+b)>0 else 0.0
    # x-faces  (bord = demi-cellule -> demi-resistance ; g = -u_D)
    for k in range(n):
        for j in range(n):
            for i in range(n+1):
                f=xid(i,j,k)
                if i==0:   sf=sig[k,j,0];   gbnd[f]=-0.0; Rdiag[f]=1.0/(2*sf*h)
                elif i==n: sf=sig[k,j,n-1]; gbnd[f]=-1.0; Rdiag[f]=1.0/(2*sf*h)
                else:      sf=hm(sig[k,j,i-1],sig[k,j,i]); Rdiag[f]=1.0/(sf*h)
    # y-faces (Neumann on j=0,n boundary -> pinned)
    for k in range(n):
        for j in range(n+1):
            for i in range(n):
                f=OFFY+yid(i,j,k)
                if j==0 or j==n: pinned[f]=True; Rdiag[f]=1.0
                else: Rdiag[f]=1.0/(hm(sig[k,j-1,i],sig[k,j,i])*h)
    # z-faces (Neumann on k=0,n -> pinned)
    for k in range(n+1):
        for j in range(n):
            for i in range(n):
                f=OFFZ+zid(i,j,k)
                if k==0 or k==n: pinned[f]=True; Rdiag[f]=1.0
                else: Rdiag[f]=1.0/(hm(sig[k-1,j,i],sig[k,j,i])*h)
    # divergence B : per cell, +out faces
    rows,cols,vals=[],[],[]
    cid=lambda i,j,k:(k*n+j)*n+i
    for k in range(n):
        for j in range(n):
            for i in range(n):
                c=cid(i,j,k)
                for f,s in [(xid(i+1,j,k),+1),(xid(i,j,k),-1),
                            (OFFY+yid(i,j+1,k),+1),(OFFY+yid(i,j,k),-1),
                            (OFFZ+zid(i,j,k+1),+1),(OFFZ+zid(i,j,k),-1)]:
                    rows.append(c);cols.append(f);vals.append(s)
    B=sp.csr_matrix((vals,(rows,cols)),shape=(n**3,NF))
    keep=~pinned
    A=sp.diags(Rdiag[keep]); Bk=B[:,keep]; gk=gbnd[keep]
    nfk=int(keep.sum()); nc=n**3
    KK=sp.bmat([[A,Bk.T],[Bk,None]],format='csc')
    sol=spla.spsolve(KK,np.concatenate([gk,np.zeros(nc)]))
    q=sol[:nfk]
    return float(q@(A@q))   # G_lo = 2*W_lower = q^T A q  (dual complementary energy, V=1)

# ---- 3D TLM scatter/connect (anytime upper bound), native on the voxel graph ----
def _graph(K):
    Kc=K.tocoo(); m=(Kc.row!=Kc.col)&(Kc.data<0)
    s,d,g=Kc.row[m],Kc.col[m],-Kc.data[m]
    key={(a,b):e for e,(a,b) in enumerate(zip(s,d))}
    rev=np.array([key[(b,a)] for a,b in zip(s,d)]); return s,d,g,rev

def tlm_upper_3d(n, sigfun, nit, alpha=0.9):
    """Renvoie G_up(k)=V(k)^T K V(k) (anytime) ; Dirichlet x=0 (0) / x=n (1)."""
    K,nid=assemble_K(n,sigfun); npn=n+1; N=npn**3; s,d,g,rev=_graph(K)
    is_dir=np.zeros(N,bool); phi=np.zeros(N)
    for k in range(npn):
        for j in range(npn):
            is_dir[nid(0,j,k)]=True; phi[nid(0,j,k)]=0.0
            is_dir[nid(n,j,k)]=True; phi[nid(n,j,k)]=1.0
    den=np.bincount(s,weights=g,minlength=N); inc=np.zeros(len(s))
    V=np.zeros(N); V[is_dir]=phi[is_dir]; G=np.empty(nit)
    for it in range(nit):
        num=np.bincount(s,weights=g*inc,minlength=N)
        Vn=np.where(den>0,2.0*num/np.where(den>0,den,1.0),0.0); Vn[is_dir]=phi[is_dir]
        refl=Vn[s]-inc; inc=alpha*refl[rev]+(1-alpha)*refl; V=Vn; G[it]=float(V@(K@V))
    return G

# ---------- validation ----------
if __name__=="__main__":
    print("KREF8 row-sum (should be ~0):", np.abs(KREF8.sum(1)).max())
    # cube homogene : G_true = sigma * L = 1
    for n in [8,12]:
        Gup,_,_,_=primal_upper(n,lambda i,j,k:1.0)
        Glo=rt0_lower(n,lambda i,j,k:1.0)
        print(f"[homogene] n={n}: G in [{Glo:.5f}, {Gup:.5f}]  (attendu 1.0)")
    # cube avec inclusion conductrice centrale (sigma=10) sur [0.3,0.7]^3
    #
    # ATTENTION : n doit etre multiple de 10 pour que les faces 0.3 et 0.7 tombent
    # EXACTEMENT sur des lignes de grille. Sinon l'inclusion est capturee par les
    # cellules dont le centre y tombe, son arete effective saute (0.500 a n=8,
    # 0.333 a n=12, 0.375 a n=16) et chaque grille certifie un PROBLEME DIFFERENT :
    # les brackets obtenus sont alors mutuellement disjoints, sans qu'aucun d'eux
    # soit faux. Prager-Synge borne l'erreur de discretisation d'une geometrie
    # donnee, il ne couvre pas l'erreur geometrique.
    def incl(i, j, k, n):
        c = ((i+0.5)/n, (j+0.5)/n, (k+0.5)/n)
        return 10.0 if all(0.3 <= t <= 0.7 for t in c) else 1.0

    def edge_seen(n):
        """Arete de l'inclusion effectivement representee par la grille."""
        c = (np.arange(n)+0.5)/n
        m = (c >= 0.3) & (c <= 0.7)
        return (m.sum())/n

    print("\n[inclusion sigma=10 sur [0.3,0.7]^3 ; grilles conformes n = 10, 20, 30]")
    res = []
    for n in [10, 20, 30]:
        Gup, _, _, _ = primal_upper(n, lambda i, j, k: incl(i, j, k, n))
        Glo = rt0_lower(n, lambda i, j, k: incl(i, j, k, n))
        res.append((n, Glo, Gup))
        print(f"  n={n:2d}: arete vue={edge_seen(n):.4f} (exacte 0.4000)  "
              f"G in [{Glo:.5f}, {Gup:.5f}]  gap={Gup-Glo:.3e}  ok={Glo<=Gup}")
    lo = [r[1] for r in res]; up = [r[2] for r in res]
    print(f"  intersection = [{max(lo):.5f}, {min(up):.5f}] -> "
          f"{'NON VIDE (coherent)' if max(lo) < min(up) else 'VIDE (INCOHERENT)'}")
    print(f"  borne inf croissante: {all(lo[i] <= lo[i+1] for i in range(len(lo)-1))} ; "
          f"borne sup decroissante: {all(up[i] >= up[i+1] for i in range(len(up)-1))}")
    h = np.array([1.0/r[0] for r in res]); gap = np.array([r[2]-r[1] for r in res])
    print(f"  ordre du gap : O(h^{np.polyfit(np.log(h), np.log(gap), 1)[0]:.2f})")

    np.savez('/tmp/tlm3d_incl.npz',
             n_ok=np.array([r[0] for r in res]),
             lo_ok=np.array([r[1] for r in res]),
             up_ok=np.array([r[2] for r in res]),
             edge_ok=np.array([edge_seen(r[0]) for r in res]))
    print("\n[contre-exemple documente : grilles NON conformes n = 8, 12, 16]")
    bad = []
    for n in [8, 12, 16]:
        Gup, _, _, _ = primal_upper(n, lambda i, j, k: incl(i, j, k, n))
        Glo = rt0_lower(n, lambda i, j, k: incl(i, j, k, n))
        bad.append((Glo, Gup))
        print(f"  n={n:2d}: arete vue={edge_seen(n):.4f}  G in [{Glo:.5f}, {Gup:.5f}]")
    lo = [b[0] for b in bad]; up = [b[1] for b in bad]
    print(f"  intersection = [{max(lo):.5f}, {min(up):.5f}] -> "
          f"{'NON VIDE' if max(lo) < min(up) else 'VIDE (brackets disjoints)'}")
    d = dict(np.load('/tmp/tlm3d_incl.npz'))
    d.update(n_bad=np.array([8, 12, 16]), lo_bad=np.array(lo), up_bad=np.array(up),
             edge_bad=np.array([edge_seen(n) for n in (8, 12, 16)]))
    np.savez('/tmp/tlm3d_incl.npz', **d)
    print("  -> /tmp/tlm3d_incl.npz")
