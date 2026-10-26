"""CW acoustics as a complex Helmholtz problem in dolfinx 0.11 + SuperLU_DIST, for ae-sim.

Time convention e^{+i omega t}, outgoing waves e^{-ikr} (ae_sim.acoustics.bowl_plane_phasor),
attenuation alpha (Np/m) as k_c = k - i alpha, density in the operator:
    div((1/rho) grad p) + (k_c^2/rho) p = -f .
PML by complex coordinate stretching s_i = 1 - i sigma_i(x_i)/omega on the box faces:
    -int (S/(rho s_i^2)) d_i p d_i v + int (k_c^2 S/rho) p v = -int S f v,   S = s_x s_y s_z.
Checks:
  1 point (Gaussian) source in water vs the free-space Green's function e^{-ikr}/(4 pi r)
    times the Gaussian's spectrum exp(-k^2 w^2 / 2);
  2 the same with a pressure-release plane p = 0 (air) vs the image-source solution;
  3 a bone slab at normal incidence (plane source p = 1 on the top face, PML below, lateral
    Neumann) vs the 1-D three-layer transfer-matrix solution, |p| below the slab.
Timing: assembly, factorisation, re-solve.   mpirun -n R python3 helmholtz_pml.py N
"""
import sys, time
import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, la, mesh
from dolfinx.la import superlu_dist as sd

comm = MPI.COMM_WORLD
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
F0 = 5e5; OMEGA = 2 * np.pi * F0
C_W, RHO_W = 1500.0, 1000.0
C_B, RHO_B, ALPHA_B = 2900.0, 1900.0, 80.6 * (F0 / 1e6)  # bone, Np/m at f0
Lbox = 12e-3
h = Lbox / N
d_pml = max(1.5e-3, 5 * h)
K_W = OMEGA / C_W
CT = np.complex128


def stretch(x, lo, hi, sides=("lo", "hi")):
    """s(x) = 1 - i sigma/omega, sigma quadratic in the PML, 0 inside; sigma_max from R = 1e-4."""
    sig_max = -3 * C_W * np.log(1e-4) / (2 * d_pml)
    s = np.ones(x.shape, CT)
    for side, edge, sign in (("lo", lo + d_pml, -1), ("hi", hi - d_pml, 1)):
        if side not in sides:
            continue
        t = sign * (x - edge) / d_pml
        m = t > 0
        s[m] = 1 - 1j * sig_max * t[m] ** 2 / OMEGA
    return s


def solve(domain, kc2_cell, inv_rho_cell, source_fn=None, dirichlet=(), pml_axes=(0, 1, 2), pml_sides=("lo", "hi"), label=""):
    V = fem.functionspace(domain, ("Lagrange", 1))
    Q = fem.functionspace(domain, ("DG", 0))
    xq = Q.tabulate_dof_coordinates()
    s = [stretch(xq[:, i], 0.0, Lbox, pml_sides) if i in pml_axes else np.ones(len(xq), CT) for i in range(3)]
    S = s[0] * s[1] * s[2]
    lam = [fem.Function(Q, dtype=CT) for _ in range(3)]
    for i in range(3):
        lam[i].x.array[:] = S / s[i] ** 2 * inv_rho_cell
    kS = fem.Function(Q, dtype=CT); kS.x.array[:] = kc2_cell * S * inv_rho_cell
    fS = fem.Function(Q, dtype=CT); fS.x.array[:] = (-S * source_fn(xq)) if source_fn else 0.0
    p, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    gp, gv = ufl.grad(p), ufl.grad(v)
    a = fem.form(-sum(lam[i] * gp[i] * ufl.conj(gv[i]) for i in range(3)) * ufl.dx + kS * p * ufl.conj(v) * ufl.dx, dtype=CT)
    L = fem.form(fS * ufl.conj(v) * ufl.dx, dtype=CT)
    bcs = [fem.dirichletbc(CT(val), fem.locate_dofs_geometrical(V, marker), V) for marker, val in dirichlet]
    t0 = time.perf_counter()
    A = fem.assemble_matrix(a, bcs=bcs); A.scatter_reverse()
    b = fem.assemble_vector(L); fem.apply_lifting(b.array, [a], [bcs]); b.scatter_reverse(la.InsertMode.add); fem.set_bc(b.array, bcs)
    t1 = time.perf_counter()
    solver = sd.superlu_dist_solver(sd.superlu_dist_matrix(A))
    x = la.vector(V.dofmap.index_map, dtype=CT)
    rc = solver.solve(b, x); x.scatter_forward()
    t2 = time.perf_counter()
    solver.set_option("Fact", "FACTORED")
    x2 = la.vector(V.dofmap.index_map, dtype=CT); solver.solve(b, x2); x2.scatter_forward()
    t3 = time.perf_counter()
    if comm.rank == 0:
        print(f"{label}: {V.dofmap.index_map.size_global} dofs, ranks {comm.size}, h {h*1e3:.2f} mm "
              f"({C_W / F0 / h:.0f} per wavelength): assemble {t1-t0:.1f} s, factor+solve {t2-t1:.1f} s (rc {rc}), "
              f"re-solve {t3-t2:.2f} s")
    return V, x


domain = mesh.create_box(comm, [np.zeros(3), np.full(3, Lbox)], [N, N, N], mesh.CellType.tetrahedron)
Q = fem.functionspace(domain, ("DG", 0)); xq = Q.tabulate_dof_coordinates()
centre = np.full(3, Lbox / 2)
w = 0.3e-3
gauss = lambda x: np.exp(-((x - centre) ** 2).sum(1) / (2 * w * w)) / ((2 * np.pi) ** 1.5 * w**3)
water_k2 = np.full(len(xq), K_W**2, CT); water_ir = np.full(len(xq), 1 / RHO_W)

# 1. free space
V, x = solve(domain, water_k2, water_ir, source_fn=gauss, label="1 water, point source")
if comm.size == 1:
    X = V.tabulate_dof_coordinates()
    r = np.linalg.norm(X - centre, axis=1)
    sel = (r > 1.5e-3) & (np.abs(X - centre).max(1) < Lbox / 2 - d_pml - h)
    G = np.exp(-1j * K_W * r[sel]) / (4 * np.pi * r[sel]) * np.exp(-K_W**2 * w**2 / 2) * RHO_W  # (1/rho) operator scaling
    err = np.abs(x.array[sel] - G) / np.abs(G)
    ph = np.angle(x.array[sel] / G)
    print(f"   vs Green's function on {sel.sum()} nodes, 1.5 mm < r < {(Lbox/2-d_pml)*1e3:.1f} mm: |rel err| median {np.median(err):.3f}, "
          f"p90 {np.percentile(err, 90):.3f}; phase err median {np.degrees(np.median(np.abs(ph))):.1f} deg; "
          f"|p| at r=3 mm FEM {np.abs(x.array[sel][np.argmin(np.abs(r[sel]-3e-3))]):.3e} vs {np.abs(G[np.argmin(np.abs(r[sel]-3e-3))]):.3e}")

# 2. pressure-release plane (air) at z = z0 above the source: image source
z0 = Lbox / 2 + 3e-3
z0 = round(z0 / h) * h
V, x = solve(domain, water_k2, water_ir, source_fn=gauss, dirichlet=[(lambda X: np.isclose(X[2], z0), 0.0)],
             label="2 water + p=0 plane")
if comm.size == 1:
    X = V.tabulate_dof_coordinates()
    below = X[:, 2] < z0 - h / 2
    r = np.linalg.norm(X - centre, axis=1)
    img = centre.copy(); img[2] = 2 * z0 - centre[2]
    ri = np.linalg.norm(X - img, axis=1)
    sel = below & (r > 1.5e-3) & (np.abs(X - centre).max(1) < Lbox / 2 - d_pml - h)
    G = (np.exp(-1j * K_W * r[sel]) / (4 * np.pi * r[sel]) - np.exp(-1j * K_W * ri[sel]) / (4 * np.pi * ri[sel])) * np.exp(-K_W**2 * w**2 / 2) * RHO_W
    err = np.abs(x.array[sel] - G) / np.abs(G)
    print(f"   vs image-source solution on {sel.sum()} nodes below the plane: |rel err| median {np.median(err):.3f}, p90 {np.percentile(err, 90):.3f}; "
          f"|p| on the plane max {np.abs(x.array[np.isclose(X[:, 2], z0)]).max():.1e}")

# 3. bone slab at normal incidence: p = 1 on the top face, PML only below (z), lateral Neumann
z_top, d_slab = Lbox / 2, 0.6e-3
z_top = round(z_top / h) * h; d_slab = max(2 * h, round(d_slab / h) * h)
in_bone = (xq[:, 2] < z_top) & (xq[:, 2] > z_top - d_slab)
kc_b = OMEGA / C_B - 1j * ALPHA_B
k2 = np.where(in_bone, kc_b**2, K_W**2).astype(CT)
ir = np.where(in_bone, 1 / RHO_B, 1 / RHO_W)
V, x = solve(domain, k2, ir, dirichlet=[(lambda X: np.isclose(X[2], Lbox), 1.0)], pml_axes=(2,), pml_sides=("lo",), label=f"3 bone slab {d_slab*1e3:.1f} mm")
if comm.size == 1:
    # 1-D transfer solution: water (top, p=1 at z=Lbox) | bone | water (outgoing below). Unknowns A,B (water above), C,D (bone), E (below).
    za, zb = z_top, z_top - d_slab
    kw, kb = K_W, kc_b
    def wave(k, z, zr): return np.exp(-1j * k * (z - zr)), np.exp(1j * k * (z - zr))  # down-going, up-going
    M = np.zeros((5, 5), CT); rhs = np.zeros(5, CT)
    # p(Lbox) = 1
    M[0, 0], M[0, 1] = wave(kw, Lbox, za); rhs[0] = 1
    # continuity at za: p and (1/rho) dp/dz
    M[1, 0], M[1, 1], M[1, 2], M[1, 3] = 1, 1, -1, -1
    M[2, 0], M[2, 1] = -1j * kw / RHO_W, 1j * kw / RHO_W
    M[2, 2], M[2, 3] = 1j * kb / RHO_B, -1j * kb / RHO_B
    # at zb: bone (C,D referenced to za) vs E e^{-i kw (z - zb)}
    cb, db = wave(kb, zb, za)
    M[3, 2], M[3, 3], M[3, 4] = cb, db, -1
    M[4, 2], M[4, 3], M[4, 4] = -1j * kb / RHO_B * cb, 1j * kb / RHO_B * db, 1j * kw / RHO_W
    A_, B_, C_, D_, E_ = np.linalg.solve(M, rhs)
    X = V.tabulate_dof_coordinates()
    zs = np.array([zb - 1e-3, zb - 2e-3, zb - 3e-3])
    for zz in zs:
        if zz < d_pml + h: continue
        sel = np.isclose(X[:, 2], round(zz / h) * h, atol=h / 4) & (np.abs(X[:, :2] - Lbox / 2).max(1) < 2e-3)
        p_fem = np.abs(x.array[sel]).mean()
        p_1d = abs(E_ * np.exp(-1j * kw * (round(zz / h) * h - zb)))
        print(f"   |p| {1e3*(zb - zz):.0f} mm below the slab: FEM {p_fem:.4f} vs 1-D transfer {p_1d:.4f} (ratio {p_fem/p_1d:.3f}); "
              f"transmitted/incident |E/A| = {abs(E_/A_):.3f}")
        break
