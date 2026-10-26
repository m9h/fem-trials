"""Cone-artefact proxy in dolfinx 0.11 with SuperLU_DIST, complex128.

Box Lx x Ly x Lz, P1 tets. From the bottom up: gel (sigma 1, eps_r 80), a film slab
(sigma 0, eps_r 2.3; meshed `film_cells` cells thick, admittivity scaled by meshed/physical
thickness as ae-sim does), cone water (sigma 5e-3, eps_r 80) up to the top. 1 V on the top
face, 0 V on the bottom, insulated sides; kappa = sigma + i omega eps at f0.
Checks (serial): the current through the top face against the series circuit
I = V / (d_w/(kappa_w A) + d_f/(kappa_f A) + d_g/(kappa_g A)); and omega -> 0 against a
float64 solve. Times assembly, factorisation and a second solve with "Factor"="FACTORED".

    mpirun -n R python3 complex_layer.py N [f0_Hz]      (N cells along z; x, y use N/2... see below)
"""
import sys, time
import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, la, mesh
from dolfinx.la import superlu_dist as sd

comm = MPI.COMM_WORLD
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
F0 = float(sys.argv[2]) if len(sys.argv) > 2 else 5e5
OMEGA, EPS0 = 2 * np.pi * F0, 8.854e-12
Lx = Ly = 8e-3; Lz = 10e-3
h = Lz / N
film_cells = 2
z_film = 7e-3                       # gel below, film [z_film, z_film + film_cells h], water above
d_film_mesh, d_film_phys = film_cells * h, 127e-6
kappa_gel = 1.0 + 1j * OMEGA * EPS0 * 80
kappa_water = 5e-3 + 1j * OMEGA * EPS0 * 80
kappa_film = (1j * OMEGA * EPS0 * 2.3) * (d_film_mesh / d_film_phys)  # layer capacitance preserved

def run(dtype):
    t0 = time.perf_counter()
    nx = max(4, int(round(N * Lx / Lz)))  # ~Lx/h cells
    domain = mesh.create_box(comm, [np.zeros(3), np.array([Lx, Ly, Lz])], [nx, nx, N], mesh.CellType.tetrahedron)
    V = fem.functionspace(domain, ("Lagrange", 1))
    Q = fem.functionspace(domain, ("DG", 0))
    kappa = fem.Function(Q, dtype=dtype)
    zc = Q.tabulate_dof_coordinates()[:, 2]
    k = np.where(zc < z_film, kappa_gel, np.where(zc < z_film + d_film_mesh, kappa_film, kappa_water))
    kappa.x.array[:] = k if dtype == np.complex128 else np.real(np.where(zc < z_film, 1.0, np.where(zc < z_film + d_film_mesh, 1e-6, 5e-3)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.form(kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, dtype=dtype)
    zero = fem.Constant(domain, dtype(0))
    L = fem.form(ufl.inner(zero, v) * ufl.dx, dtype=dtype)
    top = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[2], Lz))
    bot = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[2], 0.0))
    bcs = [fem.dirichletbc(dtype(1.0), top, V), fem.dirichletbc(dtype(0.0), bot, V)]
    t1 = time.perf_counter()
    A = fem.assemble_matrix(a, bcs=bcs); A.scatter_reverse()
    b = fem.assemble_vector(L)
    fem.apply_lifting(b.array, [a], [bcs]); b.scatter_reverse(la.InsertMode.add); fem.set_bc(b.array, bcs)
    t2 = time.perf_counter()
    As = sd.superlu_dist_matrix(A)
    solver = sd.superlu_dist_solver(As)
    x = la.vector(V.dofmap.index_map, dtype=dtype)
    rc = solver.solve(b, x); x.scatter_forward()
    t3 = time.perf_counter()
    refact = "reuse"
    for name in ("Fact", "Factor"):  # SuperLU_DIST calls it Fact; the wrapper docstring says Factor
        try:
            solver.set_option(name, "FACTORED"); break
        except RuntimeError:
            continue
    else:
        refact = "fresh"
    x2 = la.vector(V.dofmap.index_map, dtype=dtype)
    rc2 = solver.solve(b, x2); x2.scatter_forward()
    t4 = time.perf_counter()
    ndofs = V.dofmap.index_map.size_global
    # current through the top face: I = sum over top dofs of (A_nobc u), serial only
    cur = None
    if comm.size == 1:
        A0 = fem.assemble_matrix(a).to_scipy()
        cur = (A0 @ x.array)[top].sum()
    if comm.rank == 0:
        print(f"{dtype.__name__}: {ndofs} dofs, ranks {comm.size}: mesh+space {t1-t0:.1f} s, assemble {t2-t1:.1f} s, "
              f"factor+solve {t3-t2:.1f} s (rc {rc}), resolve ({refact}) {t4-t3:.2f} s (rc {rc2}), |x| max {np.abs(x.array).max():.3f}")
    return x.array, cur, ndofs

x_c, I_c, n = run(np.complex128)
if comm.size == 1:
    A = Lx * Ly
    Z = (Lz - z_film - d_film_mesh) / (kappa_water * A) + d_film_mesh / (kappa_film * A) + z_film / (kappa_gel * A)
    I_circ = 1.0 / Z
    print(f"current through the top: FEM {abs(I_c)*1e6:.4f} uA phase {np.degrees(np.angle(I_c)):.2f} deg | "
          f"circuit {abs(I_circ)*1e6:.4f} uA phase {np.degrees(np.angle(I_circ)):.2f} deg | rel err {abs(I_c - I_circ)/abs(I_circ):.2e}")
    # omega -> 0: complex solve at 1 mHz vs the float64 solve with the same real conductivities
    F0 = 1e-3; OMEGA = 2 * np.pi * F0
    kappa_gel = 1.0 + 1j * OMEGA * EPS0 * 80; kappa_water = 5e-3 + 1j * OMEGA * EPS0 * 80
    kappa_film = (1e-6 + 1j * OMEGA * EPS0 * 2.3) * 1.0
    x_lo, _, _ = run(np.complex128)
    x_r, _, _ = run(np.float64)
    print(f"omega -> 0: |complex - real| max {np.abs(x_lo - x_r).max():.2e} (|x| max {np.abs(x_r).max():.3f}); "
          f"imag part max {np.abs(x_lo.imag).max():.2e}")
