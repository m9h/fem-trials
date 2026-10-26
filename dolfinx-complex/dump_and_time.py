"""Indefinite complex Helmholtz matrix on the unit cube (10 points per wavelength at N=40):
time SuperLU_DIST in dolfinx and dump A, b for the same solve with python-mumps on the host."""
import sys, time, resource, numpy as np, scipy.sparse as sp
from mpi4py import MPI
import ufl
from dolfinx import fem, la, mesh
from dolfinx.la import superlu_dist as sd
comm = MPI.COMM_WORLD
N = int(sys.argv[1]); do_solve = len(sys.argv) < 3 or sys.argv[2] != "nosolve"
CT = np.complex128
domain = mesh.create_box(comm, [np.zeros(3), np.ones(3)], [N, N, N], mesh.CellType.tetrahedron)
V = fem.functionspace(domain, ("Lagrange", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
k2 = fem.Constant(domain, CT((2 * np.pi * N / 10) ** 2 * (1 - 0.05j)))  # lambda = 10 h, lightly absorbing
a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx, dtype=CT)
f = fem.Function(V, dtype=CT); X = V.tabulate_dof_coordinates()
f.x.array[:] = np.exp(-((X - 0.5) ** 2).sum(1) / (2 * 0.02**2))
L = fem.form(ufl.inner(f, v) * ufl.dx, dtype=CT)
A = fem.assemble_matrix(a); A.scatter_reverse(); b = fem.assemble_vector(L); b.scatter_reverse(la.InsertMode.add)
n = V.dofmap.index_map.size_global
if comm.size == 1:
    As = A.to_scipy().tocsr()
    sp.save_npz(f"/work/mat_{N}.npz", As); np.save(f"/work/rhs_{N}.npy", b.array)
    print(f"N={N}: {n} dofs, nnz {As.nnz}, saved")
if do_solve:
    t0 = time.perf_counter()
    solver = sd.superlu_dist_solver(sd.superlu_dist_matrix(A))
    x = la.vector(V.dofmap.index_map, dtype=CT); rc = solver.solve(b, x); x.scatter_forward()
    t1 = time.perf_counter()
    rss = comm.allreduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, op=MPI.SUM)
    if comm.size == 1:
        r = np.linalg.norm(As @ x.array - b.array) / np.linalg.norm(b.array)
        print(f"SuperLU_DIST (MMD ordering) N={N} {n} dofs ranks {comm.size}: factor+solve {t1-t0:.1f} s, rc {rc}, residual {r:.1e}, maxrss {rss/1e6:.2f} GB")
    elif comm.rank == 0:
        print(f"SuperLU_DIST N={N} {n} dofs ranks {comm.size}: factor+solve {t1-t0:.1f} s, rc {rc}, total maxrss {rss/1e6:.2f} GB")
