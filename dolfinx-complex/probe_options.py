import sys, time, resource, numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, la, mesh
from dolfinx.la import superlu_dist as sd
comm = MPI.COMM_WORLD
N = int(sys.argv[1]); opts = sys.argv[2:]
CT = np.complex128
domain = mesh.create_box(comm, [np.zeros(3), np.ones(3)], [N, N, N], mesh.CellType.tetrahedron)
V = fem.functionspace(domain, ("Lagrange", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
k = fem.Constant(domain, CT(1.0 + 0.3j))
a = fem.form(k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx, dtype=CT)
L = fem.form(ufl.inner(fem.Constant(domain, CT(1.0)), v) * ufl.dx, dtype=CT)
A = fem.assemble_matrix(a); A.scatter_reverse(); b = fem.assemble_vector(L); b.scatter_reverse(la.InsertMode.add)
solver = sd.superlu_dist_solver(sd.superlu_dist_matrix(A))
for o in opts:
    name, val = o.split("=")
    try:
        solver.set_option(name, val); msg = "ok"
    except RuntimeError as e:
        msg = f"rejected ({e})"
    if comm.rank == 0: print(f"option {name}={val}: {msg}")
x = la.vector(V.dofmap.index_map, dtype=CT)
t0 = time.perf_counter(); rc = solver.solve(b, x); t1 = time.perf_counter()
rss = comm.allreduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, op=MPI.SUM)
if comm.rank == 0:
    print(f"N={N} dofs {V.dofmap.index_map.size_global} ranks {comm.size} opts {opts}: factor+solve {t1-t0:.1f} s rc {rc}, total maxrss {rss/1e6:.2f} GB")
