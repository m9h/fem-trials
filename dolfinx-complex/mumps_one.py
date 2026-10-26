import sys, time, resource, numpy as np, scipy.sparse as sp, mumps
N, ordering, n_sub = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
A = sp.load_npz(f"mat_{N}.npz").tocsr(); b = np.load(f"rhs_{N}.npy")
if n_sub: A = A[:n_sub, :n_sub].tocsr(); b = b[:n_sub]
t0 = time.perf_counter(); ctx = mumps.Context(); ctx.set_matrix(A); ctx.analyze(ordering=ordering); t1 = time.perf_counter()
ctx.factor(); t2 = time.perf_counter(); x = ctx.solve(b); t3 = time.perf_counter(); x2 = ctx.solve(b); t4 = time.perf_counter()
print(f"python-mumps {ordering} {A.shape[0]} dofs: analyze {t1-t0:.1f} s factor {t2-t1:.1f} s solve {t3-t2:.2f} s re-solve {t4-t3:.2f} s "
      f"residual {np.linalg.norm(A @ x - b)/np.linalg.norm(b):.1e} maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
