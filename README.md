# fem-trials

Solver trials for ae-sim and ti-sim (Morgan Hough; Apache-2.0). Nothing here is a package;
each directory is a dated experiment with its scripts, the container that ran it and the
numbers it produced.

## dolfinx-complex (2026-10-25, t2.macbook.pro.lan, Fedora 44, 16 cores, 10 GB memory cap)

Question (Morgan): does the dolfinx SuperLU_DIST complex path cover ae-sim's two complex
problems, the 500 kHz cone artefact (kappa = sigma + i omega eps, quasi-static) and CW
acoustics with air as a pressure-release boundary (Helmholtz with PML)?

Stack: `python3-dolfinx-openmpi 0.11.0-5` from the mhough/neurofedora COPR in a fedora:44
container (`Containerfile`: the JIT needs gcc, python3-devel, fenics-basix-devel,
fenics-dolfinx-openmpi-devel, ufcx-devel on top of the runtime package), `module load
mpi/openmpi-x86_64`, forms with `dtype=np.complex128`, `fem.assemble_matrix` ->
`la.MatrixCSR[complex128]` -> `dolfinx.la.superlu_dist`. Re-solves with the factor kept:
`solver.set_option("Fact", "FACTORED")` (the wrapper's docstring says "Factor", which is
rejected).

### It works, and it is accurate

| check (`complex_layer.py`, `helmholtz_pml.py`) | result |
|---|---|
| film layer (sigma 0, eps_r 2.3, 2 cells, capacitance-scaled) between gel and cone water, 1 V / 0 V, 500 kHz: current through the top vs the series circuit | 28.9337 uA, 78.94 deg both ways, rel err 1.8e-13 |
| omega -> 0 (1 mHz) complex solve vs the float64 solve | max diff 1.3e-10 |
| point source in water vs the free-space Green's function, P1 at 10 points per wavelength, 1.5-4.5 mm from the source | median 14.5 %, phase 8 deg (P1 pollution at 10 ppw; the 12.5 ppw run was killed by memory, see below) |
| pressure-release plane (p = 0 Dirichlet) vs the image-source solution | median 19 %, same cause; p = 0 enforced exactly |
| 0.6 mm bone slab at normal incidence vs the 1-D transfer-matrix solution, 1 mm below the slab | ratio 1.10 at 10 ppw |

### Memory is the problem, and it is Fedora's SuperLU_DIST, not dolfinx

Indefinite complex Helmholtz on the unit cube, lambda = 10 h (`dump_and_time.py` dumps the
matrices, `mumps_one.py` solves them on the host with python-mumps 0.0.6 / MUMPS 5.7.3):

| dofs | SuperLU_DIST in dolfinx (MMD_AT_PLUS_A ordering) | python-mumps (auto = METIS) |
|---|---|---|
| 68,921 | 13.2 s, 4.65 GB | 0.5 s analyse + 3.5 s factor, 0.1 s per solve, 1.19 GB |
| 132,651 | killed at 10 GB | 1.0 + 8.9 s, 0.2 s per solve, 2.76 GB |
| 274,625 | not tried | 2.2 + 32.5 s, 0.5 s per solve, 7.17 GB |

Residuals 1e-14 in every case. SuperLU_DIST's `ColPerm=METIS_AT_PLUS_A` aborts with
"Invalid ISPEC at line 545 in get_perm_c.c" although `libsuperlu_dist.so.8` links
`libmetis.so.0` and `libptscotchparmetisv3`; `COLAMD` is 4x slower and uses more memory;
so the only usable fill-reducing ordering is MMD, whose fill on 3-D tets is what kills the
132k run. A 1e6-dof Helmholtz (the rodent head at 1 MHz) extrapolates to ~40 GB with
MUMPS/METIS (the DGX Spark's 120 GB) and to nothing feasible with MMD.

### What this means for ae-sim

1. The cone-artefact problem (159 k unknowns today) is a direct complex solve either way;
   python-mumps (already ae-sim's MUMPS backend, complex works: zmumps) does it in ~10 s
   and ~3 GB with the factor reusable across the film variants. dolfinx is not needed.
2. The Helmholtz-with-PML solve is where dolfinx earns its place (forms, PML stretching,
   Dirichlet p = 0 on an irregular voxel-derived surface, MPI). Pair it with MUMPS rather
   than SuperLU_DIST until the superlu_dist ordering is fixed: assemble in dolfinx, export
   `MatrixCSR.to_scipy()` (serial) to python-mumps, or use dolfinx's PETSc path once a
   complex PETSc exists. P2 elements or 20+ points per wavelength are needed for percent
   accuracy; at 1 MHz over a 28 mm rodent head that is the 1e6-dof regime.
3. Two Fedora packaging items (drafts below, not filed): superlu_dist's METIS ordering,
   and python-mumps aborting with "buffer overflow detected" from `possible_orderings()`
   and from `analyze(ordering="scotch")` (MUMPS 5.7.3 + scotch 7.0.10); `auto`, `amd`
   and `metis` work.

### Fedora bug drafts (nothing filed)

**superlu_dist-openmpi 8.2.0-10: METIS column ordering unusable.** Selecting
`ColPerm=METIS_AT_PLUS_A` (through dolfinx 0.11's SuperLU_DIST wrapper, serial) aborts with
`Invalid ISPEC at line 545 in file .../SRC/get_perm_c.c`. The library links libmetis and
libptscotchparmetisv3, so the build seems to lack `-DTPL_ENABLE_PARMETISLIB=ON` (the
`#ifdef HAVE_PARMETIS` branch in get_perm_c.c). Without it only MMD/COLAMD orderings exist
and 3-D problems above ~1e5 unknowns exhaust memory. Reproducer: `probe_options.py 30
ColPerm=METIS_AT_PLUS_A` in the container of this directory.

**python-mumps 0.0.6-1 with MUMPS 5.7.3-7: aborts "buffer overflow detected".**
`mumps.possible_orderings()` and `Context.analyze(ordering="scotch")` terminate the
interpreter with glibc's fortify abort (scotch 7.0.10); `ordering` in {auto, amd, metis}
works. Reproducer: three lines in `mumps_one.py`'s neighbourhood (README of this directory).
