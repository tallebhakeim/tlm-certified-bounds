# Certified bounds for the transmission-line-matrix method

Reference implementation for the paper

> H. Talleb, *A Certified Transmission-Line-Matrix Method for Elliptic Field Problems,
> and Why the Certificate Fails on Approximated Geometry* (submitted).

The two dual TLM diffusion networks realise a Prager-Synge complementary pair: their
steady states are a conforming potential and an equilibrated flux, so their energies
bracket the quantity of interest, and the width of the bracket is a **guaranteed** bound
on the error. Passivity of the scattering matrix makes every iterate admissible, so a
valid interval exists at every iteration and tightens monotonically.

The repository also reproduces the paper's negative result: a complementary bound
encloses the exact solution of the problem the discretisation actually represents, and
is **silent about geometric error**. On uniform voxel grids the same solver returns five
rigorous but mutually disjoint brackets for one capacitance.

## Install

```sh
python3 -m pip install -r requirements.txt
```

Python 3.9 or later. No compilation, no external solver.

## Scripts

Each script writes its figure next to itself and prints the numbers quoted in the paper.

| Script | Reproduces | Runtime |
|---|---|---|
| `src/poc_2d.py` | Table I, Figure 3. Two-dimensional validation, gap in O(h^1.45) | seconds |
| `src/passivity.py` | Figure 2. Anytime certificate, 0 violations in 9 000 iterations | ~1 min |
| `src/contraction.py` | Section 3. Spectral radius, 1 - rho in O(n^-2) | ~1 min |
| `src/compare.py` | Tables II and IIIa, Figure 4. Contrast sweep, convergence by geometry, effectivity index 2.00 | ~2 min |
| `src/tlm3d.py` | Table IV and Section 5.2. Three-dimensional validation on conforming grids, and the documented counter-example on non-conforming ones | ~5 min |
| `src/duel_dgm.py` | Table V, Figure 5. Non-circular benchmark against a dual discrete-geometric solver (see *Dependencies*) | ~1 min |
| `src/ic_package_refine.py` | Tables VI and VIII. Certified capacitance on conforming grids, and the counter-proof on uniform ones | ~25 min |
| `src/ic_package.py` | Figure 7. Package figure; delegates the physics to the previous script and caches the finest solve | ~5 min |
| `src/geometric_error.py` | Figure 6. Requires `ic_package_refine.py` and `tlm3d.py` to have been run | seconds |

Suggested order:

```sh
cd src
python3 poc_2d.py && python3 passivity.py && python3 compare.py
python3 tlm3d.py
python3 ic_package_refine.py      # writes /tmp/ic_pkg_refine.npz and /tmp/ic_pkg_uniform.npz
python3 ic_package.py
python3 geometric_error.py
```

## Main results

Certified pad-to-substrate capacitance of the package model, geometry-conforming grids:

| Cells | DOF (upper / lower) | Certified bracket | Half-width | CPU |
|---|---|---|---|---|
| 1 960 | 1 750 / 6 009 | [849.00, 1130.43] fF | 14.22 % | 0.03 s |
| 6 776 | 5 721 / 20 717 | [901.88, 1080.56] fF | 9.01 % | 0.40 s |
| 14 896 | 12 453 / 46 025 | [925.93, 1066.50] fF | 7.06 % | 2.88 s |
| 38 400 | 31 601 / 119 673 | [953.43, 1047.15] fF | 4.68 % | 21.4 s |
| 102 060 | 81 097 / 311 277 | **[968.51, 1037.37] fF** | 3.43 % | 241 s |

The same solver on uniform grids gives `[944.14, 1141.39]`, `[889.82, 1017.71]`,
`[829.85, 919.97]`, `[996.80, 1073.05]` and `[986.28, 1047.05]` fF: five rigorous
brackets whose intersection is **empty**, because each grid sees a different pad width
(-8.6 to +1.3 per cent) and a different gap (-5.1 to +4.4 per cent).

**A cheap test worth running on any certified computation:** brackets obtained on
successively refined grids must have a non-empty intersection. An empty one indicts the
geometry, not the solver.

## The package model

`model/ic_package.stl` is the quad-flat-package model used for the extraction, drawn by
the author and released here under the same MIT licence as the code. The solver reads
three bounding boxes from it, the die pad, the substrate ground plane and the moulding
compound footprint, whose values are listed in Table VII of the paper; the remaining
bodies are the leads, used only for the top view of Figure 7.

## Dependencies

`duel_dgm.py` is the only script that needs anything beyond the requirements: it
compares against the author's dual discrete-geometric solver, which is the subject of a
separate publication and is not redistributed here. Set `DGM_PATH` to its location to
run that comparison. Everything else, including the certified bracket itself, is
self-contained.

## Citing

See `CITATION.cff`. Please cite the archived release through its concept DOI, which
always resolves to the latest version, together with the paper.

## Licence

MIT, see `LICENSE`.
