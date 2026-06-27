# GraKeL — Agent Guide

## Build

```bash
uv sync --group test
uv run python setup.py build_ext --inplace   # compiles Cython extensions (_c_functions, _isomorphism)
```

Cython is a build-time dependency (not optional). The extensions produce platform-specific `.so` files under `grakel/kernels/`.

## Test

Single command:
```bash
uv run pytest
```

Test files (7 files, ~3,100 LOC):
- `tests/test_Kernel.py` — main kernel suite, one `def test_*` per kernel class (hand-written, not parametrized)
- `tests/test_kernels.py` — eigenvalue (positive semi-definite) tests using real MUTAG/Cuneiform datasets
- `tests/test_graph_kernels.py` — `GraphKernel` string-dispatcher wrapper tests
- `tests/test_common.py` — picklability tests for every kernel
- `tests/test_utils.py` — converter tests (graph_from_pandas, graph_from_networkx, etc.)
- `tests/test_graph.py` — `Graph` class consistency across internal formats
- `tests/test_windows_sdp_issue.py` — single parametrized test, currently disabled (`DISABLED = True`)
- `tests/__main__.py` — legacy bootstrap (calls each test file as a subprocess), not used in CI

`LovaszTheta` / `SvmTheta` tests gate on `cvxopt` importability.

## Format & Lint

Pre-commit hooks (`.pre-commit-config.yaml`): ruff (lint+fix), ruff-format, isort, black.
Line length: 130 across all tools. Configured in `pyproject.toml`.

```bash
uv run ruff check . --fix
uv run ruff format .
```

## Architecture — Key Entry Points

- `grakelx/graph.py` — `Graph` class (1,779 LOC), dual internal representation (edge dictionary + adjacency matrix). Shared by all kernels.
- `grakelx/kernels/kernel.py` — `Kernel` base class (410 LOC). All 22 concrete kernel classes inherit from this. Provides `fit`, `transform`, `fit_transform`, `parse_input`, `_calculate_kernel_matrix` (parallel dispatch via `k_to_ij_*` Cython helpers).
- `grakelx/graph_kernels.py` — `GraphKernel` (584 LOC). String-dispatcher wrapper. The typical user entry point for sklearn pipelines.
- `grakelx/kernels/` — 22 concrete classes in 16 files, divided into **base kernels** and **frameworks** (kernels that wrap a base kernel):

  | Category | Classes |
  |----------|---------|
  | Base kernels (16) | VertexHistogram, EdgeHistogram, ShortestPath, GraphletSampling, RandomWalk, NeighborhoodHash, NeighborhoodSubgraphPairwiseDistance, LovaszTheta, SvmTheta, OddSth, GraphHopper, Propagation, PyramidMatch, SubgraphMatching, MultiscaleLaplacian, WeisfeilerLehmanOptimalAssignment |
  | Frameworks (3) | WeisfeilerLehman, HadamardCode, CoreFramework |
  | Labeled/attr variants (3) | ShortestPathAttr (sibling of ShortestPath), RandomWalkLabeled (subclass of RW), PropagationAttr (subclass of Prop) |

  Default base kernel for all frameworks: `VertexHistogram`.
  `CoreFramework` directly imports `ShortestPath` (bypassing GraphKernel).

- `grakelx/kernels/_c_functions/` — Cython extension: `APHash`, `sm_kernel`, `ConSubg`, `k_to_ij_rectangular`, `k_to_ij_triangular`. Consumers: `kernel.py`, `subgraph_matching.py`, `neighborhood_subgraph_pairwise_distance.py`, `graphlet_sampling.py`.
- `grakelx/kernels/_isomorphism/` — Cython wrapper around vendored BLISS 0.50 (graph automorphism, LGPL). Consumer: `graphlet_sampling.py` only.
- `grakelx/datasets/` — TU-Dortmund dataset fetcher (~50 datasets hardcoded in `base.py`) + synthetic generator (`testing.py`).
- `grakelx/utils/` — `KMTransformer`, `cross_validate_Kfold_SVM`, `graph_from_networkx`, `graph_from_pandas`, `graph_from_csv`, `graph_from_torch_geometric`.

## Important Quirks

- **Version**: `grakelx/__init__.py` now declares `__version__ = "0.1.12"`, matching `pyproject.toml`.
- **Stale docs**: `doc/classes.rst` autosummary lists `MultiscaleLaplacianFast` — this class was removed after 0.1a8 and does not exist in source.
- **`.pyx` / `.pxd` files**: Python `ast` cannot parse them (`cdef`, `cpdef`, `cimport` syntax). Graphify and any AST-based tool will produce dangling import references for symbols defined in `_c_functions`/`_isomorphism`.
- **Empty `__init__.pyx`**: `grakelx/kernels/_c_functions/__init__.pyx` is 0 bytes. This is a Cython build marker, not a package init. The package resolves via `functions.pyx`.
- **Generated `.cpp` files**: `functions.cpp` (~1MB) and `bliss.cpp` (~970KB) are Cython-generated build artifacts, not hand-authored C++. `.gitignore` tracks only the linux-3.5 variant; the Mac darwin builds are checked in.
- **Vendored BLISS**: `grakelx/kernels/_isomorphism/bliss-0.50/` is unmodified upstream LGPL code (~8,000 LOC). Not authored by GraKeL. Only `graphlet_sampling.py` uses it.
- **Platform .so files**: Compiled `_c_functions.cpython-312-darwin.so`, `bliss.cpython-312-darwin.so`, `intpybliss.cpython-312-darwin.so` are tracked in the working tree but should be recompiled for each Python version.
- **`Propagation` has a CLI**: `propagation.py` contains a 147-line `if __name__ == "__main__":` block that runs a benchmark between BZR/MUTAG datasets. `import tqdm` only happens inside this block.
- **`cvxopt` optional**: Required only for `LovaszTheta` and `SvmTheta`. Install with `uv sync --extra lovasz`.
- **Extras**: `io` (networkx, pandas), `torch` (torch+torch-geometric), `examples` (nltk, networkx), `tutorials` (matplotlib, nltk, networkx), `docs` (sphinx, sphinx-rtd-theme, numpydoc, sphinx-gallery, sphinxcontrib-bibtex).

## Graphify

The knowledge graph lives in `graphify-out/` (gitignored). To rebuild from scratch:

```bash
/graphify .
```

Outputs: `graph.html` (interactive), `graph.json` (GraphRAG-ready), `GRAPH_REPORT.md` (audit).

Known issue: Cython-generated `.cpp` files introduce ~600 Cython-internal nodes (`__Pyx_*` functions). The real GraKeL architecture (`Graph`, `Kernel`, `GraphKernel`) is visible beneath this noise.
