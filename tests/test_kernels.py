"""Eigenvalue (positive semi-definiteness) tests for all kernels.

Uses real datasets (MUTAG, Cuneiform) with offline fallback.
"""

import os
import sys
import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_less

from grakelx.datasets import get_dataset_info
from grakelx.datasets.base import read_data
from grakelx.kernels import (
    CoreFramework,
    EdgeHistogram,
    GraphHopper,
    GraphletSampling,
    HadamardCode,
    LovaszTheta,
    MultiscaleLaplacian,
    NeighborhoodHash,
    NeighborhoodSubgraphPairwiseDistance,
    OddSth,
    Propagation,
    PropagationAttr,
    PyramidMatch,
    RandomWalk,
    RandomWalkLabeled,
    ShortestPath,
    ShortestPathAttr,
    SubgraphMatching,
    SvmTheta,
    VertexHistogram,
    WeisfeilerLehman,
    WeisfeilerLehmanOptimalAssignment,
)

is_windows = sys.platform.lower().startswith("win")

fdir = os.path.dirname(__file__)

default_eigvalue_precision = float("-1e-5")

cvxopt_available = True
try:
    import cvxopt  # noqa: F401
except ImportError:
    cvxopt_available = False

warnings.filterwarnings("ignore", category=UserWarning)
normalize = False

dataset_name = "MUTAG"
dataset_attr_name = "Cuneiform"

# consistency check for the dataset
dinfo = get_dataset_info(dataset_name)
if dinfo is None:
    raise TypeError("dataset not found")
elif not dinfo["nl"] and not dinfo["el"]:
    raise TypeError("dataset must have either node and edge labels")

# consistency check for the attribute dataset
dinfo_attr = get_dataset_info(dataset_attr_name)
if dinfo_attr is None:
    raise TypeError("dataset for attributes not found")
elif not dinfo_attr["nl"] and not dinfo_attr["el"]:
    raise TypeError("dataset must have node attributes")

# The baseline dataset for node, edge_labels (offline, subset for speed)
cwd = os.getcwd()
os.chdir(os.path.join(fdir, "data"))
dataset_all = read_data("MUTAG", with_classes=False).data
rs = np.random.RandomState(42)
rs.shuffle(dataset_all)
dataset = dataset_all[:40]
os.chdir(cwd)

# The baseline dataset for node/edge-attributes (offline, subset for speed)
cwd = os.getcwd()
os.chdir(os.path.join(fdir, "data"))
dataset_attr_all = read_data("Cuneiform", with_classes=False, prefer_attr_nodes=True, prefer_attr_edges=True).data
rs = np.random.RandomState(42)
rs.shuffle(dataset_attr_all)
dataset_attr = dataset_attr_all[:20]
dataset_attr_sp = dataset_attr_all[:10]  # smaller subset for O(V^4) ShortestPathAttr
os.chdir(cwd)


def positive_eig(kernel, X):
    """Assert that the kernel matrix is valid: correct shape, symmetric, and positive semi-definite."""
    K = kernel.fit_transform(X)
    n = len(X)
    assert K.shape == (n, n), f"Expected shape ({n}, {n}), got {K.shape}"
    assert np.allclose(K, K.T, atol=1e-10), "Kernel matrix is not symmetric"
    min_eig = np.real(np.min(np.linalg.eig(K)[0]))
    assert_array_less(default_eigvalue_precision, min_eig)


def test_random_walk():
    """Eigenvalue test for the Simple, Labelled Random Walk kernel."""
    rw_kernel = RandomWalk(verbose=False, normalize=normalize, lamda=0.01)
    positive_eig(rw_kernel, dataset)

    rw_kernel_lab = RandomWalkLabeled(verbose=False, normalize=normalize, lamda=0.0001)
    positive_eig(rw_kernel_lab, dataset)


def test_shortest_path():
    """Eigenvalue test for the Shortest Path kernel."""
    sp_kernel = ShortestPath(verbose=False, normalize=normalize)
    positive_eig(sp_kernel, dataset)

    sp_kernel_attr = ShortestPathAttr(verbose=False, normalize=normalize)
    positive_eig(sp_kernel_attr, dataset_attr_sp)


def test_graphlet_sampling():
    """Eigenvalue test for the Graphlet Sampling Kernel."""
    gs_kernel = GraphletSampling(verbose=False, normalize=normalize, sampling=dict(n_samples=150))
    positive_eig(gs_kernel, dataset)


def test_weisfeiler_lehman():
    """Eigenvalue test for the Weisfeiler Lehman kernel."""
    wl_st_kernel = WeisfeilerLehman(verbose=False, normalize=normalize, base_graph_kernel=VertexHistogram)
    positive_eig(wl_st_kernel, dataset)


def test_weisfeiler_lehman_optimal_assignment():
    """Eigenvalue test for the Weisfeiler Lehman Optimal Assignment kernel."""
    wl_oa_kernel = WeisfeilerLehmanOptimalAssignment(verbose=False, normalize=normalize)
    positive_eig(wl_oa_kernel, dataset)


def test_pyramid_match():
    """Eigenvalue test for the Pyramid Match kernel."""
    pm_kernel = PyramidMatch(verbose=False, normalize=normalize)
    positive_eig(pm_kernel, dataset)


def test_neighborhood_hash():
    """Eigenvalue test for the Neighborhood Hash kernel."""
    nh_kernel = NeighborhoodHash(verbose=False, normalize=normalize)
    positive_eig(nh_kernel, dataset)


def test_subgraph_matching():
    """Eigenvalue test for the subgraph_matching kernel."""
    sm_kernel = SubgraphMatching(verbose=False, normalize=normalize)
    positive_eig(sm_kernel, dataset)


def test_neighborhood_subgraph_pairwise_distance():
    """Eigenvalue test for the neighborhood subgraph pairwise distance kernel."""
    nspd_kernel = NeighborhoodSubgraphPairwiseDistance(verbose=False, normalize=normalize)
    positive_eig(nspd_kernel, dataset)


@pytest.mark.cvxopt
@pytest.mark.skipif(not cvxopt_available, reason="requires cvxopt")
@pytest.mark.xfail(condition=is_windows, reason="See https://github.com/ysig/GraKeL/pull/83#issuecomment-1267069069")
def test_lovasz_theta():
    """Eigenvalue test for the Lovasz-theta distance kernel."""
    lt_kernel = LovaszTheta(verbose=False, normalize=normalize)
    positive_eig(lt_kernel, dataset)


def test_svm_theta():
    """Eigenvalue test for the SVM-theta distance kernel."""
    svm_kernel = SvmTheta(verbose=False, normalize=normalize)
    positive_eig(svm_kernel, dataset)


def test_odd_sth():
    """Eigenvalue test for the ODD-STh kernel."""
    odd_sth_kernel = OddSth(verbose=False, normalize=normalize)
    positive_eig(odd_sth_kernel, dataset)


def test_propagation():
    """Eigenvalue test for the Propagation kernel."""
    propagation_kernel = Propagation(verbose=False, normalize=normalize)
    positive_eig(propagation_kernel, dataset)

    propagation_kernel_attr = PropagationAttr(verbose=False, normalize=normalize)
    positive_eig(propagation_kernel_attr, dataset_attr)


def test_hadamard_code():
    """Eigenvalue test for the Hadamard Code kernel."""
    hadamard_code_kernel = HadamardCode(verbose=False, normalize=normalize, base_graph_kernel=VertexHistogram)
    positive_eig(hadamard_code_kernel, dataset)


def test_multiscale_laplacian():
    """Eigenvalue test for the Multiscale Laplacian kernel."""
    mlf_kernel = MultiscaleLaplacian(verbose=False, normalize=normalize)
    positive_eig(mlf_kernel, dataset_attr)


def test_vertex_histogram():
    """Eigenvalue test for the Vertex Histogram Kernel."""
    vh_kernel = VertexHistogram(verbose=False, normalize=normalize)
    positive_eig(vh_kernel, dataset)


def test_edge_histogram():
    """Eigenvalue test for the Edge Histogram Kernel."""
    eh_kernel = EdgeHistogram(verbose=False, normalize=normalize)
    positive_eig(eh_kernel, dataset)


def test_graph_hopper():
    """Eigenvalue test for the Graph Hopper Kernel."""
    gh_kernel = GraphHopper(verbose=False, normalize=normalize)
    positive_eig(gh_kernel, dataset_attr)


def test_core_framework():
    """Eigenvalue test for the Core kernel Framework."""
    base_graph_kernel = (WeisfeilerLehman, dict(base_graph_kernel=VertexHistogram))
    core_framework = CoreFramework(verbose=False, normalize=normalize, base_graph_kernel=base_graph_kernel)
    positive_eig(core_framework, dataset)
