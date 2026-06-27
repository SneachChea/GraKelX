"""Init file for the whole grakel project."""

from grakelx import datasets
from grakelx.graph import Graph
from grakelx.graph_kernels import GraphKernel
from grakelx.kernels import (
    CoreFramework,
    EdgeHistogram,
    GraphHopper,
    GraphletSampling,
    HadamardCode,
    Kernel,
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
from grakelx.utils import (
    KMTransformer,
    cross_validate_Kfold_SVM,
    graph_from_csv,
    graph_from_networkx,
    graph_from_pandas,
    graph_from_torch_geometric,
)

__all__ = [
    "datasets",
    "GraphKernel",
    "Graph",
    "Kernel",
    "GraphletSampling",
    "RandomWalk",
    "RandomWalkLabeled",
    "ShortestPath",
    "ShortestPathAttr",
    "WeisfeilerLehman",
    "NeighborhoodHash",
    "PyramidMatch",
    "SubgraphMatching",
    "NeighborhoodSubgraphPairwiseDistance",
    "LovaszTheta",
    "SvmTheta",
    "OddSth",
    "Propagation",
    "PropagationAttr",
    "HadamardCode",
    "MultiscaleLaplacian",
    "VertexHistogram",
    "EdgeHistogram",
    "GraphHopper",
    "CoreFramework",
    "WeisfeilerLehmanOptimalAssignment",
    "graph_from_networkx",
    "graph_from_pandas",
    "graph_from_csv",
    "graph_from_torch_geometric",
    "KMTransformer",
    "cross_validate_Kfold_SVM",
]

# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = "0.1.12"
