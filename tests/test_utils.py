"""Tests for utils.py"""

# Author: Ioannis Siglidis <y.siglidis@gmail.com>
# License: BSD 3 clause
from __future__ import print_function

import os
from warnings import warn

from numpy import arange
from numpy.random import RandomState

from grakelx import (
    Graph,
    ShortestPath,
    VertexHistogram,
    WeisfeilerLehman,
    cross_validate_Kfold_SVM,
    graph_from_csv,
    graph_from_networkx,
    graph_from_pandas,
    graph_from_torch_geometric,
)
from grakelx.datasets import fetch_dataset
from grakelx.datasets.base import read_data

global verbose

# Add extra arguments for allowing unit testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A test file for all `Graph` type objects")
    parser.add_argument("--verbose", help="verbose outputs on stdout", action="store_true")
    parser.add_argument("--ignore_warnings", help="ignore warnings produced by kernel executions", action="store_true")

    args = parser.parse_args()
    verbose = bool(args.verbose)

    if bool(args.ignore_warnings):
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning)
else:
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    verbose = False


def test_pandas():
    """Testing Graph object consistency for an adjacency-type initialization object."""
    # Input

    try:
        import pandas as pd
    except ImportError:
        return

    ga = {0: {0: 1.0, 1: 1.0, 3: 3.0}, 1: {0: 1.0, 3: 2.0}, 2: {0: 2.0, 1: 3.0, 3: 1.0}, 3: {0: 1.0}}

    ganl = {0: "l1", 1: "l2", 2: "l3", 3: "l4"}

    gael = {
        (0, 0): "el1",
        (0, 1): "el2",
        (0, 3): "el3",
        (1, 0): "el4",
        (1, 3): "el5",
        (2, 0): "el6",
        (2, 1): "el7",
        (2, 3): "el8",
        (3, 0): "el9",
    }

    gb = {4: {4: 1.0, 5: 1.0, 7: 3.0}, 5: {4: 1.0, 7: 2.0}, 6: {4: 2.0, 5: 3.0, 7: 1.0}, 7: {4: 1.0}}

    gbnl = {4: "l1", 5: "l2", 6: "l3", 7: "l4"}

    gbel = {
        (4, 4): "el1",
        (4, 5): "el2",
        (4, 7): "el3",
        (5, 4): "el4",
        (5, 7): "el5",
        (6, 4): "el6",
        (6, 5): "el7",
        (6, 7): "el8",
        (7, 4): "el9",
    }

    nodes_df = pd.DataFrame({"l": ["l1", "l2", "l3", "l4", "l1", "l2", "l3", "l4"], "g": [0, 0, 0, 0, 1, 1, 1, 1]})

    edges_df = pd.DataFrame(
        {
            "src": [0, 0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7],
            "dst": [0, 1, 3, 0, 3, 0, 1, 3, 0, 4, 5, 7, 4, 7, 4, 5, 7, 4],
            "w": [1.0, 1.0, 3.0, 1.0, 2.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 2.0, 2.0, 3.0, 1.0, 1.0],
            "g": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "l": [
                "el1",
                "el2",
                "el3",
                "el4",
                "el5",
                "el6",
                "el7",
                "el8",
                "el9",
                "el1",
                "el2",
                "el3",
                "el4",
                "el5",
                "el6",
                "el7",
                "el8",
                "el9",
            ],
        }
    )
    gs = graph_from_pandas((edges_df, "g", ("src", "dst"), "w", "l"), (nodes_df, "g", "l"), directed=True)

    assert gs[0][0] == ga and gs[0][1] == ganl and gs[0][2] == gael and gs[1][0] == gb and gs[1][1] == gbnl and gs[1][2] == gbel


def test_networkx():
    """Testing networkx conversion for an edge-dictionary-type initialization object."""

    try:
        import networkx as nx
    except ImportError:
        return

    ga = {0: {0: 1.0, 1: 1.0, 3: 3.0}, 1: {0: 1.0, 3: 2.0}, 2: {0: 2.0, 1: 3.0, 3: 1.0}, 3: {0: 1.0}}

    ganl = {0: "l1", 1: "l2", 2: "l3", 3: "l4"}

    gael = {
        (0, 0): "el1",
        (0, 1): "el2",
        (0, 3): "el3",
        (1, 0): "el4",
        (1, 3): "el5",
        (2, 0): "el6",
        (2, 1): "el7",
        (2, 3): "el8",
        (3, 0): "el9",
    }

    g = nx.DiGraph()

    for n in ganl.keys():
        g.add_node(n, nl=ganl[n])

    for e in gael.keys():
        g.add_edge(e[0], e[1], w=ga[e[0]][e[1]], el=gael[e])

    gs = graph_from_networkx(g, "nl", "el", "w")

    assert isinstance(gs, Graph)
    assert gs.get_edge_dictionary() == ga
    assert gs.get_labels(purpose="dictionary") == ganl
    assert gs.get_labels(label_type="edge", purpose="dictionary") == gael


def test_networkx_simple():
    """Test graph_from_networkx on simple undirected graphs with no labels or weights."""
    try:
        import networkx as nx
    except ImportError:
        return

    g1 = nx.Graph()
    g1.add_nodes_from([0, 1, 2])
    g1.add_edges_from([(0, 1), (1, 2)])

    g2 = nx.Graph()
    g2.add_nodes_from([0, 1, 2])
    g2.add_edges_from([(0, 1), (0, 2), (1, 2)])

    result = graph_from_networkx([g1, g2])

    assert isinstance(result, list) and len(result) == 2
    for grakel_g, nx_g in zip(result, [g1, g2]):
        assert isinstance(grakel_g, Graph)
        ed = grakel_g.get_edge_dictionary()
        for u, v in nx_g.edges():
            assert v in ed[u] and ed[u][v] == 1.0
            assert u in ed[v] and ed[v][u] == 1.0


def test_networkx_node_labeled():
    """Test graph_from_networkx on undirected graphs with node labels."""
    try:
        import networkx as nx
    except ImportError:
        return

    g1 = nx.Graph()
    g1.add_nodes_from([0, 1, 2])
    g1.add_edges_from([(0, 1), (1, 2)])
    nx.set_node_attributes(g1, {0: "a", 1: "b", 2: "a"}, "label")

    g2 = nx.Graph()
    g2.add_nodes_from([0, 1, 2])
    g2.add_edges_from([(0, 1), (0, 2), (1, 2)])
    nx.set_node_attributes(g2, {0: "a", 1: "b", 2: "c"}, "label")

    result = graph_from_networkx([g1, g2], node_labels_tag="label")

    assert isinstance(result, list) and len(result) == 2
    for grakel_g, nx_g in zip(result, [g1, g2]):
        assert isinstance(grakel_g, Graph)
        nl = grakel_g.get_labels(purpose="dictionary")
        for node in nx_g.nodes():
            assert nl[node] == nx_g.nodes[node]["label"]


def test_networkx_node_attributed():
    """Test graph_from_networkx on undirected graphs with numpy array node attributes."""
    try:
        import networkx as nx
    except ImportError:
        return

    import numpy as np

    g1 = nx.Graph()
    g1.add_nodes_from([0, 1, 2])
    g1.add_edges_from([(0, 1), (1, 2)])
    nx.set_node_attributes(g1, {0: np.array([1.1, 0.8]), 1: np.array([0.2, -0.3]), 2: np.array([0.9, 1.0])}, "attributes")

    g2 = nx.Graph()
    g2.add_nodes_from([0, 1, 2])
    g2.add_edges_from([(0, 1), (0, 2), (1, 2)])
    nx.set_node_attributes(g2, {0: np.array([1.8, 0.5]), 1: np.array([-0.1, 0.2]), 2: np.array([2.3, 1.2])}, "attributes")

    result = graph_from_networkx([g1, g2], node_labels_tag="attributes")

    assert isinstance(result, list) and len(result) == 2
    for grakel_g, nx_g in zip(result, [g1, g2]):
        assert isinstance(grakel_g, Graph)
        nl = grakel_g.get_labels(purpose="dictionary")
        for node in nx_g.nodes():
            np.testing.assert_array_equal(nl[node], nx_g.nodes[node]["attributes"])


def test_csv():
    """Testing Graph object consistency for an edge-dictionary-type initialization object."""

    ga = {0: {0: 1.0, 1: 1.0, 3: 3.0}, 1: {0: 1.0, 3: 2.0}, 2: {0: 2.0, 1: 3.0, 3: 1.0}, 3: {0: 1.0}}

    ganl = {0: "l1", 1: "l2", 2: "l3", 3: "l4"}

    gael = {
        (0, 0): "el1",
        (0, 1): "el2",
        (0, 3): "el3",
        (1, 0): "el4",
        (1, 3): "el5",
        (2, 0): "el6",
        (2, 1): "el7",
        (2, 3): "el8",
        (3, 0): "el9",
    }

    node_files = ["temp1_n.csv", "temp2_n.csv", "temp3_n.csv"]
    edge_files = ["temp1_e.csv", "temp2_e.csv", "temp3_e.csv"]
    for nf, ef in zip(node_files, edge_files):
        with open(nf, "w+") as f:
            for k in ganl.keys():
                print(str(k) + "," + ganl[k], file=f)
        with open(ef, "w+") as f:
            for k in gael.keys():
                print(",".join(map(str, [k[0], k[1], ga[k[0]][k[1]], gael[k]])), file=f)

    graphs = list(
        graph_from_csv(edge_files=(edge_files, True, False), node_files=(node_files, False), index_type=int, directed=True)
    )

    for f in node_files + edge_files:
        os.remove(f)

    assert all(gs[0] == ga and gs[1] == ganl and gs[2] == gael for gs in graphs)


def test_KM_Kfold():
    """Testing KFold execution on wl, rw, sp input."""

    def load_mutag():
        try:
            dataset = fetch_dataset("MUTAG", with_classes=True, verbose=verbose)
        except Exception:
            # Offline testing
            warn("MUTAG could not be downloaded: using an offline version..")
            cwd = os.getcwd()
            os.chdir(os.path.join(os.path.dirname(__file__), "data"))
            dataset = read_data("MUTAG", with_classes=True)
            os.chdir(cwd)
        return dataset.data, dataset.target

    # Input
    X, y = load_mutag()
    K = [
        ShortestPath(normalize=True).fit_transform(X),
        [
            WeisfeilerLehman(base_graph_kernel=VertexHistogram, n_iter=3, normalize=True).fit_transform(X),
            WeisfeilerLehman(base_graph_kernel=VertexHistogram, n_iter=5, normalize=True).fit_transform(X),
        ],
    ]

    # Parametrization
    n_splits = 10
    rs = RandomState(42)
    n_iter = 10
    C_grid = list(((10.0 ** arange(-7, 7, 2)) / len(y)).tolist())
    scoring = "accuracy"

    # Execute Kfold
    cross_validate_Kfold_SVM(
        K, y, n_iter=n_iter, n_splits=n_splits, C_grid=C_grid, random_state=rs, scoring=scoring, fold_reduce=None
    )


def test_torch_geometric():
    """Testing Graph object consistency for a torch geometric initialization object."""
    import numpy as np

    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
    except ImportError:
        return

    # Single graph
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.float)
    edge_attr = torch.tensor([[-1.2, 2.5], [0.4, 1.8], [-2.4, 5.1], [0.4, -2.2]], dtype=torch.float)
    y = torch.tensor([0])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    d = graph_from_torch_geometric(data, node_one_hot=False)
    gs, y = d["graph"], d["y"]
    gs.desired_format("adjacency")

    edge_index = np.array(np.nonzero(gs.adjacency_matrix))
    edge_index = torch.from_numpy(edge_index).float()

    x = np.array([gs.node_labels[n] for n in gs.node_labels])
    x = torch.from_numpy(x)

    edge_attr = np.array([gs.edge_labels[e] for e in gs.edge_labels])
    edge_attr = torch.from_numpy(edge_attr)

    label_f = data.y.item() == y
    edge_f = torch.norm(data.edge_index.float() - edge_index).item() == 0.0
    node_f = torch.norm(data.x - x).item() == 0.0
    edge_attr_f = torch.norm(data.edge_attr - edge_attr).item() == 0.0

    assert label_f and edge_f and node_f and edge_attr_f

    # Batch of graphs
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.float)
    edge_attr = torch.tensor([[-1.2, 2.5], [0.4, 1.8], [-2.4, 5.1], [0.4, -2.2]], dtype=torch.float)
    y = torch.tensor([0])
    g1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long)
    x = torch.tensor([[0, 1], [1, 0], [1, 0]], dtype=torch.float)
    edge_attr = torch.tensor([[-0.4, -4.2], [0.2, -1.5], [2.3, -2.1], [1.8, -0.1], [-3.5, -0.2], [-2.8, 0.1]], dtype=torch.float)

    y = torch.tensor([1])
    g2 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    loader = DataLoader([g1, g2], batch_size=2, shuffle=False)
    for data in loader:
        d = graph_from_torch_geometric(data, node_one_hot=False, edge_one_hot=False)

    gs, y = d["graph"], d["y"]
    gs[0].desired_format("adjacency")
    gs[1].desired_format("adjacency")

    edge_index = np.array(np.nonzero(gs[0].adjacency_matrix))
    edge_index = torch.from_numpy(edge_index).float()

    x = np.array([gs[0].node_labels[n] for n in gs[0].node_labels])
    x = torch.from_numpy(x)

    edge_attr = np.array([gs[0].edge_labels[e] for e in gs[0].edge_labels])
    edge_attr = torch.from_numpy(edge_attr)

    label_f = g1.y.item() == y[0]
    edge_f = torch.norm(g1.edge_index.float() - edge_index).item() == 0.0
    node_f = torch.norm(g1.x - x).item() == 0.0
    edge_attr_f = torch.norm(g1.edge_attr - edge_attr).item() == 0.0

    assert label_f and edge_f and node_f and edge_attr_f

    edge_index = np.array(np.nonzero(gs[1].adjacency_matrix))
    edge_index = torch.from_numpy(edge_index).float()

    x = np.array([gs[1].node_labels[n] for n in gs[1].node_labels])
    x = torch.from_numpy(x)

    edge_attr = np.array([gs[1].edge_labels[e] for e in gs[1].edge_labels])
    edge_attr = torch.from_numpy(edge_attr)

    label_f = g2.y.item() == y[1]
    edge_f = torch.norm(g2.edge_index.float() - edge_index).item() == 0.0
    node_f = torch.norm(g2.x - x).item() == 0.0
    edge_attr_f = torch.norm(g2.edge_attr - edge_attr).item() == 0.0

    assert label_f and edge_f and node_f and edge_attr_f


if __name__ == "__main__":
    test_pandas()
    test_networkx()
    test_csv()
    test_KM_Kfold()
    test_torch_geometric()
