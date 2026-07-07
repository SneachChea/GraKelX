"""Tests for the Graph class."""

import warnings

import numpy as np
import numpy.testing as npt

from grakelx.graph import Graph

warnings.filterwarnings("ignore", category=UserWarning)


def test_graph_adjacency():
    """Testing Graph object consistency for an adjacency-type initialization object."""
    X = np.array([[1, 1, 0, 3], [1, 0, 0, 2], [2, 3, 0, 1], [1, 0, 0, 0]])
    labels = {0: "banana", 1: "cherry", 2: "banana", 3: "cherry"}

    g = dict()
    g["auto"] = Graph(X, labels, {}, "auto")
    g["dict"] = Graph(X, labels, {}, "dictionary")
    g["adjc"] = Graph(X, labels, {}, "adjacency")
    g["all"] = Graph(X, labels, {}, "all")

    desired_output_label_group = {"cherry": [1, 3], "banana": [0, 2]}

    for k in g.keys():
        gklg = g[k].get_label_group()
        npt.assert_equal(desired_output_label_group, gklg)

    spm_do = [[0.0, 1.0, float("Inf"), 3.0], [1.0, 0.0, float("Inf"), 2.0], [2.0, 3.0, 0.0, 1.0], [1.0, 2.0, float("Inf"), 0.0]]

    for k in g.keys():
        spm, spl = g[k].build_shortest_path_matrix(algorithm_type="auto")
        npt.assert_array_equal(spm, spm_do)
        npt.assert_equal(spl, labels)


def test_graph_edge_dictionary():
    """Testing Graph object consistency for an edge-dictionary-type initialization object."""
    X = {"a": {"a": 1, "b": 1, "d": 3}, "b": {"a": 1, "d": 2}, "c": {"a": 2, "b": 3, "d": 1}, "d": {"a": 1}}

    labels = {"a": "banana", "b": "cherry", "c": "banana", "d": "cherry"}

    g = dict()
    g["auto"] = Graph(X, labels, {}, "auto")
    g["dict"] = Graph(X, labels, {}, "dictionary")
    g["adjc"] = Graph(X, labels, {}, "adjacency")
    g["all"] = Graph(X, labels, {}, "all")

    desired_output_label_group = {"cherry": {"d", "b"}, "banana": {"a", "c"}}
    desired_output_label_group_idx = {"banana": {0, 2}, "cherry": {1, 3}}

    def proper_dict(x):
        return {key: set(x[key]) for key in x.keys()}

    for k in g.keys():
        gklg = g[k].get_label_group()
        if k == "adjc":
            npt.assert_equal(desired_output_label_group_idx, proper_dict(gklg))
        else:
            npt.assert_equal(desired_output_label_group, proper_dict(gklg))

    spm_do = [[0.0, 1.0, float("Inf"), 3.0], [1.0, 0.0, float("Inf"), 2.0], [2.0, 3.0, 0.0, 1.0], [1.0, 2.0, float("Inf"), 0.0]]

    desired_labels = {0: "banana", 1: "cherry", 2: "banana", 3: "cherry"}

    for k in g.keys():
        spm, spl = g[k].build_shortest_path_matrix(algorithm_type="auto")
        npt.assert_array_equal(spm, spm_do)
        npt.assert_equal(spl, desired_labels)
