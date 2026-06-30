from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from grakelx import Graph
from grakelx.utils._type_hints import Label

if TYPE_CHECKING:
    import networkx as nx


def networkx_from_graph(
    X: Graph | Iterable[Graph],
    node_labels_tag: str | None = None,
    edge_labels_tag: str | None = None,
    edge_weight_tag: str = "weight",
    create_using: type[nx.Graph] | None = None,
) -> nx.Graph | list[nx.Graph]:
    """Transform grakel.Graph objects to networkx graph objects.

    Inverse operation of :func:`graph_from_networkx`. Iterates the
    ``edge_dictionary`` of each input graph and writes nodes, edges and
    (optionally) labels and weights as networkx node/edge attributes.

    Parameters
    ----------
        X : grakel.Graph or Iterable[grakel.Graph]
            A GraKeL graph or an iterable of GraKeL graphs.

        node_labels_tag : str or None, default=None
            If provided, the dictionary labels of every node are written as
            a node attribute with this tag. If ``None`` no node attribute
            is written.

        edge_labels_tag : str or None, default=None
            If provided, the dictionary labels of every edge are written as
            an edge attribute with this tag. If ``None`` no edge attribute
            is written.

        edge_weight_tag : str or None, default="weight"
            If provided, the weight of every edge is written as an edge
            attribute with this tag. The default ``"weight"`` follows the
            NetworkX convention. Set to ``None`` to skip writing weights.

        create_using : type[networkx.Graph] or None, default=None
            NetworkX graph class used to construct each output graph.
            Defaults to ``None`` (an undirected ``networkx.Graph``). Any
            subclass is accepted, e.g. ``nx.DiGraph`` or ``nx.MultiGraph``.
            GraKeL graphs are inherently non-directed, so when using a
            directed container the symmetric entries of the edge
            dictionary are merged into a single canonical edge
            ``(min(u, v), max(u, v))``.

    Returns
    -------
        converted_graphs : networkx.Graph or list[networkx.Graph]
            Returns a networkx.Graph if ``X`` is a single graph and a list
            of networkx.Graph objects otherwise.

    """
    import networkx as nx

    if not (node_labels_tag is None or isinstance(node_labels_tag, str)):
        raise ValueError("node_labels_tag must be a str indicating the tag of the labels inside nodes or None")
    if not (edge_labels_tag is None or isinstance(edge_labels_tag, str)):
        raise ValueError("edge_labels_tag must be a str indicating the tag of the labels inside edges or None")
    if not (edge_weight_tag is None or isinstance(edge_weight_tag, str)):
        raise ValueError("edge_weight_tag must be a str indicating the tag of the weights inside edges or None")
    if create_using is not None and not (isinstance(create_using, type) and issubclass(create_using, nx.Graph)):
        raise ValueError("create_using must be a subclass of networkx.Graph or None")

    container_cls = nx.Graph if create_using is None else create_using

    if isinstance(X, Graph):
        graphs = [X]
        return_single_graph = True
    elif isinstance(X, Iterable):
        graphs = list(X)
        return_single_graph = False
    else:
        raise ValueError("X must be a grakel.Graph or an iterable of grakel.Graph objects")

    converted_graphs = []
    for G in graphs:
        if not isinstance(G, Graph):
            raise ValueError("each element of X must be a grakel.Graph")

        nx_g = container_cls()

        nl = G.get_labels(purpose="dictionary", return_none=True) if node_labels_tag is not None else None
        el = G.get_labels(label_type="edge", purpose="dictionary", return_none=True) if edge_labels_tag is not None else None

        for u in G.get_vertices(purpose="dictionary"):
            attrs = {}
            if nl is not None and u in nl:
                attrs[node_labels_tag] = nl[u]
            nx_g.add_node(u, **attrs)

        edge_dictionary = G.get_edge_dictionary()
        if not isinstance(nx_g, (nx.DiGraph, nx.MultiDiGraph)) and not isinstance(nx_g, (nx.MultiGraph,)):
            seen = set()
            for u, nbrs in edge_dictionary.items():
                for v, w in nbrs.items():
                    key = (u, v) if u <= v else (v, u)
                    if key in seen:
                        continue
                    seen.add(key)
                    attrs = {}
                    if edge_weight_tag is not None:
                        attrs[edge_weight_tag] = w
                    if el is not None and key in el:
                        attrs[edge_labels_tag] = el[key]
                    nx_g.add_edge(key[0], key[1], **attrs)
        else:
            for u, nbrs in edge_dictionary.items():
                for v, w in nbrs.items():
                    attrs = {}
                    if edge_weight_tag is not None:
                        attrs[edge_weight_tag] = w
                    if el is not None and (u, v) in el:
                        attrs[edge_labels_tag] = el[(u, v)]
                    nx_g.add_edge(u, v, **attrs)

        converted_graphs.append(nx_g)

    if return_single_graph:
        return converted_graphs[0]
    return converted_graphs


def graph_from_networkx(
    X: nx.Graph | Iterable[nx.Graph],
    node_labels_tag: str | None = None,
    edge_labels_tag: str | None = None,
    edge_weight_tag: str | None = None,
    val_node_labels: Label | None = None,
    val_edge_labels: Label | None = None,
) -> Graph | list[Graph]:
    """Transform networkx objects to grakel.Graph objects.

    A function for helping a user that has a collection of graphs in networkx to use grakel.

    Parameters
    ----------
        X : networkx.Graph or Iterable[networkx.Graph]
            A networkx graph or an iterable of networkx graphs.
        node_labels_tag : str or None
            Define where to search for labels of nodes, inside the `node` attribute of each graph.
            If None no labels are assigned.

        edge_labels_tag : str or None
            Define where to search for labels of edges, inside the `edge` attribute of each graph.
            If None no labels are assigned.

        edge_weight_tag : str or None
            Define where to search for weights inside the `edge` attribute of each graph.
            If None 1.0 weights are assigned.

        val_node_labels : Label, default=None
            Sets constant value to all nodes labels of a graph if missing.
            See ``grakelx.utils._type_hints.Label`` for the accepted values.

        val_edge_labels : Label, default=None
            Sets constant value to all edges labels of a graph if missing.
            See ``grakelx.utils._type_hints.Label`` for the accepted values.


    Returns
    -------
        converted_graphs : grakel.Graph or list[grakel.Graph]
            Returns a grakel.Graph if ``X`` is a single graph and a list
            of grakel.Graph objects otherwise.

    """
    import networkx as nx

    if node_labels_tag is None:

        def nodel_init():
            if val_node_labels is None:
                return None
            else:
                return dict()

        def nodel_put(nl, u, d):
            if val_node_labels is not None:
                nl[u] = val_node_labels

    elif isinstance(node_labels_tag, str):

        def nodel_init():
            return dict()

        def nodel_put(nl, u, d):
            attrs = d[u]  # G.nodes[u] -> node attribute dict
            if node_labels_tag in attrs:
                nl[u] = attrs[node_labels_tag]
            elif val_node_labels is not None:
                nl[u] = val_node_labels
            else:
                raise ValueError("could not find node label tag '{0}' for node {1}".format(node_labels_tag, u))

    else:
        raise ValueError("node_labels_tag must be a str indicating the tag of the labels inside nodes or None")

    if edge_labels_tag is None:

        def edgel_init():
            if val_edge_labels is None:
                return None
            else:
                return dict()

        def edgel_put(el, u, d):
            if val_edge_labels is not None:
                el[u] = val_edge_labels

    elif isinstance(edge_labels_tag, str):

        def edgel_init():
            return dict()

        def edgel_put(el, u, d):
            # d is the adjacency dict G[src]; u is (src, dst); d[u[1]] is edge attr dict
            attrs = d[u[1]]
            if edge_labels_tag in attrs:
                el[u] = attrs[edge_labels_tag]
            elif val_edge_labels is not None:
                el[u] = val_edge_labels
            else:
                raise ValueError("could not find edge label tag '{0}' for edge {1}".format(edge_labels_tag, u))

    else:
        raise ValueError("edge_labels_tag must be a str indicating the tag of the labels inside edges or None")

    if edge_weight_tag is None:

        def get_weight(*args):
            return 1.0

    elif isinstance(edge_weight_tag, str):

        def get_weight(d, e):
            # d is adjacency dict G[src]; e is (src, dst); d[e[1]] is edge attr dict
            attrs = d[e[1]]
            if edge_weight_tag in attrs:
                return attrs[edge_weight_tag]
            return 1.0

    else:
        raise ValueError("weight_labels_tag must be a str indicating  tag of the labels inside edges or None (1.0)")

    if isinstance(X, nx.Graph):
        graphs = [X]
        return_single_graph = True
    elif isinstance(X, Iterable):
        graphs = list(X)
        return_single_graph = False
    else:
        raise ValueError("X must be a networkx graph or an iterable of networkx graphs")

    converted_graphs = []
    for G in graphs:
        if not isinstance(G, nx.Graph):
            raise ValueError("each element of X must be a networkx graph")

        graph_object = dict()
        nl = nodel_init()
        el = edgel_init()
        for u in G.nodes():
            graph_object[u] = dict()
            nodel_put(nl, u, G.nodes)  # G.nodes[u] -> node attribute dict
            for v in G.neighbors(u):
                adj = G[u]  # adjacency dict for u: adj[v] = edge attr dict
                graph_object[u][v] = get_weight(adj, (u, v))
                edgel_put(el, (u, v), adj)

        converted_graphs.append(Graph(graph_object, nl, el))

    if return_single_graph:
        return converted_graphs[0]
    return converted_graphs
