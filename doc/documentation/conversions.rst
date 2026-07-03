.. _conversions:

=============================
Converting Between Formats
=============================

*GraKeLX* provides utility functions to convert graphs between different formats.
These functions live in the :mod:`grakelx.utils` module and enable interoperability
with popular graph libraries such as NetworkX, as well as file formats like CSV
and Pandas DataFrames. This section covers each conversion function with
concrete examples.

.. note::

   To use the NetworkX and Pandas conversion functions you need the ``io`` extras:

   .. code-block:: bash

      pip install grakelx[io]

   For the PyTorch Geometric converter, install the ``torch`` extras:

   .. code-block:: bash

      pip install grakelx[torch]


Converting from NetworkX to GraKeLX
------------------------------------

:func:`graph_from_networkx` converts a NetworkX graph (or an iterable thereof)
into a :class:`~grakelx.Graph` object (or a list).

.. code-block:: python

   >>> import networkx as nx
   >>> from grakelx.utils import graph_from_networkx
   >>>
   >>> nx_g = nx.Graph()
   >>> nx_g.add_node(1, label="A")
   >>> nx_g.add_node(2, label="B")
   >>> nx_g.add_edge(1, 2, weight=1.0)
   >>>
   >>> gk_g = graph_from_networkx(nx_g, node_labels_tag="label", edge_weight_tag="weight")

.. autosummary::
   :toctree: ../generated/
   :template: function.rst

   grakelx.utils.graph_from_networkx

**Parameters in brief:**

* ``node_labels_tag`` — name of the NetworkX node attribute to use as node label.
* ``edge_labels_tag`` — name of the NetworkX edge attribute to use as edge label.
* ``edge_weight_tag`` — name of the NetworkX edge attribute to use as edge weight
  (default ``None``, meaning weight = 1.0).
* ``val_node_labels`` / ``val_edge_labels`` — fallback value when a node/edge
  is missing the corresponding attribute.


Converting from GraKeLX to NetworkX (New in 0.1.13)
----------------------------------------------------

:func:`networkx_from_graph` is the inverse of :func:`graph_from_networkx`.
It converts a :class:`~grakelx.Graph` (or an iterable) into a NetworkX graph
(or a list).

.. code-block:: python

   >>> from grakelx import Graph
   >>> from grakelx.utils import networkx_from_graph
   >>>
   >>> gk_g = Graph(initialization_object=[[0, 1], [1, 0]],
   ...              node_labels={0: "X", 1: "Y"})
   >>> nx_g = networkx_from_graph(gk_g, node_labels_tag="label")
   >>> nx_g.nodes(data=True)
   NodeDataView({0: {'label': 'X'}, 1: {'label': 'Y'}})

.. autosummary::
   :toctree: ../generated/
   :template: function.rst

   grakelx.utils.networkx_from_graph

**Parameters in brief:**

* ``node_labels_tag`` — writes node labels as a NetworkX node attribute with
  this name (``None`` = skip).
* ``edge_labels_tag`` — writes edge labels as a NetworkX edge attribute
  (``None`` = skip).
* ``edge_weight_tag`` — writes edge weight as a NetworkX edge attribute
  (default ``"weight"``; ``None`` = skip).
* ``create_using`` — NetworkX graph class to use (e.g. ``nx.DiGraph``).
  Defaults to an undirected :class:`nx.Graph`.


Converting from Pandas DataFrames
----------------------------------

:func:`graph_from_pandas` builds graph objects from one or more
Pandas DataFrames. The expected format is one DataFrame per graph,
with columns ``from``, ``to`` and optionally ``weight``, ``node_label``,
and ``edge_label``.

.. autosummary::
   :toctree: ../generated/
   :template: function.rst

   grakelx.utils.graph_from_pandas


Converting from CSV Files
--------------------------

:func:`graph_from_csv` is useful when your data is stored as CSV files on
disk — one edge per row, one file per graph (or one file containing all
graphs with a ``graph_id`` column).

.. autosummary::
   :toctree: ../generated/
   :template: function.rst

   grakelx.utils.graph_from_csv


Converting from PyTorch Geometric
----------------------------------

:func:`graph_from_torch_geometric` converts a ``torch_geometric.data.Data``
or ``torch_geometric.data.Batch`` object into a dictionary of GraKeLX graphs
and labels.

.. code-block:: python

   >>> import torch
   >>> from torch_geometric.data import Data
   >>> from grakelx.utils import graph_from_torch_geometric
   >>>
   >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
   >>> x = torch.tensor([[1], [2], [3]], dtype=torch.float)
   >>> data = Data(x=x, edge_index=edge_index)
   >>> result = graph_from_torch_geometric(data, node_one_hot=False)
   >>> grakel_graphs = result["graph"]
   >>> labels = result.get("y", None)

.. autosummary::
   :toctree: ../generated/
   :template: function.rst

   grakelx.utils.graph_from_torch_geometric

**Parameters in brief:**

* ``node_one_hot`` / ``edge_one_hot`` — whether to treat ``data.x`` /
  ``data.edge_attr`` as one-hot encoded vectors.
* ``ignore_y`` — if ``True``, omit the ``y`` key from the returned dict.
