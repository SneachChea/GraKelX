.. _api_ref:

=============
API Reference
=============

This is the class and function reference of *GraKeL*. In order for the user to understand how to use the package, we suggest he reads :ref:`documentation` section.

:mod:`grakelx.graph`: Graph class with its utility functions
===========================================================

Base Class
----------
.. currentmodule:: grakelx

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Graph


Utility Functions
-----------------
.. currentmodule:: grakelx

.. autosummary::
   :toctree: generated/
   :template: function.rst

   graph.is_adjacency
   graph.is_edge_dictionary
   graph.laplacian
   graph.floyd_warshall

**User guide:** See the :ref:`graph` section for further details.

:mod:`grakelx.graph_kernels`: A kernel decorator
===============================================
.. currentmodule:: grakelx

Graph Kernel (decorator)
------------------------
.. autosummary::
   :toctree: generated/
   :template: class.rst

   grakelx.GraphKernel

**User guide:** See the :ref:`graph_kernel` section for further details.

:mod:`grakelx.kernels`: A collection of graph kernels
====================================================

Kernels
-------

.. currentmodule:: grakelx

.. autosummary::
   :toctree: generated/
   :template: kernel.rst

   Kernel
   RandomWalk
   RandomWalkLabeled
   PyramidMatch
   NeighborhoodHash
   ShortestPath
   ShortestPathAttr
   GraphletSampling
   SubgraphMatching
   WeisfeilerLehman
   HadamardCode
   NeighborhoodSubgraphPairwiseDistance
   LovaszTheta
   SvmTheta
   Propagation
   PropagationAttr
   OddSth
   MultiscaleLaplacian
    VertexHistogram
   EdgeHistogram
   GraphHopper
   CoreFramework
   WeisfeilerLehmanOptimalAssignment

**User guide:** See the :ref:`kernels` section for further details.

:mod:`grakelx.datasets`: Datasets
=================================

Fetch
-----

.. currentmodule:: grakelx.datasets

.. autosummary::
   :toctree: generated/
   :template: function_bib.rst

   fetch_dataset

.. autosummary::
   :toctree: generated/
   :template: function.rst

   get_dataset_info


**User guide:** See the :ref:`datasets` section for further details.


:mod:`grakelx`: Utils
=================================

.. currentmodule:: grakelx

Use a kernel matrix as a transformer
------------------------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   KMTransformer

Cross Validation
----------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   cross_validate_Kfold_SVM

Load from other file formats
----------------------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   graph_from_networkx
   graph_from_pandas
   graph_from_csv

**User guide:** Usefull functions for applying to existing datasets, of other formats.

.. _gd:	https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
