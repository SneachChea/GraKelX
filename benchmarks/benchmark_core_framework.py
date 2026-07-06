"""Benchmark for the CoreFramework kernel.

Usage:
    python benchmarks/benchmark_core_framework.py [--dataset DATASET] [--mc MIN_CORE] [--full]
"""

import argparse

import numpy as np

from benchmarks.utils import run_kfold_benchmark
from grakelx.datasets import fetch_dataset
from grakelx.kernels import CoreFramework, VertexHistogram, WeisfeilerLehman

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark for CoreFramework kernel")
    parser.add_argument("--dataset", type=str, default="MUTAG")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--mc", type=int, default=-1)

    args = parser.parse_args()

    dataset = fetch_dataset(args.dataset, with_classes=True, produce_labels_nodes=True, prefer_attr_nodes=False, verbose=True)
    C_grid = (10.0 ** np.arange(-7, 7, 2) / len(dataset.data)).tolist()

    base_graph_kernel = (WeisfeilerLehman, dict(base_graph_kernel=VertexHistogram))
    kernel = CoreFramework(normalize=True, base_graph_kernel=base_graph_kernel, min_core=args.mc)

    run_kfold_benchmark("Core-Framework/WL/Subtree", kernel, dataset, C_grid, full=bool(args.full))
