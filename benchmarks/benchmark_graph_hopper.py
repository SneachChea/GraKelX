"""Benchmark for the GraphHopper kernel.

Usage:
    python benchmarks/benchmark_graph_hopper.py [--dataset DATASET] [--linear] [--gaussian [MU]] [--bridge] [--full]
"""

import argparse

import numpy as np

from benchmarks.utils import run_kfold_benchmark
from grakelx.datasets import fetch_dataset
from grakelx.kernels import GraphHopper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark for GraphHopper kernel")
    parser.add_argument("--dataset", type=str, default="BZR")
    parser.add_argument("--full", action="store_true")

    mec = parser.add_mutually_exclusive_group()
    mec.add_argument("--linear", action="store_true")
    mec.add_argument("--gaussian", nargs="?", type=str, const="1", default=None)
    mec.add_argument("--bridge", action="store_true")

    args = parser.parse_args()

    if args.gaussian is not None:
        kernel_type = ("gaussian", float(args.gaussian))
    elif bool(args.bridge):
        kernel_type = "bridge"
    else:
        kernel_type = "linear"

    dataset = fetch_dataset(args.dataset, with_classes=True, prefer_attr_nodes=True, verbose=True)
    C_grid = (10.0 ** np.arange(-7, 7, 2) / len(dataset.data)).tolist()

    kernel = GraphHopper(normalize=True, kernel_type=kernel_type)

    run_kfold_benchmark("GraphHopper", kernel, dataset, C_grid, full=bool(args.full))
