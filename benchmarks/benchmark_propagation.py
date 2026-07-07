"""Benchmark for the Propagation kernel family.

Usage:
    python benchmarks/benchmark_propagation.py [--dataset DATASET] [--attr] [--tmax TMAX] [--w W] [--full]
"""

import argparse

import numpy as np

from benchmarks.utils import run_kfold_benchmark
from grakelx.datasets import fetch_dataset
from grakelx.kernels import Propagation, PropagationAttr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark for Propagation kernels")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--attr", action="store_true")
    parser.add_argument("--tmax", type=int, default=5)
    parser.add_argument("--w", type=float, default=0.01)

    args = parser.parse_args()
    has_attributes = bool(args.attr)
    dataset_name = args.dataset if args.dataset else ("BZR" if has_attributes else "MUTAG")

    dataset = fetch_dataset(dataset_name, with_classes=True, prefer_attr_nodes=has_attributes, verbose=True)
    C_grid = (10.0 ** np.arange(-7, 7, 2) / len(dataset.data)).tolist()

    if has_attributes:
        kernel = PropagationAttr(M="L1", t_max=args.tmax, w=args.w, normalize=True)
    else:
        kernel = Propagation(M="H", t_max=args.tmax, w=args.w, normalize=True)

    run_kfold_benchmark("Propagation" + ("Attr" if has_attributes else ""), kernel, dataset, C_grid, full=bool(args.full))
