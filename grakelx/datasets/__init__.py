"""Import datasets related with graph kernels, from a large collection."""

from grakelx.datasets.base import fetch_dataset, get_dataset_info
from grakelx.datasets.testing import generate_dataset

__all__ = ["get_dataset_info", "fetch_dataset", "generate_dataset"]
