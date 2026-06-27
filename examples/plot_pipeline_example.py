"""
====================================================
Example of building a graph classification pipeline.
====================================================

Script makes use of :class:`grakelx.ShortestPath`
"""

from __future__ import print_function

print(__doc__)

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from grakelx.datasets import fetch_dataset
from grakelx.kernels import ShortestPath

# Loads the Mutag dataset from:
MUTAG = fetch_dataset("MUTAG", verbose=False)
G, y = MUTAG.data, MUTAG.target

# Values of C parameter of SVM
C_grid = (10.0 ** np.arange(-4, 6, 1) / len(G)).tolist()

# Creates pipeline
estimator = make_pipeline(
    ShortestPath(normalize=True), GridSearchCV(SVC(kernel="precomputed"), dict(C=C_grid), scoring="accuracy", cv=10)
)

# Performs cross-validation and computes accuracy
n_folds = 10
acc = accuracy_score(y, cross_val_predict(estimator, G, y, cv=n_folds))
print("Accuracy:", str(round(acc * 100, 2)) + "%")
