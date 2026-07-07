import numpy as np


def sec_to_time(sec):
    """Print time in a correct format."""
    dt = list()
    days = int(sec // 86400)
    if days > 0:
        sec -= 86400 * days
        dt.append(str(days) + " d")
    hrs = int(sec // 3600)
    if hrs > 0:
        sec -= 3600 * hrs
        dt.append(str(hrs) + " h")
    mins = int(sec // 60)
    if mins > 0:
        sec -= 60 * mins
        dt.append(str(mins) + " m")
    if sec > 0:
        dt.append(str(round(sec, 2)) + " s")
    return " ".join(dt)


def run_kfold_benchmark(name, kernel, dataset, C_grid, n_splits=10, random_state=42, full=False):
    """Run 10-fold cross-validation with SVM grid search and print results."""
    from time import time

    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold
    from tqdm import tqdm

    G, y = dataset.data, dataset.target

    stats = {"acc": list(), "time": list()}

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    niter = kf.get_n_splits(y)

    for k, (train_index, test_index) in tqdm(enumerate(kf.split(G, y)), total=niter):
        tri = train_index.tolist()
        tei = test_index.tolist()

        G_train, G_test = list(), list()
        y_train, y_test = list(), list()
        for i, (g, t) in enumerate(zip(G, y)):
            if len(tri) and i == tri[0]:
                G_train.append(g)
                y_train.append(t)
                tri.pop(0)
            elif len(tei) and i == tei[0]:
                G_test.append(g)
                y_test.append(t)
                tei.pop(0)

        start = time()

        if full:
            K = kernel.fit_transform(G)
            K_train = K[train_index, :][:, train_index]
            K_test = K[test_index, :][:, train_index]
        else:
            K_train = kernel.fit_transform(G_train)
            K_test = kernel.transform(G_test)
        end = time()

        acc = 0
        for c in C_grid:
            clf = svm.SVC(kernel="precomputed", C=c)
            clf.fit(K_train, y_train)
            y_pred = clf.predict(K_test)
            acc = max(acc, accuracy_score(y_test, y_pred))

        stats["acc"].append(acc)
        stats["time"].append(end - start)

    print("Mean values of", niter, "iterations:")
    print(name, "> Accuracy:", str(round(np.mean(stats["acc"]) * 100, 2)), "% | Took:", sec_to_time(np.mean(stats["time"])))
