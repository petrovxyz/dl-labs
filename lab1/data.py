import numpy as np
from sklearn.datasets import load_iris

N_TRAIN_PER_CLASS = 35


def load_and_prepare_iris():
    """Iris: стратифікований поділ 35/15 на клас і стандартизація за train."""
    iris = load_iris()

    X = iris.data.astype(np.float64)
    y = iris.target

    rng = np.random.default_rng(0)

    train_indices = []
    test_indices = []

    for class_id in (0, 1, 2):
        class_indices = np.flatnonzero(y == class_id)
        rng.shuffle(class_indices)

        train_indices.extend(class_indices[:N_TRAIN_PER_CLASS])
        test_indices.extend(class_indices[N_TRAIN_PER_CLASS:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # статистики лише з навчальної вибірки, ddof=0
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test
