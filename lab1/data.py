import numpy as np  # імпортуємо NumPy для роботи з масивами та числами
from sklearn.datasets import load_iris  # імпортуємо функцію завантаження Iris

N_TRAIN_PER_CLASS = 35  # кількість об’єктів кожного класу для train

def load_and_prepare_iris():
    """Iris: стратифікований поділ 35/15 на клас і стандартизація за train."""

    iris = load_iris()  # завантажуємо датасет Iris

    X = iris.data.astype(np.float64)  # матриця ознак, переводимо у float64
    y = iris.target  # вектор класів: 0, 1, 2

    # default_rng() створює сучасний генератор випадкових чисел NumPy
    # seed — це початкове число для генератора псевдовипадкових чисел
    rng = np.random.default_rng(0)  # фіксований seed = 0 для відтворюваного поділу

    # індекс об’єкта — це його номер у датасеті
    train_indices = []  # індекси об’єктів для train
    test_indices = []  # індекси об’єктів для test

    for class_id in (0, 1, 2):  # окремо обробляємо кожен клас
        # np.flatnonzero() повертає індекси всіх ненульових / True елементів
        # y == class_id створює масив True/False
        class_indices = np.flatnonzero(y == class_id)  # знаходимо індекси об’єктів цього класу
        rng.shuffle(class_indices)  # випадково перемішати об’єкти всередині кожного класу перед поділом на train і test, це робить поділ випадковим
        # без shuffle() ми просто взяли б перші 35 об’єктів класу в train, а останні 15 — у test
        # після перемішування: 
        # - 35 випадкових об’єктів класу → train;
        # - 15 інших → test

        train_indices.extend(class_indices[:N_TRAIN_PER_CLASS])  # перші 35 → train
        test_indices.extend(class_indices[N_TRAIN_PER_CLASS:])  # решта 15 → test

    train_indices = np.array(train_indices)  # список train-індексів → NumPy-масив
    test_indices = np.array(test_indices)  # список test-індексів → NumPy-масив

    X_train, y_train = X[train_indices], y[train_indices]  # формуємо train-вибірку
    X_test, y_test = X[test_indices], y[test_indices]  # формуємо test-вибірку

    mean = X_train.mean(axis=0)  # середнє кожної ознаки тільки за train
    std = X_train.std(axis=0, ddof=0)  # стандартне відхилення кожної ознаки за train
    # чому тільки для train? щоб не використовувати інформацію з test-вибірки під час підготовки моделі
    # а загалом для того, щоб:
    # 1. великі числові значення однієї ознаки не домінували над іншими
    # 2. різні ознаки були приблизно в одному масштабі

    X_train = (X_train - mean) / std  # стандартизуємо train
    X_test = (X_test - mean) / std  # стандартизуємо test тими самими mean і std
    # тобто модель бачить train і test в однаковому масштабі, без витоку інформації з test

    return X_train, y_train, X_test, y_test  # повертаємо готові train і test дані