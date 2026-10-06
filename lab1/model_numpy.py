import numpy as np  # імпортуємо NumPy для роботи з масивами та числовими обчисленнями

# 1. створюємо ваги та bias
def initialize_parameters():
    rng = np.random.default_rng(0)  # створюємо генератор випадкових чисел із seed=0

    W1 = rng.normal(  # генеруємо ваги першого шару з нормального розподілу
        loc=0.0,  # середнє значення розподілу дорівнює 0
        scale=np.sqrt(2.0 / 4.0),  # стандартне відхилення за He initialization
        size=(4, 8),  # 4 входи та 8 нейронів прихованого шару
    ).astype(np.float64)  # переводимо ваги у тип float64

    b1 = np.zeros(8, dtype=np.float64)  # створюємо 8 нульових bias для першого шару

    W2 = rng.normal(  # генеруємо ваги другого шару з нормального розподілу
        loc=0.0,  # середнє значення розподілу дорівнює 0
        scale=np.sqrt(2.0 / (8.0 + 3.0)),  # стандартне відхилення за Xavier initialization
        size=(8, 3),  # 8 входів із прихованого шару та 3 вихідні класи
    ).astype(np.float64)  # переводимо ваги у тип float64

    b2 = np.zeros(3, dtype=np.float64)  # створюємо 3 нульових bias для вихідного шару

    return W1, b1, W2, b2  # повертаємо всі параметри мережі

# 2. прямий прохід
def forward(X, W1, b1, W2, b2):
    assert X.ndim == 2  # перевіряємо, що X є двовимірною матрицею
    assert X.shape[1] == 4  # перевіряємо, що кожен об’єкт має 4 ознаки

    assert W1.shape == (4, 8)  # перевіряємо форму ваг першого шару
    assert b1.shape == (8,)  # перевіряємо форму bias першого шару
    assert W2.shape == (8, 3)  # перевіряємо форму ваг другого шару
    assert b2.shape == (3,)  # перевіряємо форму bias другого шару

    Z1 = X @ W1 + b1  # обчислюємо лінійний вихід першого шару
    A1 = np.maximum(0.0, Z1)  # застосовуємо ReLU до виходу першого шару
    Z2 = A1 @ W2 + b2  # обчислюємо logits вихідного шару

    assert Z1.shape == (X.shape[0], 8)  # перевіряємо форму Z1
    assert A1.shape == (X.shape[0], 8)  # перевіряємо форму активацій A1
    assert Z2.shape == (X.shape[0], 3)  # перевіряємо, що є 3 logits на об’єкт

    cache = {  # зберігаємо проміжні значення для backward
        "X": X,  # вхідні дані потрібні для градієнта W1
        "Z1": Z1,  # потрібне для похідної ReLU
        "A1": A1,  # потрібне для градієнта W2
        "Z2": Z2,  # зберігаємо logits вихідного шару
    }

    return Z2, cache  # повертаємо logits і збережені проміжні значення

# 3. рахуємо loss та ймовірності
def cross_entropy_loss(logits, y):
    assert logits.ndim == 2  # перевіряємо, що logits є матрицею
    assert logits.shape[1] == 3  # перевіряємо, що є 3 logits для трьох класів

    assert y.ndim == 1  # перевіряємо, що y є вектором класів
    assert y.shape[0] == logits.shape[0]  # перевіряємо однакову кількість об’єктів

    assert np.issubdtype(y.dtype, np.integer)  # перевіряємо, що класи є цілими числами
    assert np.all((y >= 0) & (y < 3))  # перевіряємо, що класи мають значення 0, 1 або 2

    max_logits = np.max(
        logits,
        axis=1,
        keepdims=True,
    )  # знаходимо максимальний logit кожного об’єкта для чисельної стабільності

    shifted_logits = logits - max_logits  # зсуваємо logits, щоб уникнути переповнення exp

    log_sum_exp = np.log(  # обчислюємо log(sum(exp(...))) для кожного об’єкта
        np.sum(
            np.exp(shifted_logits),  # обчислюємо експоненти стабілізованих logits
            axis=1,
            keepdims=True,
        )
    )

    log_probs = shifted_logits - log_sum_exp  # обчислюємо log-softmax

    n = logits.shape[0]  # отримуємо кількість об’єктів у вибірці

    loss = -np.mean(  # обчислюємо середню cross-entropy loss
        log_probs[np.arange(n), y]  # беремо log-ймовірність правильного класу кожного об’єкта
    )

    probs = np.exp(log_probs)  # перетворюємо log-ймовірності у звичайні ймовірності

    assert np.allclose(probs.sum(axis=1), 1.0)  # перевіряємо, що ймовірності сумуються до 1
    assert probs.shape == logits.shape  # перевіряємо форму матриці ймовірностей
    assert np.all(np.isfinite(log_probs))  # перевіряємо відсутність inf і NaN
    assert np.all(np.isfinite(probs))  # перевіряємо відсутність inf і NaN
    assert np.isfinite(loss)  # перевіряємо, що loss є скінченним числом

    return loss, probs  # повертаємо loss і ймовірності класів

# 4. вручну рахуємо градієнти
def backward(probs, y, cache, W2, omit_batch_mean=False):
    X = cache["X"]  # беремо вхідні дані з forward
    Z1 = cache["Z1"]  # беремо pre-activation першого шару
    A1 = cache["A1"]  # беремо активації прихованого шару

    N = X.shape[0]  # кількість об’єктів у batch

    assert probs.shape == (N, 3)  # перевіряємо форму ймовірностей
    assert y.shape == (N,)  # перевіряємо форму вектора класів
    assert W2.shape == (8, 3)  # перевіряємо форму ваг другого шару

    Y = np.zeros_like(probs)  # створюємо нульову one-hot матрицю класів
    Y[np.arange(N), y] = 1.0  # ставимо 1 у позиції правильного класу кожного об’єкта

    assert Y.shape == probs.shape  # перевіряємо форму one-hot матриці

    delta2 = probs - Y  # обчислюємо градієнт за logits без усереднення

    if not omit_batch_mean:  # якщо не запускаємо навмисно помилковий режим
        delta2 = delta2 / N  # ділимо на N, бо loss є середнім по batch

    assert delta2.shape == probs.shape  # перевіряємо форму градієнта за logits

    dW2 = A1.T @ delta2  # обчислюємо градієнт loss за вагами W2
    db2 = delta2.sum(axis=0)  # обчислюємо градієнт loss за bias b2

    assert dW2.shape == W2.shape  # перевіряємо, що dW2 має форму W2
    assert db2.shape == (3,)  # перевіряємо форму db2

    dA1 = delta2 @ W2.T  # передаємо градієнт назад до прихованого шару
    assert dA1.shape == A1.shape  # перевіряємо форму градієнта за A1

    delta1 = dA1 * (Z1 > 0)  # множимо на похідну ReLU: 1 для Z1>0, інакше 0
    assert delta1.shape == Z1.shape  # перевіряємо форму delta1

    dW1 = X.T @ delta1  # обчислюємо градієнт loss за вагами W1
    db1 = delta1.sum(axis=0)  # обчислюємо градієнт loss за bias b1

    assert dW1.shape == (4, 8)  # перевіряємо форму dW1
    assert db1.shape == (8,)  # перевіряємо форму db1

    return dW1, db1, dW2, db2  # повертаємо всі ручно обчислені градієнти