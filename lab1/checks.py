import numpy as np  # Імпортуємо NumPy для роботи з масивами та числовими обчисленнями.

from model_numpy import forward, cross_entropy_loss  # Імпортуємо forward і функцію втрат з NumPy-моделі.

EPS = 1e-6  # Крок h для чисельного диференціювання за умовою лабораторної.
TOL_NUM = 1e-7  # Допустима абсолютна похибка чисельної перевірки.


def _central_difference(param, index, X, y, W1, b1, W2, b2, eps):
    """Центральна різниця для одного параметра."""

    original_value = param[index]  # Зберігаємо початкове значення параметра.

    # Маска ReLU на початкових параметрах.
    _, original_cache = forward(X, W1, b1, W2, b2)  # Робимо forward на початкових параметрах.
    original_mask = original_cache["Z1"] > 0  # Запам’ятовуємо, які нейрони ReLU були активними.

    param[index] = original_value + eps  # Збільшуємо лише вибраний параметр на eps.
    logits_plus, cache_plus = forward(X, W1, b1, W2, b2)  # Робимо forward для theta + eps.
    loss_plus, _ = cross_entropy_loss(logits_plus, y)  # Обчислюємо loss для theta + eps.

    param[index] = original_value - eps  # Зменшуємо той самий параметр на eps від початкового значення.
    logits_minus, cache_minus = forward(X, W1, b1, W2, b2)  # Робимо forward для theta - eps.
    loss_minus, _ = cross_entropy_loss(logits_minus, y)  # Обчислюємо loss для theta - eps.

    param[index] = original_value  # Повертаємо параметру його початкове значення.

    # Перевіряємо, що при ±eps не змінилася гілка ReLU.
    same_mask = np.array_equal(
        original_mask,
        cache_plus["Z1"] > 0,
    ) and np.array_equal(
        original_mask,
        cache_minus["Z1"] > 0,
    )

    return (loss_plus - loss_minus) / (2.0 * eps), same_mask  # Повертаємо чисельний градієнт і результат перевірки ReLU.


def numerical_gradient(param, index, X, y, W1, b1, W2, b2, eps=EPS):
    """Обчислюємо чисельний градієнт одного параметра."""

    gradient, same_mask = _central_difference(
        param,
        index,
        X,
        y,
        W1,
        b1,
        W2,
        b2,
        eps,
    )  # Обчислюємо центральну різницю.

    assert same_mask, "збурення перетнуло злам ReLU"  # Перевіряємо, що не перетнули точку зламу ReLU.

    return gradient  # Повертаємо чисельний градієнт.


def run_numerical_gradient_checks(X, y, W1, b1, W2, b2, dW1, db1, dW2, db2):
    checks = [  # Задаємо чотири параметри, які потрібно перевірити.
        ("W1[0, 0]", W1, (0, 0), dW1[0, 0]),  # Перевірка W1[0,0].
        ("b1[0]", b1, 0, db1[0]),  # Перевірка b1[0].
        ("W2[0, 0]", W2, (0, 0), dW2[0, 0]),  # Перевірка W2[0,0].
        ("b2[0]", b2, 0, db2[0]),  # Перевірка b2[0].
    ]

    results = []  # Сюди записуємо результати всіх перевірок.

    for name, param, index, manual in checks:  # Послідовно перевіряємо кожен із чотирьох параметрів.
        numerical = numerical_gradient(
            param,
            index,
            X,
            y,
            W1,
            b1,
            W2,
            b2,
        )  # Обчислюємо чисельний градієнт.

        assert np.isfinite(manual) and np.isfinite(numerical)  # Перевіряємо, що обидва градієнти скінченні.

        abs_diff = abs(manual - numerical)  # Обчислюємо абсолютну різницю між ручним і чисельним градієнтом.

        results.append(  # Додаємо результат перевірки до списку.
            {
                "parameter": name,  # Назва параметра.
                "manual": float(manual),  # Градієнт з ручного backward.
                "numerical": float(numerical),  # Градієнт через центральну різницю.
                "abs_diff": float(abs_diff),  # Абсолютна різниця між ними.
                "passed": bool(abs_diff <= TOL_NUM),  # Чи пройшла перевірка з допуском 1e-7.
            }
        )

    return results  # Повертаємо результати чотирьох перевірок.


STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10)  # Різні значення h для додаткового дослідження.


def step_sensitivity(X, y, W1, b1, W2, b2, dW1, db1, dW2, db2, steps=STEPS):
    """Перевіряємо, як похибка чисельного градієнта залежить від кроку h."""

    targets = [  # Ті самі чотири параметри для перевірки.
        (W1, (0, 0), dW1[0, 0]),  # W1[0,0].
        (b1, 0, db1[0]),  # b1[0].
        (W2, (0, 0), dW2[0, 0]),  # W2[0,0].
        (b2, 0, db2[0]),  # b2[0].
    ]

    table = []  # Сюди записуємо результати для всіх значень h.

    for h in steps:  # Перебираємо різні значення кроку h.
        row = []  # Результати для одного конкретного h.

        for param, index, manual in targets:  # Перевіряємо кожен із чотирьох параметрів.
            numerical, same_mask = _central_difference(
                param,
                index,
                X,
                y,
                W1,
                b1,
                W2,
                b2,
                h,
            )  # Обчислюємо чисельний градієнт із поточним h.

            assert np.isfinite(numerical)  # Перевіряємо, що чисельний градієнт скінченний.

            row.append(
                (
                    float(abs(manual - numerical)),  # Зберігаємо абсолютну похибку.
                    bool(same_mask),  # Зберігаємо, чи не змінилася ReLU-маска.
                )
            )

        table.append((h, row))  # Додаємо результати для цього h у таблицю.

    return table  # Повертаємо похибки для всіх значень h.