import numpy as np

from model_numpy import forward, cross_entropy_loss

EPS = 1e-6
TOL_NUM = 1e-7


def _central_difference(param, index, X, y, W1, b1, W2, b2, eps):
    """Центральна різниця для param[index] і ознака, що маски ReLU не змінились."""
    original_value = param[index]

    # маска ReLU на початкових параметрах
    _, original_cache = forward(X, W1, b1, W2, b2)
    original_mask = original_cache["Z1"] > 0

    param[index] = original_value + eps
    logits_plus, cache_plus = forward(X, W1, b1, W2, b2)
    loss_plus, _ = cross_entropy_loss(logits_plus, y)

    param[index] = original_value - eps
    logits_minus, cache_minus = forward(X, W1, b1, W2, b2)
    loss_minus, _ = cross_entropy_loss(logits_minus, y)

    param[index] = original_value

    # збурення ±eps не повинні перетинати злам ReLU
    same_mask = np.array_equal(original_mask, cache_plus["Z1"] > 0) and np.array_equal(
        original_mask, cache_minus["Z1"] > 0
    )

    return (loss_plus - loss_minus) / (2.0 * eps), same_mask


def numerical_gradient(param, index, X, y, W1, b1, W2, b2, eps=EPS):
    """Центральна різниця для param[index]; param є одним із W1, b1, W2, b2.

    Змінюється лише цей елемент, після обчислень початкове значення відновлюється.
    """
    gradient, same_mask = _central_difference(param, index, X, y, W1, b1, W2, b2, eps)
    assert same_mask, "збурення перетнуло злам ReLU"
    return gradient


def run_numerical_gradient_checks(X, y, W1, b1, W2, b2, dW1, db1, dW2, db2):
    checks = [
        ("W1[0, 0]", W1, (0, 0), dW1[0, 0]),
        ("b1[0]", b1, 0, db1[0]),
        ("W2[0, 0]", W2, (0, 0), dW2[0, 0]),
        ("b2[0]", b2, 0, db2[0]),
    ]

    results = []

    for name, param, index, manual in checks:
        numerical = numerical_gradient(param, index, X, y, W1, b1, W2, b2)

        assert np.isfinite(manual) and np.isfinite(numerical)

        abs_diff = abs(manual - numerical)

        results.append(
            {
                "parameter": name,
                "manual": float(manual),
                "numerical": float(numerical),
                "abs_diff": float(abs_diff),
                "passed": bool(abs_diff <= TOL_NUM),
            }
        )

    return results


STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10)


def step_sensitivity(X, y, W1, b1, W2, b2, dW1, db1, dW2, db2, steps=STEPS):
    """|g_num − g_manual| для тих самих чотирьох параметрів при різних h.

    Додаткове дослідження; критерій ЛР1 застосовується лише до h = EPS.
    Повертає список (h, [(abs_diff, same_mask) для кожного параметра]).
    """
    targets = [
        (W1, (0, 0), dW1[0, 0]),
        (b1, 0, db1[0]),
        (W2, (0, 0), dW2[0, 0]),
        (b2, 0, db2[0]),
    ]
    table = []
    for h in steps:
        row = []
        for param, index, manual in targets:
            numerical, same_mask = _central_difference(param, index, X, y, W1, b1, W2, b2, h)
            assert np.isfinite(numerical)
            row.append((float(abs(manual - numerical)), bool(same_mask)))
        table.append((h, row))
    return table
