import numpy as np

from model_numpy import forward, cross_entropy_loss

EPS = 1e-6
TOL_NUM = 1e-7


def numerical_gradient(param, index, X, y, W1, b1, W2, b2, eps=EPS):
    """Центральна різниця для param[index]; param є одним із W1, b1, W2, b2.

    Змінюється лише цей елемент, після обчислень початкове значення відновлюється.
    """
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
    assert np.array_equal(original_mask, cache_plus["Z1"] > 0)
    assert np.array_equal(original_mask, cache_minus["Z1"] > 0)

    return (loss_plus - loss_minus) / (2.0 * eps)


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
