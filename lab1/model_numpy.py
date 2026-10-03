import numpy as np

def initialize_parameters():
    rng = np.random.default_rng(0)

    W1 = rng.normal(
        loc=0.0,
        scale=np.sqrt(2.0 / 4.0),
        size=(4, 8),
    ).astype(np.float64)

    b1 = np.zeros(8, dtype=np.float64)

    W2 = rng.normal(
        loc=0.0,
        scale=np.sqrt(2.0 / (8.0 + 3.0)),
        size=(8, 3),
    ).astype(np.float64)

    b2 = np.zeros(3, dtype=np.float64)

    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    assert X.ndim == 2
    assert X.shape[1] == 4

    assert W1.shape == (4, 8)
    assert b1.shape == (8,)
    assert W2.shape == (8, 3)
    assert b2.shape == (3,)

    Z1 = X @ W1 + b1
    A1 = np.maximum(0.0, Z1)
    Z2 = A1 @ W2 + b2

    assert Z1.shape == (X.shape[0], 8)
    assert A1.shape == (X.shape[0], 8)
    assert Z2.shape == (X.shape[0], 3)

    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
    }
    return Z2, cache

def cross_entropy_loss(logits, y):
    assert logits.ndim == 2
    assert logits.shape[1] == 3

    assert y.ndim == 1
    assert y.shape[0] == logits.shape[0]

    assert np.issubdtype(y.dtype, np.integer)
    assert np.all((y >= 0) & (y < 3))

    max_logits = np.max(logits, axis=1, keepdims=True)

    shifted_logits = logits - max_logits

    log_sum_exp = np.log(
        np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    )

    log_probs = shifted_logits - log_sum_exp

    n = logits.shape[0]

    loss = -np.mean(
        log_probs[np.arange(n), y]
    )

    probs = np.exp(log_probs)

    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs.shape == logits.shape
    assert np.all(np.isfinite(log_probs))
    assert np.all(np.isfinite(probs))
    assert np.isfinite(loss)

    return loss, probs

def backward(probs, y, cache, W2, omit_batch_mean=False):
    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]

    N = X.shape[0]

    assert probs.shape == (N, 3)
    assert y.shape == (N,)
    assert W2.shape == (8, 3)

    # one-hot labels Y
    Y = np.zeros_like(probs)
    Y[np.arange(N), y] = 1.0

    assert Y.shape == probs.shape

    # output error signal:
    # correct: Delta^(2) = (P - Y) / N
    delta2 = probs - Y

    if not omit_batch_mean:
        delta2 = delta2 / N

    assert delta2.shape == probs.shape

    # gradients of the second affine layer
    dW2 = A1.T @ delta2
    db2 = delta2.sum(axis=0)

    assert dW2.shape == W2.shape
    assert db2.shape == (3,)

    # propagate the signal through the second affine layer
    dA1 = delta2 @ W2.T
    assert dA1.shape == A1.shape

    # propagate through ReLU
    delta1 = dA1 * (Z1 > 0)
    assert delta1.shape == Z1.shape

    # gradients of the first affine layer
    dW1 = X.T @ delta1
    db1 = delta1.sum(axis=0)

    assert dW1.shape == (4, 8)
    assert db1.shape == (8,)

    return dW1, db1, dW2, db2