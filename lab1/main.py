"""ЛР1. Реалізація та перевірка зворотного поширення похибки.

uv run main.py            # правильна реалізація: звірка з PyTorch + чисельна перевірка
uv run main.py --buggy    # дослід із пропущеним діленням на N у градієнті за логітами
"""
import argparse
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel

from checks import EPS, TOL_NUM, run_numerical_gradient_checks
from data import load_and_prepare_iris
from model_numpy import backward, cross_entropy_loss, forward, initialize_parameters
from model_torch import torch_reference
from report import Section, save_markdown, show, verdict

TOL_TORCH = 1e-12
HERE = Path(__file__).parent

GRAD_NAMES = ("W1", "b1", "W2", "b2")

PREDICTION = (
    "1. Втрата. L залежить лише від прямого проходу й крос-ентропії, а прибирання ділення "
    "стосується тільки backward(). На тих самих параметрах L не зміниться: "
    "|L_NumPy − L_PyTorch| лишиться ≤ 1e-12.\n"
    "2. Ненульові градієнти. Δ⁽²⁾ = P − Y замість (P − Y)/N, а backprop лінійний за Δ⁽²⁾, "
    "тому всі чотири градієнти зростуть рівно в N = 105 разів (нулі лишаться нулями, "
    "напрямок градієнта збережеться): g_bug = N · g.\n"
    "3. Які перевірки виявлять помилку. Звірка з PyTorch: градієнти W1, b1, W2, b2 не пройдуть "
    "(різниці ≫ 1e-12), а втрата пройде. Чисельна перевірка: для всіх чотирьох параметрів "
    "g_manual ≈ N · g_num, |g_num − g_manual| ≈ (N − 1)|g_num| ≫ 1e-7. "
    "Перевірка втрати помилку не виявить."
)


def fmt(x: float) -> str:
    return f"{x:.3e}"


def shapes_section(X, y, params, cache, logits, probs, grads) -> Section:
    W1, b1, W2, b2 = params
    dW1, db1, dW2, db2 = grads
    rows = [
        ["X", "A⁽⁰⁾", str(X.shape), "N × 4"],
        ["y", "мітки класів c_i", str(y.shape), "N"],
        ["W1 / b1", "параметри шару 1", f"{W1.shape} / {b1.shape}", "4 × 8 / 8"],
        ["Z1", "Z⁽¹⁾ = XW1 + b1", str(cache["Z1"].shape), "N × 8"],
        ["A1", "A⁽¹⁾ = ReLU(Z⁽¹⁾)", str(cache["A1"].shape), "N × 8"],
        ["W2 / b2", "параметри шару 2", f"{W2.shape} / {b2.shape}", "8 × 3 / 3"],
        ["Z2", "Z⁽²⁾ = A⁽¹⁾W2 + b2 (логіти)", str(logits.shape), "N × 3"],
        ["P", "softmax(Z⁽²⁾)", str(probs.shape), "N × 3"],
        ["∇W1", "Xᵀ Δ⁽¹⁾", str(dW1.shape), "4 × 8"],
        ["∇b1", "Σ рядків Δ⁽¹⁾", str(db1.shape), "8"],
        ["∇W2", "(A⁽¹⁾)ᵀ Δ⁽²⁾", str(dW2.shape), "8 × 3"],
        ["∇b2", "Σ рядків Δ⁽²⁾", str(db2.shape), "3"],
    ]
    return Section(
        "Розмірності масивів (N = 105)",
        ["Масив", "Зміст", "Фактична форма", "Очікувана"],
        rows,
        left_cols={0, 1},
    )


def loss_values_section(loss, torch_loss) -> Section:
    return Section(
        "Значення втрати",
        ["Реалізація", "Втрата L"],
        [["NumPy", f"{float(loss)!r}"], ["PyTorch", f"{float(torch_loss)!r}"]],
    )


def torch_section(loss, grads, torch_loss, torch_grads, title) -> Section:
    values = [(loss, torch_loss)] + list(zip(grads, torch_grads))
    for pair in values:
        assert np.all(np.isfinite(pair[0])) and np.all(np.isfinite(pair[1]))

    diffs = [float(np.max(np.abs(a - b))) for a, b in values]
    labels = ["Втрата"] + [f"Градієнт {name}" for name in GRAD_NAMES]
    rows = [[label, fmt(d), verdict(d <= TOL_TORCH)] for label, d in zip(labels, diffs)]
    return Section(
        title,
        ["Величина", "Максимальна абсолютна різниця NumPy / PyTorch", "Перевірку пройдено"],
        rows,
        note=f"критерій: |a − b| ≤ {TOL_TORCH:.0e}",
    ), diffs


def numerical_section(results, title) -> Section:
    rows = [
        [
            r["parameter"],
            f"{r['manual']:.9e}",
            f"{r['numerical']:.9e}",
            fmt(r["abs_diff"]),
            verdict(r["passed"]),
        ]
        for r in results
    ]
    return Section(
        title,
        ["Параметр", "Градієнт backward()", "Чисельна похідна", "Абсолютна різниця", "Перевірку пройдено"],
        rows,
        note=f"ε = {EPS:.0e}, критерій: |g_num − g_manual| ≤ {TOL_NUM:.0e}",
    )


def prepare():
    X_train, y_train, X_test, y_test = load_and_prepare_iris()
    assert X_train.shape == (105, 4) and X_test.shape == (45, 4)
    assert np.array_equal(np.bincount(y_train), [35, 35, 35])
    assert np.array_equal(np.bincount(y_test), [15, 15, 15])

    params = initialize_parameters()
    W1, b1, W2, b2 = params

    logits, cache = forward(X_train, W1, b1, W2, b2)
    loss, probs = cross_entropy_loss(logits, y_train)
    assert np.all(np.isfinite(logits)) and np.all(np.isfinite(probs)) and np.isfinite(loss)

    reference = torch_reference(X_train, y_train, W1, b1, W2, b2)
    return X_train, y_train, params, logits, cache, loss, probs, reference


def run_correct(console: Console) -> None:
    X, y, params, logits, cache, loss, probs, ref = prepare()
    W1, b1, W2, b2 = params
    torch_loss, *torch_grads = ref

    grads = backward(probs, y, cache, W2)

    console.rule("[bold]ЛР1 · правильна реалізація[/]")
    sections = [
        shapes_section(X, y, params, cache, logits, probs, grads),
        loss_values_section(loss, torch_loss),
    ]
    cmp_section, diffs = torch_section(
        loss, grads, torch_loss, torch_grads, "Звірка NumPy / PyTorch"
    )
    sections.append(cmp_section)
    results = run_numerical_gradient_checks(X, y, W1, b1, W2, b2, *grads)
    sections.append(numerical_section(results, "Чисельна перевірка градієнтів"))

    for section in sections:
        show(console, section)

    save_markdown(HERE / "results.md", "Результати: правильна реалізація", sections)
    console.print(f"Таблиці збережено у [bold]{HERE / 'results.md'}[/]")

    assert all(d <= TOL_TORCH for d in diffs), "звірка з PyTorch не пройшла"
    assert all(r["passed"] for r in results), "чисельна перевірка не пройшла"


def run_buggy(console: Console) -> None:
    X, y, params, logits, cache, loss, probs, ref = prepare()
    W1, b1, W2, b2 = params
    torch_loss, *torch_grads = ref
    N = X.shape[0]

    good = backward(probs, y, cache, W2)
    bug = backward(probs, y, cache, W2, omit_batch_mean=True)

    console.rule("[bold]ЛР1 · дослід із навмисною помилкою (без ділення на N)[/]")
    console.print(Panel(PREDICTION, title="Прогноз (записано до запуску)", border_style="yellow"))
    console.print()

    sections = []

    sections.append(
        Section(
            "Втрата не залежить від помилки в backward()",
            ["Реалізація", "Втрата L"],
            [["NumPy (з помилкою в backward)", f"{float(loss)!r}"], ["PyTorch", f"{float(torch_loss)!r}"]],
        )
    )

    scale_rows = []
    for name, g_bug, g_ok in zip(GRAD_NAMES, bug, good):
        nonzero = g_ok != 0
        ratio = g_bug[nonzero] / g_ok[nonzero]
        scale_rows.append(
            [
                f"Градієнт {name}",
                f"{ratio.min():.10f}",
                f"{ratio.max():.10f}",
                fmt(float(np.max(np.abs(g_bug - N * g_ok)))),
            ]
        )
    sections.append(
        Section(
            f"Масштаб ненульових градієнтів (N = {N})",
            ["Величина", "min g_bug / g", "max g_bug / g", "max abs(g_bug − N·g)"],
            scale_rows,
        )
    )

    cmp_section, diffs = torch_section(
        loss, bug, torch_loss, torch_grads, "Звірка NumPy (з помилкою) / PyTorch"
    )
    sections.append(cmp_section)

    results = run_numerical_gradient_checks(X, y, W1, b1, W2, b2, *bug)
    sections.append(numerical_section(results, "Чисельна перевірка (з помилкою)"))

    for section in sections:
        show(console, section)

    detected_torch = all(d > TOL_TORCH for d in diffs[1:])
    detected_num = all(not r["passed"] for r in results)
    loss_ok = diffs[0] <= TOL_TORCH

    verdict_rows = [
        ["Втрата не змінилась", verdict(loss_ok)],
        ["Усі ненульові градієнти зросли в N разів", verdict(all(float(r[3]) < 1e-10 for r in scale_rows))],
        ["Помилку виявила звірка з PyTorch", verdict(detected_torch)],
        ["Помилку виявила чисельна перевірка", verdict(detected_num)],
    ]
    sections.append(
        Section("Зіставлення з прогнозом", ["Твердження прогнозу", "Підтверджено"], verdict_rows)
    )
    show(console, sections[-1])

    save_markdown(
        HERE / "results_buggy.md",
        "Результати: дослід із навмисною помилкою",
        ["**Прогноз (записано до запуску)**\n\n" + PREDICTION.replace("\n", "\n\n"), *sections],
    )
    console.print(f"Таблиці збережено у [bold]{HERE / 'results_buggy.md'}[/]")

    assert loss_ok and detected_torch and detected_num, "помилку не виявлено"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--buggy",
        action="store_true",
        help="дослід: прибрати ділення на N у градієнті за логітами",
    )
    args = parser.parse_args()

    console = Console()
    (run_buggy if args.buggy else run_correct)(console)


if __name__ == "__main__":
    main()
