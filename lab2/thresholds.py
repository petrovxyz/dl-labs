"""ЛР2. Підрахунок помилок за порогом і пошук порога за мінімумом вартості.

Правило рішення: прогнозуємо шахрайство (клас 1), якщо s_i >= t.
Вартість помилок: C(t) = 10 * FN(t) + FP(t); Amount у формулу не входить.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

C_FN = 10  # ціна пропущеного шахрайства
C_FP = 1  # ціна хибної тривоги
COLUMNS = ["threshold", "TP", "FP", "FN", "TN", "C"]


def check_inputs(s: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Одновимірні масиви однакової довжини, скінченні оцінки, мітки 0/1."""
    s = np.asarray(s)
    y = np.asarray(y)
    if s.ndim != 1 or y.ndim != 1 or s.shape != y.shape:
        raise ValueError(f"Очікуються 1-D масиви однакової форми, отримано {s.shape} і {y.shape}.")
    if len(s) == 0:
        raise ValueError("Порожня вибірка.")
    if not np.isfinite(s).all():
        raise ValueError("Оцінки мають бути скінченними.")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Мітки мають бути 0 або 1.")
    return s, y.astype(np.int64)


def cost(fn: int, fp: int) -> int:
    return C_FN * fn + C_FP * fp


def confusion_at(s: np.ndarray, y: np.ndarray, t: float) -> dict[str, float | int]:
    """TP, FP, FN, TN і C для одного порога t прямим застосуванням правила s >= t."""
    s, y = check_inputs(s, y)
    pred = s >= t
    tp = int(np.count_nonzero(pred & (y == 1)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero(~pred & (y == 1)))
    tn = int(np.count_nonzero(~pred & (y == 0)))
    return {"threshold": float(t), "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fn, fp)}


def threshold_table(s: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Повна таблиця кандидатів за O(n log n): сортування й кумулятивні суми.

    Кандидати - +inf («усі негативні») та всі унікальні оцінки за спаданням;
    найменша оцінка дає «усі позитивні». Для порога u позитивними є всі приклади
    з s >= u, тобто відсортований префікс до кінця групи рівних оцінок u
    включно. Тому рівні оцінки завжди отримують однаковий прогноз.
    """
    s, y = check_inputs(s, y)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    order = np.argsort(s, kind="stable")[::-1]  # за спаданням оцінки
    s_sorted = s[order]
    y_sorted = y[order]
    cum_tp = np.cumsum(y_sorted)  # позитиви серед перших k+1 прикладів
    cum_fp = np.cumsum(1 - y_sorted)  # негативи серед перших k+1 прикладів

    # Останній індекс кожної групи рівних оцінок у відсортованому масиві.
    group_end = np.flatnonzero(np.append(s_sorted[1:] != s_sorted[:-1], True))

    thresholds = np.concatenate(([np.inf], s_sorted[group_end].astype(np.float64)))
    tp = np.concatenate(([0], cum_tp[group_end]))
    fp = np.concatenate(([0], cum_fp[group_end]))
    fn = n_pos - tp
    tn = n_neg - fp
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "C": C_FN * fn + C_FP * fp,
        },
        columns=COLUMNS,
    )


def select_threshold(table: pd.DataFrame) -> pd.Series:
    """Рядок із мінімальною C; за кількох мінімумів - найбільший поріг.

    Таблиця впорядкована за спаданням порога (+inf перший), тож np.argmin, що
    повертає перше входження мінімуму, дає саме найбільший поріг.
    """
    thresholds = table["threshold"].to_numpy()
    if not (np.diff(thresholds) < 0).all():
        raise ValueError("Пороги в таблиці мають строго спадати.")
    return table.iloc[int(np.argmin(table["C"].to_numpy()))]


def precision_recall(tp: int, fp: int, fn: int) -> tuple[float | None, float | None]:
    """precision = TP/(TP+FP), recall = TP/(TP+FN); None, якщо знаменник нульовий."""
    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = tp / (tp + fn) if tp + fn > 0 else None
    return precision, recall
