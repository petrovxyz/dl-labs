"""ЛР2. Аналіз порогів зі збережених прогнозів, без навчання і без вихідного CSV.

    uv run python analysis.py --saved results --out outputs

Порядок роботи:
1. Перевірка пошуку на наборах A, B, C (checks.py); якщо її не пройдено,
   поріг не фіксується і до тесту не переходимо.
2. Validation: PR- і ROC-криві, ROC-AUC, повна таблиця кандидатів,
   поріг t* за мінімумом C = 10 FN + FP і порівняння з 0,5 та +inf.
3. Test: зафіксований t* з threshold.json застосовується один раз;
   матриця плутанини, C, precision, recall і перші п'ять FP та FN за рядком CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from checks import run_checks
from lab2_starter import load_predictions
from thresholds import (
    C_FN,
    C_FP,
    confusion_at,
    precision_recall,
    select_threshold,
    threshold_table,
)

DEFAULT_THRESHOLD = 0.5
MARKERS = {"0,5": ("o", "tab:orange"), "t*": ("s", "tab:green")}


def threshold_to_json(t: float) -> float | str:
    """JSON без NaN/inf: +inf записуємо рядком, скінченний поріг - без округлення."""
    return "inf" if np.isinf(t) else float(t)


def threshold_from_json(value: float | str) -> float:
    return np.inf if value == "inf" else float(value)


def fmt_number(value: float, spec: str) -> str:
    """Округлення лише для відображення; десяткова кома."""
    return format(value, spec).replace(".", ",")


def fmt_threshold(t: float) -> str:
    return "+∞" if np.isinf(t) else fmt_number(t, ".6g")


def fmt_metric(value: float | None) -> str:
    return "не визначено" if value is None else fmt_number(value, ".4f")


def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(":--" if i == 0 else "--:" for i in range(len(df.columns))) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def plot_curves(y: np.ndarray, s: np.ndarray, points: dict[str, dict], out: Path) -> None:
    """PR- і ROC-криві на validation (функції scikit-learn) з робочими точками."""
    precision, recall, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    pi = y.mean()
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    markers = MARKERS

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(recall, precision, where="post", color="tab:blue", label="validation")
    ax.axhline(pi, color="grey", ls=":", label=f"базова лінія π = {pi:.4%}")
    for name, row in points.items():
        p, r = precision_recall(row["TP"], row["FP"], row["FN"])
        m, c = markers[name]
        ax.plot(r, p, m, color=c, ms=10, label=f"{name}: recall={r:.3f}, precision={p:.3f}")
    ax.set(xlabel="Повнота (recall)", ylabel="Точність (precision)", xlim=(0, 1.01), ylim=(0, 1.02))
    ax.set_title(f"Крива точність–повнота, validation (AP = {average_precision_score(y, s):.4f})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pr_curve_validation.png", dpi=150)
    plt.close(fig)

    auc = roc_auc_score(y, s)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, log in zip(axes, (False, True)):
        ax.plot(fpr, tpr, color="tab:blue", label=f"validation, ROC-AUC = {auc:.4f}")
        if not log:
            ax.plot([0, 1], [0, 1], color="grey", ls=":", label="випадкове ранжування")
        for name, row in points.items():
            m, c = markers[name]
            ax.plot(row["FP"] / n_neg, row["TP"] / n_pos, m, color=c, ms=10,
                    label=f"{name}: FPR={row['FP'] / n_neg:.5f}, TPR={row['TP'] / n_pos:.3f}")
        ax.set(xlabel="FPR = FP / (FP + TN)", ylabel="TPR = TP / (TP + FN)", ylim=(0, 1.02))
        if log:
            ax.set_xscale("log")
            ax.set_xlim(1 / n_neg / 2, 1)
            ax.set_title("ROC, логарифмічна вісь FPR")
        else:
            ax.set_xlim(0, 1)
            ax.set_title("ROC-крива, validation")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "roc_curve_validation.png", dpi=150)
    plt.close(fig)


def plot_cost(table: pd.DataFrame, points: dict[str, dict], n_pos: int, out: Path) -> None:
    """C(t) на validation; обидві осі логарифмічні.

    Для t з (u_{k+1}, u_k] прогноз той самий, що й для кандидата u_k, тому
    сходинка тягнеться від u_k вниз до наступного кандидата (where="post" за
    спадним порядком порогів). Показано лише t >= 1e-4, де видно мінімум.
    """
    shown = table[np.isfinite(table["threshold"]) & (table["threshold"] >= 1e-4)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(shown["threshold"], shown["C"], where="post", color="tab:blue", lw=1)
    ax.axhline(C_FN * n_pos, color="grey", ls=":", label=f"усі негативні (+∞): C = {C_FN * n_pos}")
    ax.axvline(C_FP / (C_FP + C_FN), color="tab:purple", ls="--",
               label="c_FP / (c_FP + c_FN) = 1/11 (теоретичний орієнтир)")
    for name, row in points.items():
        m, c = MARKERS[name]
        ax.plot(row["threshold"], row["C"], m, color=c, ms=10, zorder=3,
                label=f"{name} = {row['threshold']:.4g}: C = {int(row['C'])}")
    ax.set(xscale="log", yscale="log", xlabel="поріг t", ylabel="C = 10 FN + FP", xlim=(1e-4, 1.2))
    ax.set_title("Вартість помилок залежно від порога, validation")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "cost_validation.png", dpi=150)
    plt.close(fig)


def validation_analysis(data: dict[str, np.ndarray], out: Path) -> dict:
    y = data["validation_y"]
    s = data["validation_scores"]

    table = threshold_table(s, y)
    # Пороги записуються точно (repr float64); для точного зчитування pandas
    # потрібен pd.read_csv(..., float_precision="round_trip").
    table.to_csv(out / "candidates_validation.csv", index=False)
    best = select_threshold(table)
    t_star = float(best["threshold"])

    rows = {
        "Стандартний поріг": confusion_at(s, y, DEFAULT_THRESHOLD),
        "Мінімум вартості": best.to_dict(),
        "Усі негативні": table.iloc[0].to_dict(),
    }
    # Узгодженість: рядок t* у таблиці збігається з прямим підрахунком за правилом s >= t.
    direct = confusion_at(s, y, t_star)
    assert all(int(direct[k]) == int(best[k]) for k in ("TP", "FP", "FN", "TN", "C"))
    # Крайні кандидати: +inf - «усі негативні», найменша оцінка - «усі позитивні».
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    first, last = table.iloc[0], table.iloc[-1]
    assert np.isinf(first["threshold"]) and first["TP"] == 0 and first["FP"] == 0
    assert last["threshold"] == s.min() and last["TP"] == n_pos and last["FP"] == n_neg

    # Підсумкова таблиця: 0,5, t* і +inf.
    summary = pd.DataFrame(
        [
            {
                "Правило на validation": name,
                "Поріг": fmt_threshold(r["threshold"]),
                "TP": int(r["TP"]),
                "FP": int(r["FP"]),
                "FN": int(r["FN"]),
                "TN": int(r["TN"]),
                "C": int(r["C"]),
            }
            for name, r in rows.items()
        ]
    )
    # Ті самі робочі точки в частках: для пояснення PR/ROC.
    rates = pd.DataFrame(
        [
            {
                "Правило на validation": name,
                "precision = TP/(TP+FP)": fmt_metric(precision_recall(r["TP"], r["FP"], r["FN"])[0]),
                "recall = TPR = TP/(TP+FN)": fmt_metric(precision_recall(r["TP"], r["FP"], r["FN"])[1]),
                "FPR = FP/(FP+TN)": fmt_number(r["FP"] / n_neg, ".5f"),
            }
            for name, r in rows.items()
        ]
    )
    minima = table[table["C"] == best["C"]]["threshold"].to_numpy()
    auc = float(roc_auc_score(y, s))
    ap = float(average_precision_score(y, s))

    record = {
        "rule": "fraud if s >= t",
        "cost": "C = 10*FN + FP",
        "selected_on": "validation",
        "t_star": threshold_to_json(t_star),
        "t_star_float32_hex": float(np.float32(t_star)).hex() if np.isfinite(t_star) else "inf",
        "validation": {k: int(best[k]) for k in ("TP", "FP", "FN", "TN", "C")},
        "n_candidates": len(table),
        "n_thresholds_with_min_cost": int(len(minima)),
        "min_cost_threshold_range": [float(minima.min()), float(minima.max())],
        "roc_auc_validation": auc,
        "average_precision_validation": ap,
    }
    (out / "threshold.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    points = {"0,5": rows["Стандартний поріг"], "t*": rows["Мінімум вартості"]}
    plot_curves(y, s, points, out)
    plot_cost(table, points, int(y.sum()), out)

    lines = [
        "## Вибір порога на validation",
        "",
        f"n = {len(y)}, позитивних P = {n_pos}, негативних N = {n_neg}, "
        f"π = {fmt_number(n_pos / len(y), '.6f')}.",
        f"Кандидатів: {len(table)} (усі унікальні оцінки validation і +∞).",
        f"ROC-AUC = {fmt_number(auc, '.4f')}; середня точність AP = {fmt_number(ap, '.4f')}.",
        "",
        markdown_table(summary),
        "",
        markdown_table(rates),
        "",
        f"t* = {t_star!r} (float32: {record['t_star_float32_hex']}).",
        f"Порогів із мінімальною C = {int(best['C'])}: {len(minima)}; "
        f"вибрано найбільший із них.",
    ]
    report = "\n".join(lines) + "\n"
    (out / "validation.md").write_text(report, encoding="utf-8")
    print(report)
    return record


def undefined_reason(tp: int, fp: int, fn: int) -> list[str]:
    reasons = []
    if tp + fp == 0:
        reasons.append("precision не визначена: TP + FP = 0, модель не зробила жодного позитивного рішення.")
    if tp + fn == 0:
        reasons.append("recall не визначений: TP + FN = 0, у вибірці немає шахрайських транзакцій.")
    return reasons


def test_evaluation(data: dict[str, np.ndarray], out: Path) -> dict:
    """Тестова оцінка. Поріг береться лише із зафіксованого threshold.json і не змінюється."""
    record = json.loads((out / "threshold.json").read_text(encoding="utf-8"))
    assert record["selected_on"] == "validation"
    t_star = threshold_from_json(record["t_star"])

    y = data["test_y"]
    s = data["test_scores"]
    amount = data["test_amount"]
    row_ids = data["test_row_ids"]

    # Єдине застосування порога до test. Інших порогів на test не рахуємо: порівняння
    # порогів за тестом - та сама помилка, що й вибір гіперпараметрів за тестом.
    res = confusion_at(s, y, t_star)
    precision, recall = precision_recall(res["TP"], res["FP"], res["FN"])

    # Перші п'ять FP і перші п'ять FN за номером рядка CSV (від 0, без заголовка).
    pred = s >= t_star
    errors = []
    for kind, mask in (("FP", pred & (y == 0)), ("FN", ~pred & (y == 1))):
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(row_ids[idx], kind="stable")][:5]
        for i in idx:
            errors.append(
                {
                    "csv_row": int(row_ids[i]),
                    "error": kind,
                    "label": int(y[i]),
                    "score": float(s[i]),
                    "amount": float(amount[i]),
                    "score_minus_t_star": float(s[i]) - t_star,
                }
            )
    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(out / "errors_test.csv", index=False)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    result = {
        "t_star": record["t_star"],
        "test": {k: int(res[k]) for k in ("TP", "FP", "FN", "TN", "C")},
        "precision": precision,
        "recall": recall,
        "n_fp_total": int(res["FP"]),
        "n_fn_total": int(res["FN"]),
    }
    (out / "test.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    confusion = pd.DataFrame(
        [
            {"": "Справжній клас 1 (шахрайство)", "Прогноз 1": res["TP"], "Прогноз 0": res["FN"]},
            {"": "Справжній клас 0 (легітимна)", "Прогноз 1": res["FP"], "Прогноз 0": res["TN"]},
        ]
    )
    shown = pd.DataFrame(
        {
            "Рядок CSV (від 0, без заголовка)": errors_df["csv_row"],
            "Тип помилки": errors_df["error"],
            "Мітка": errors_df["label"],
            "Оцінка s": [fmt_number(v, ".6g") for v in errors_df["score"]],
            "Amount": [fmt_number(v, ".2f") for v in errors_df["amount"]],
            # Додатково до шаблону: відстань оцінки від порога для порівняння FP і FN.
            "s − t*": [fmt_number(v, "+.6g") for v in errors_df["score_minus_t_star"]],
        }
    )
    lines = [
        "## Тестова оцінка та аналіз помилок",
        "",
        f"Зафіксований на validation поріг t* = {fmt_threshold(t_star)} ({record['t_star']!r}); "
        f"застосовано до test один раз.",
        f"n = {len(y)}, позитивних P = {n_pos}, негативних N = {n_neg}.",
        "",
        markdown_table(confusion),
        "",
        f"TP = {res['TP']}, FP = {res['FP']}, FN = {res['FN']}, TN = {res['TN']}; "
        f"C = 10·FN + FP = 10·{res['FN']} + {res['FP']} = {res['C']}.",
        f"Точність (precision) = TP/(TP+FP) = {fmt_metric(precision)}; "
        f"повнота (recall) = TP/(TP+FN) = {fmt_metric(recall)}.",
        *undefined_reason(res["TP"], res["FP"], res["FN"]),
        "",
        f"Перші п'ять FP і перші п'ять FN за номером рядка CSV (усього FP = {res['FP']}, FN = {res['FN']}):",
        "",
        markdown_table(shown),
    ]
    report = "\n".join(lines) + "\n"
    (out / "test.md").write_text(report, encoding="utf-8")
    print(report)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saved", type=Path, default=Path("results"), help="Тека збережених прогнозів")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Тека для результатів аналізу")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Спершу незалежна перевірка пошуку, потім фіксуємо поріг і оцінюємо його на test.
    # Перевірки не залежать від моделі, тому виконуємо їх першими.
    if not run_checks(args.out):
        raise SystemExit("Перевірку пошуку порога на наборах A, B, C не пройдено; поріг не фіксуємо.")
    print("Усі три перевірки пройдено.\n")
    data = load_predictions(args.saved)
    validation_analysis(data, args.out)  # фіксує t* у threshold.json
    test_evaluation(data, args.out)  # лише читає зафіксований t*


if __name__ == "__main__":
    main()
