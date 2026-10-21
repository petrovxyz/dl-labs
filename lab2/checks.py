"""ЛР2. Незалежна перевірка пошуку порога на трьох малих наборах.

    uv run python checks.py --out outputs

Перевірочна реалізація (повільний перебір, дозволений лише тут) написана на
звичайному Python і не використовує ні швидкий алгоритм, ні його результати:
власний список кандидатів, пряме застосування правила s_i >= t до кожного
приклада, власний вибір порога. Потім обидві реалізації порівнюються точно.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from thresholds import select_threshold, threshold_table

# Малі набори для перевірки: оцінки s і мітки y.
SMALL_SETS: dict[str, tuple[list[float], list[int]]] = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    # Одинадцять оцінок 0,5 підряд: перша з міткою 1, решта десять - з міткою 0.
    "C": ([0.8] + [0.5] * 11 + [0.2], [1] + [1] + [0] * 10 + [0]),
}

# Що має перевіряти кожен набір (для звіту).
PURPOSE = {
    "A": "рівні оцінки: дві оцінки 0,7 з мітками 1 і 0 отримують однаковий прогноз",
    "B": "відсутність позитивних міток: P = 0, мінімум C на крайньому кандидаті +∞",
    "C": "два кандидати з однаковою вартістю: за правилом розв'язання нічиєї - найбільший поріг",
}


def brute_force_table(s: list[float], y: list[int]) -> list[dict[str, float | int]]:
    """Повільний перебір: для кожного кандидата пряме правило s_i >= t."""
    candidates = [math.inf] + sorted(set(s), reverse=True)
    rows = []
    for t in candidates:
        tp = fp = fn = tn = 0
        for s_i, y_i in zip(s, y, strict=True):
            if s_i >= t:
                if y_i == 1:
                    tp += 1
                else:
                    fp += 1
            elif y_i == 1:
                fn += 1
            else:
                tn += 1
        rows.append({"threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": 10 * fn + fp})
    return rows


def brute_force_choice(rows: list[dict[str, float | int]]) -> float:
    """Мінімум C; серед кількох мінімумів - найбільший поріг."""
    c_min = min(r["C"] for r in rows)
    return max(r["threshold"] for r in rows if r["C"] == c_min)


def compare(name: str) -> tuple[bool, list[dict], float, float]:
    s, y = SMALL_SETS[name]
    slow = brute_force_table(s, y)
    slow_t = brute_force_choice(slow)

    fast_table = threshold_table(np.array(s), np.array(y))
    fast_t = float(select_threshold(fast_table)["threshold"])
    fast = fast_table.to_dict("records")

    rows = []
    ok = len(slow) == len(fast)
    for r_slow, r_fast in zip(slow, fast):
        same = all(r_slow[k] == r_fast[k] for k in ("threshold", "TP", "FP", "FN", "TN", "C"))
        ok &= same
        rows.append({**r_slow, "C_fast": int(r_fast["C"]), "match": same})
    ok &= slow_t == fast_t
    return ok, rows, slow_t, fast_t


def fmt_t(t: float) -> str:
    return "+∞" if math.isinf(t) else format(t, "g").replace(".", ",")


def fmt_ratio(num: int, den: int) -> str:
    return "не визначено (0/0)" if den == 0 else format(num / den, ".4f").replace(".", ",")


def report(results: dict[str, tuple[bool, list[dict], float, float]]) -> str:
    lines = ["## Перевірка пошуку порога на малих наборах", ""]
    for name, (ok, rows, slow_t, fast_t) in results.items():
        s, y = SMALL_SETS[name]
        lines += [
            f"### Набір {name}: {PURPOSE[name]}",
            "",
            f"s = [{'; '.join(fmt_t(v) for v in s)}], y = [{'; '.join(map(str, y))}]",
            "",
            "Лічильники - повільний перебір (пряме правило s_i >= t); C порівнюється з реалізацією "
            "через кумулятивні суми; «Збіг» - точна рівність порога, TP, FP, FN, TN і C в обох реалізаціях.",
            "",
            "| Поріг t | TP | FP | FN | TN | C (перебір) | C (кумулятивні суми) | Збіг |",
            "| :-- | --: | --: | --: | --: | --: | --: | :-: |",
        ]
        for r in rows:
            lines.append(
                f"| {fmt_t(r['threshold'])} | {r['TP']} | {r['FP']} | {r['FN']} | {r['TN']} "
                f"| {r['C']} | {r['C_fast']} | {'так' if r['match'] else 'НІ'} |"
            )
        best = next(r for r in rows if r["threshold"] == slow_t)
        lines += [
            "",
            f"Обраний поріг (мінімум C, за нічиєї - найбільший): перебір t* = {fmt_t(slow_t)}, "
            f"кумулятивні суми t* = {fmt_t(fast_t)}; C(t*) = {best['C']}; "
            f"precision = {fmt_ratio(best['TP'], best['TP'] + best['FP'])}, "
            f"recall = {fmt_ratio(best['TP'], best['TP'] + best['FN'])}.",
            f"Перевірку пройдено: {'так' if ok else 'НІ'}.",
            "",
        ]
    return "\n".join(lines)


def run_checks(out: Path | None = None) -> bool:
    results = {name: compare(name) for name in SMALL_SETS}
    text = report(results)
    print(text)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "checks.md").write_text(text, encoding="utf-8")
    return all(ok for ok, *_ in results.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Тека для checks.md")
    args = parser.parse_args()
    if not run_checks(args.out):
        raise SystemExit("Перевірку пошуку порога не пройдено.")
    print("Усі три перевірки пройдено.")


if __name__ == "__main__":
    main()
