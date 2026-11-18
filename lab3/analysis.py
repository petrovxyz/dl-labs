"""Команда аналізу ЛР3.

1. Перевірка: SHA-256 split.csv однаковий у всіх семи журналах і збігається з поточним файлом.
2. Зі збережених журналів: таблиця семи запусків, криві, розклад ефектів, перевірка вибору.
3. Лише потім, один раз і на CPU: test для lab3_model.pt, матриця плутанини, п'ять помилок.

    uv run python analysis.py --data /шлях/до/intel-image-classification
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from torch import nn  # noqa: E402
from torchvision.models import resnet18  # noqa: E402

from common import (  # noqa: E402
    CLASSES, FIG_DIR, IMG_SIZE, MODEL_PATH, NUM_CLASSES, OUT_DIR, RUNS, SPLIT_CSV, read_json, run_dir,
    select_run, sha256_file, write_json,
)
from data import eval_loader, load_part, load_test_set, part_rows, read_split, test_dir  # noqa: E402
from model import macro_f1, predict_logits  # noqa: E402

CPU = torch.device("cpu")
COLORS = {"scratch": "tab:red", "frozen": "tab:blue", "unfreeze": "tab:green"}
STYLES = {1: "-", 2: "--", 3: ":"}


def fail(msg: str) -> None:
    sys.exit(f"ПОМИЛКА: {msg}")


def f4(x: float) -> str:
    return f"{x:.4f}"


def pp(x: float) -> str:
    """Різниця в пунктах macro-F1 (×100)."""
    return f"{100 * x:+.2f} п."


# ------------------------------------------------------------------ журнали

def load_runs() -> list[dict]:
    current = sha256_file(SPLIT_CSV)
    runs = []
    for num, regime, seed in RUNS:
        path = run_dir(regime, seed) / "run.json"
        if not path.exists():
            fail(f"немає журналу {path}")
        run = read_json(path)
        if run["split_csv_sha256"] != current:
            fail(f"хеш split.csv у {path} ({run['split_csv_sha256'][:12]}…) не збігається з поточним "
                 f"({current[:12]}…): запуски на різних поділах порівнювати не можна, перенавчіть усі сім")
        if (run["run"], run["regime"], run["seed"]) != (num, regime, seed):
            fail(f"журнал {path} не відповідає запуску {num} ({regime}, зерно {seed})")
        runs.append(run)
    hashes = {r["split_csv_sha256"] for r in runs}
    if len(hashes) != 1:
        fail("хеші split.csv у журналах різні")
    print(f"Хеш split.csv однаковий у всіх 7 журналах і збігається з поточним: {current}")
    return runs


def check_val_logits(runs: list[dict], val_labels: np.ndarray) -> None:
    """macro-F1, перерахований зі збережених логітів validation, збігається з журналом."""
    for run in runs:
        logits = np.load(run_dir(run["regime"], run["seed"]) / "val_logits.npy")
        if logits.shape != (len(val_labels), NUM_CLASSES):
            fail(f"val_logits.npy {run['regime']}_seed{run['seed']}: форма {logits.shape}")
        f = macro_f1(val_labels, logits)
        if abs(f - run["final"]["val_macro_f1"]) > 1e-9:
            fail(f"macro-F1 з val_logits ({f}) ≠ журнал ({run['final']['val_macro_f1']})")
    print("macro-F1, перераховані зі збережених val_logits.npy, збігаються з журналами.")


def runs_table(runs: list[dict]) -> str:
    lines = [
        "| # | Режим | Зерно | Втрата train, епоха 8 | macro-F1 validation, епоха 8 | Час навчання, с | "
        "Час разом з validation, с |",
        "| --: | :-- | --: | --: | --: | --: | --: |",
    ]
    for r in runs:
        lines.append(
            f"| {r['run']} | {r['regime']} | {r['seed']} | {f4(r['final']['train_loss'])} | "
            f"{f4(r['final']['val_macro_f1'])} | {r['time_seconds']['train_steps']:.1f} | "
            f"{r['time_seconds']['total_with_validation']:.1f} |"
        )
    devices = sorted({r["env"]["device"] for r in runs})
    commits = sorted({(r["git"]["commit"] or "?")[:10] + (" (dirty)" if r["git"].get("dirty") else "") for r in runs})
    lines += ["", f"Пристрій: {', '.join(devices)}. Коміти коду: {', '.join(commits)}."]
    return "\n".join(lines)


def curves_figure(runs: list[dict], path: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    epochs = np.arange(1, len(runs[0]["history"]["train_loss"]) + 1)
    for r in runs:
        label = f"#{r['run']} {r['regime']}, зерно {r['seed']}"
        kw = dict(color=COLORS[r["regime"]], ls=STYLES[r["seed"]], marker="o", ms=3, label=label)
        a1.plot(epochs, r["history"]["train_loss"], **kw)
        a2.plot(epochs, r["history"]["val_macro_f1"], **kw)
    a1.set(title="Втрата на train", xlabel="епоха", ylabel="крос-ентропія", xticks=epochs)
    a2.set(title="macro-F1 на validation", xlabel="епоха", ylabel="macro-F1", xticks=epochs, ylim=(0, 1))
    a2.axhline(1 / 6, color="gray", lw=0.8, ls="--")
    a2.text(1.05, 1 / 6 + 0.01, "1/6", color="gray", fontsize=8)
    for a in (a1, a2):
        a.grid(alpha=0.3)
    a2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def effects(runs: list[dict]) -> tuple[str, dict]:
    F = {(r["regime"], r["seed"]): r["final"]["val_macro_f1"] for r in runs}
    fz = [F[("frozen", s)] for s in (1, 2, 3)]
    uf = [F[("unfreeze", s)] for s in (1, 2, 3)]
    e = {
        "F_scratch_1": F[("scratch", 1)],
        "F_frozen_mean": float(np.mean(fz)), "R_frozen": max(fz) - min(fz),
        "F_unfreeze_mean": float(np.mean(uf)), "R_unfreeze": max(uf) - min(uf),
    }
    e["R"] = max(e["R_frozen"], e["R_unfreeze"])
    e["delta_pre"] = e["F_frozen_mean"] - e["F_scratch_1"]
    e["delta_unf"] = e["F_unfreeze_mean"] - e["F_frozen_mean"]
    e["delta_total"] = e["F_unfreeze_mean"] - e["F_scratch_1"]

    def verdict(d: float) -> str:
        if abs(d) <= e["R"]:
            return "|Δ| ≤ R: різниця не відрізняється від розкиду, результату немає"
        return "|Δ| > R: ефект вартий перевірки, але не доведений"

    e["verdict_pre"] = verdict(e["delta_pre"])
    e["verdict_unf"] = verdict(e["delta_unf"])
    md = "\n".join([
        "| Величина | Значення |",
        "| :-- | :-- |",
        f"| $F_{{\\text{{scratch}},1}}$ | {f4(e['F_scratch_1'])} |",
        f"| $\\bar F_\\text{{frozen}}$, $R_\\text{{frozen}}$ | {f4(e['F_frozen_mean'])}, {f4(e['R_frozen'])} |",
        f"| $\\bar F_\\text{{unfreeze}}$, $R_\\text{{unfreeze}}$ | {f4(e['F_unfreeze_mean'])}, {f4(e['R_unfreeze'])} |",
        f"| $R = \\max(R_\\text{{frozen}}, R_\\text{{unfreeze}})$ | {f4(e['R'])} |",
        f"| $\\Delta_\\text{{pre}} = \\bar F_\\text{{frozen}} - F_{{\\text{{scratch}},1}}$ | "
        f"{e['delta_pre']:+.4f} ({pp(e['delta_pre'])}); {e['verdict_pre']} |",
        f"| $\\Delta_\\text{{unf}} = \\bar F_\\text{{unfreeze}} - \\bar F_\\text{{frozen}}$ | "
        f"{e['delta_unf']:+.4f} ({pp(e['delta_unf'])}); {e['verdict_unf']} |",
        f"| $\\Delta_\\text{{pre}} + \\Delta_\\text{{unf}} = \\bar F_\\text{{unfreeze}} - F_{{\\text{{scratch}},1}}$ | "
        f"{e['delta_total']:+.4f} ({pp(e['delta_total'])}) |",
    ])
    return md, e


# ------------------------------------------------------------------ вибір

def load_selected_model() -> tuple[nn.Module, dict]:
    if not MODEL_PATH.exists():
        fail(f"немає {MODEL_PATH.name}")
    ckpt = torch.load(MODEL_PATH, map_location=CPU, weights_only=True)
    expected_keys = {"state_dict", "regime", "seed", "classes", "img_size"}
    if set(ckpt) != expected_keys:
        fail(f"ключі {MODEL_PATH.name}: {sorted(ckpt)}; очікувано {sorted(expected_keys)}")
    if ckpt["classes"] != CLASSES or ckpt["img_size"] != IMG_SIZE:
        fail(f"classes/img_size у {MODEL_PATH.name} не відповідають умові")
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def check_selection(runs: list[dict], ckpt: dict, model: nn.Module, val_loader, val_labels) -> tuple[str, dict]:
    records = [{"regime": r["regime"], "seed": r["seed"], "F": r["final"]["val_macro_f1"]} for r in runs]
    choice = select_run(records)
    if (ckpt["regime"], ckpt["seed"]) != (choice["regime"], choice["seed"]):
        fail(f"{MODEL_PATH.name}: {ckpt['regime']}, зерно {ckpt['seed']}, а правило вибору за журналами дає "
             f"{choice['regime']}, зерно {choice['seed']}")
    # Ваги справді з цього запуску: прогнози validation на CPU vs збережені логіти запуску (MPS/GPU).
    cpu_logits, labels = predict_logits(model, val_loader, CPU)
    assert np.array_equal(labels, val_labels)
    saved = np.load(run_dir(choice["regime"], choice["seed"]) / "val_logits.npy")
    agree = float((cpu_logits.argmax(1) == saved.argmax(1)).mean())
    max_diff = float(np.abs(cpu_logits - saved).max())
    f_cpu = macro_f1(labels, cpu_logits)
    if agree < 0.99:
        fail(f"ваги {MODEL_PATH.name} не відтворюють логіти validation вибраного запуску (збіг {agree:.4f})")
    means = choice["regime_means"]
    lines = [
        "| Режим | $\\bar F_r$ (validation) |",
        "| :-- | --: |",
        *[f"| {reg} | {f4(m)} |" for reg, m in means.items()],
        "",
        f"Правило вибору за журналами: режим **{choice['regime']}**, у ньому запуск із найбільшим "
        f"$F_{{r,s}}$: зерно **{choice['seed']}**, $F = {f4(choice['F'])}$.",
        f"`{MODEL_PATH.name}`: regime = `{ckpt['regime']}`, seed = {ckpt['seed']}, "
        f"img_size = {ckpt['img_size']} - **відповідає правилу**.",
        f"Ваги `{MODEL_PATH.name}` на CPU відтворюють збережені логіти validation цього запуску: "
        f"збіг argmax {agree:.4f}, max |Δлогіт| = {max_diff:.2e}, macro-F1 на CPU {f4(f_cpu)}.",
    ]
    return "\n".join(lines), {**choice, "val_argmax_agreement_cpu": agree, "val_max_abs_logit_diff": max_diff,
                              "val_macro_f1_cpu": f_cpu}


# ------------------------------------------------------------------ test

def confusion_figure(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(NUM_CLASSES), CLASSES)
    ax.set_xlabel("Передбачений клас")
    ax.set_ylabel("Справжній клас")
    thr = cm.max() / 2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thr else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def confusion_table(cm: np.ndarray) -> str:
    lines = [
        "| справжній \\ передбачений | " + " | ".join(CLASSES) + " | Усього |",
        "| :-- | " + " | ".join("--:" for _ in CLASSES) + " | --: |",
    ]
    for i, c in enumerate(CLASSES):
        lines.append(f"| **{c}** | " + " | ".join(
            f"**{cm[i, j]}**" if i == j else str(cm[i, j]) for j in range(NUM_CLASSES)) + f" | {cm[i].sum()} |")
    return "\n".join(lines)


def five_errors(cm: np.ndarray, files: list[str], y: np.ndarray, pred: np.ndarray, probs: np.ndarray) -> list[dict]:
    """П'ять найчастіших позадіагональних пар (за рівних - за індексом справжнього, потім передбаченого);
    для кожної - перший за іменем (як рядок) файл; якщо пар < 5 - другий файл найчастішої пари і т. д."""
    pairs = sorted(
        ((int(cm[i, j]), i, j) for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if i != j and cm[i, j] > 0),
        key=lambda t: (-t[0], t[1], t[2]),
    )[:5]
    by_pair = {
        (i, j): sorted((files[k] for k in range(len(files)) if y[k] == i and pred[k] == j))
        for _, i, j in pairs
    }
    index = {f: k for k, f in enumerate(files)}
    chosen: list[dict] = []
    rank = 0
    while len(chosen) < 5 and any(rank < len(v) for v in by_pair.values()):
        for count, i, j in pairs:
            if len(chosen) == 5:
                break
            if rank < len(by_pair[(i, j)]):
                f = by_pair[(i, j)][rank]
                k = index[f]
                chosen.append({"file": f, "true": CLASSES[i], "pred": CLASSES[j],
                               "prob_pred": float(probs[k, j]), "pair_count": count})
        rank += 1
    return chosen


def errors_figure(errors: list[dict], data_root: Path, path: Path) -> None:
    fig, axes = plt.subplots(1, len(errors), figsize=(3.2 * len(errors), 3.9))
    for ax, e in zip(np.atleast_1d(axes), errors):
        with Image.open(test_dir(data_root) / e["true"] / e["file"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{e['file']}\nсправжній: {e['true']}\nпередбачено: {e['pred']} ({e['prob_pred']:.2f})",
                     fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path,
                    help="тека розпакованого архіву (містить seg_train/ і seg_test/)")
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    # 1. Версія даних.
    runs = load_runs()
    rows = read_split()
    val_labels = np.array([CLASSES.index(r["label"]) for r in part_rows(rows, "validation")])
    check_val_logits(runs, val_labels)

    # 2. Зі збережених журналів.
    table_md = runs_table(runs)
    curves_figure(runs, FIG_DIR / "curves.png")
    effects_md, eff = effects(runs)
    print("\n" + table_md + "\n\n" + effects_md)

    model, ckpt = load_selected_model()
    val_loader = eval_loader(load_part(args.data, "validation", rows))
    selection_md, sel = check_selection(runs, ckpt, model, val_loader, val_labels)
    print("\n" + selection_md)
    noise_note = (
        f"|Δ_unf| = {abs(eff['delta_unf']):.4f} ≤ R = {eff['R']:.4f}: вибір між frozen і unfreeze визначив шум."
        if abs(eff["delta_unf"]) <= eff["R"] else
        f"|Δ_unf| = {abs(eff['delta_unf']):.4f} > R = {eff['R']:.4f}."
    )

    # 3. Test - один раз, на CPU, лише для вибраної моделі.
    test_ds = load_test_set(args.data)
    test_logits, y = predict_logits(model, eval_loader(test_ds), CPU)
    files = [r["file"] for r in test_ds.rows]
    pred = test_logits.argmax(1)
    probs = torch.from_numpy(test_logits).softmax(1).numpy()
    f1_test = macro_f1(y, test_logits)
    acc = float((pred == y).mean())
    n = len(y)
    se = math.sqrt(acc * (1 - acc) / n)
    cm = confusion_matrix(y, pred, labels=list(range(NUM_CLASSES)))
    np.save(OUT_DIR / "test_logits.npy", test_logits.astype(np.float32))
    confusion_figure(cm, FIG_DIR / "confusion_matrix.png",
                     f"Test, {sel['regime']} зерно {sel['seed']}: macro-F1 = {f1_test:.4f}")
    errors = five_errors(cm, files, y, pred, probs)
    errors_figure(errors, args.data, FIG_DIR / "errors.png")
    with open(OUT_DIR / "errors_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(errors[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(errors)

    test_md = "\n".join([
        f"Модель: `{MODEL_PATH.name}` ({sel['regime']}, зерно {sel['seed']}), обчислення на CPU, n = {n}.",
        "",
        "| Величина | Значення |",
        "| :-- | --: |",
        f"| macro-F1 на test | {f4(f1_test)} |",
        f"| $F_{{r,s}}$ вибраного запуску (validation) | {f4(sel['F'])} |",
        f"| test − validation | {f1_test - sel['F']:+.4f} ({pp(f1_test - sel['F'])}) |",
        f"| accuracy $p$ на test | {f4(acc)} |",
        f"| $\\sqrt{{p(1-p)/n}}$, n = {n} | {f4(se)} ({100 * se:.2f} п.) |",
        "",
        confusion_table(cm),
    ])
    err_md = "\n".join([
        "| # | Файл | Справжній клас | Передбачений клас | Імовірність передбаченого класу | "
        "Кількість таких помилок у test |",
        "| --: | :-- | :-- | :-- | --: | --: |",
        *[f"| {k} | {e['file']} | {e['true']} | {e['pred']} | {e['prob_pred']:.4f} | {e['pair_count']} |"
          for k, e in enumerate(errors, 1)],
    ])
    print("\n" + test_md + "\n\n" + err_md)

    report = "\n\n".join([
        "# Результати команди аналізу\n\nЗгенеровано `analysis.py`; числа README взято звідси.",
        f"SHA-256 `split.csv`: `{sha256_file(SPLIT_CSV)}` - однаковий у всіх 7 журналах.",
        "## Сім запусків\n\n" + table_md,
        "## Розклад ефектів\n\n" + effects_md,
        "## Вибір конфігурації\n\n" + selection_md + "\n\n" + noise_note,
        "## Test\n\n" + test_md,
        "## П'ять характерних помилок\n\n" + err_md,
    ])
    (OUT_DIR / "analysis.md").write_text(report + "\n", encoding="utf-8")
    write_json(OUT_DIR / "analysis.json", {
        "effects": eff, "selection": sel, "noise_note": noise_note,
        "test": {"macro_f1": f1_test, "accuracy": acc, "n": n, "se_accuracy": se,
                 "confusion_matrix": cm.tolist(), "classes": CLASSES},
        "errors": errors,
    })
    print(f"\nЗбережено: {OUT_DIR / 'analysis.md'}, рисунки в {FIG_DIR}/")


if __name__ == "__main__":
    main()
