"""Команда навчання: поділ → перевірки → сім запусків → журнали → вибір за validation → lab3_model.pt.

Test не використовується.

    uv run python train.py --data /шлях/до/intel-image-classification
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import torch

from checks import batch_figure, bn_bug_experiment, change_report, initial_loss, param_report
from common import (
    CLASSES, P, EPOCHS, IMG_SIZE, RUNS, env_info, git_info, pick_device,
    read_json, run_dir, select_run, sha256_file, write_json,
)
from data import build_split, check_split, eval_loader, load_part, loader_settings, read_split, transform_settings, write_split
from model import REGIMES, model_settings, train_run


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path,
                    help="тека розпакованого архіву (містить seg_train/ і seg_test/)")
    ap.add_argument("--out", type=Path, default=None,
                    help="тека результатів (split.csv, runs/, lab3_model.pt, outputs/, figures/); за замовчуванням - тека коду")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    ap.add_argument("--checks-only", action="store_true", help="лише поділ і перевірки, без запусків")
    args = ap.parse_args()

    device = pick_device(args.device)
    log(f"Пристрій: {device}")
    if args.out is not None:
        P.set_root(args.out)
    P.out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Тека результатів: {P.root}")

    # ------------------------------------------------------------ поділ
    rows = build_split(args.data)
    split_ok, split_md = check_split(rows)
    write_split(rows)
    rows = read_split()  # далі дані беруться лише з split.csv
    split_sha = sha256_file(P.split_csv)
    log(split_md)
    log(f"SHA-256 split.csv: {split_sha}")
    if not split_ok:
        sys.exit("Поділ не збігається з контрольною таблицею - зупинка.")

    train_ds = load_part(args.data, "train", rows)
    val_ds = load_part(args.data, "validation", rows)
    val_loader = eval_loader(val_ds)
    log(f"train: {len(train_ds)}, validation: {len(val_ds)}")

    # ------------------------------------------------------------ перевірки
    log("\n== Перевірки до навчання ==")
    batch_md = batch_figure(train_ds)
    loss_md, loss_data = initial_loss(train_ds, device)
    log(loss_md)
    params_md, params_ok = param_report(device)
    log(params_md)
    change_md, change_ok = change_report(train_ds, device)
    log(change_md)
    bug_md, bug_data = bn_bug_experiment(train_ds, device)
    log(bug_md)

    checks_md = "\n\n".join([
        f"# Перевірки до навчання\n\nПристрій: {env_info(device)['device']}. "
        f"Згенеровано `train.py` {dt.datetime.now().isoformat(timespec='seconds')}.",
        "## Поділ\n\n" + split_md + f"\n\nSHA-256 `split.csv`: `{split_sha}`",
        "## Пакет після перетворень\n\n" + batch_md,
        "## Початкова втрата (зерно 1, перший пакет, до кроків оптимізатора)\n\n" + loss_md,
        "## Параметри режимів\n\n" + params_md,
        "## Що змінюється за 5 кроків (set_train_mode, зерно 1)\n\n" + change_md,
        "## Дослід з навмисною помилкою: frozen з model.train()\n\n" + bug_md,
    ])
    (P.out_dir / "checks.md").write_text(checks_md + "\n", encoding="utf-8")
    write_json(P.out_dir / "checks.json", {
        "split_ok": split_ok, "params_ok": params_ok, "changes_ok": change_ok,
        "initial_loss": loss_data,
        "bn_bug": {k: {kk: vv for kk, vv in v.items() if kk != "changes"} for k, v in bug_data.items()},
    })
    if not (params_ok and change_ok):
        sys.exit("Перевірки режимів не збіглися з очікуваними - зупинка.")
    log("Усі перевірки збіглися з очікуваними.")
    if args.checks_only:
        return

    # ------------------------------------------------------------ сім запусків
    records = []
    for num, regime, seed in RUNS:
        log(f"\n== Запуск {num}: {regime}, зерно {seed} ==")
        model, optimizer, history, val_logits, seconds = train_run(
            regime, seed, train_ds, val_loader, device, log=log
        )
        rd = run_dir(regime, seed)
        rd.mkdir(parents=True, exist_ok=True)
        np.save(rd / "val_logits.npy", val_logits.astype(np.float32))
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, rd / "model.pt")
        run = {
            "run": num,
            "regime": regime,
            "seed": seed,
            "settings": {
                "classes": CLASSES,
                "img_size": IMG_SIZE,
                "transforms": transform_settings(),
                "model": model_settings(regime, model, optimizer),
                "loaders": loader_settings(),
                "epochs": EPOCHS,
                "steps_per_epoch": history["steps"] // EPOCHS,
                "optimizer_steps": history["steps"],
                "seeding": "torch.manual_seed(seed); np.random.seed(seed) перед побудовою моделі",
                "train_loss_definition": "sum_b L_b*|b| / 1200, L_b - середня CE пакета до кроку",
                "validation_metric": "sklearn f1_score(average='macro') після кожної епохи, model.eval()",
                "weights_used": "після 8-ї епохи",
            },
            "git": git_info(),
            "split_csv_sha256": split_sha,
            "env": env_info(device),
            "history": {"train_loss": history["train_loss"], "val_macro_f1": history["val_macro_f1"]},
            "final": {"train_loss": history["train_loss"][-1], "val_macro_f1": history["val_macro_f1"][-1]},
            "time_seconds": seconds,
            "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        write_json(rd / "run.json", run)
        records.append({"regime": regime, "seed": seed, "F": history["val_macro_f1"][-1]})
        log(f"  час: {seconds['train_steps']:.1f} с (кроки), {seconds['total_with_validation']:.1f} с (з validation)")
        del model, optimizer

    # ------------------------------------------------------------ вибір за validation
    for num, regime, seed in RUNS:  # журнали на диску збігаються з отриманими числами
        assert read_json(run_dir(regime, seed) / "run.json")["final"]["val_macro_f1"] == next(
            r["F"] for r in records if r["regime"] == regime and r["seed"] == seed)
    choice = select_run(records)
    state_dict = torch.load(run_dir(choice["regime"], choice["seed"]) / "model.pt", weights_only=True)
    torch.save({
        "state_dict": state_dict,
        "regime": choice["regime"],
        "seed": choice["seed"],
        "classes": CLASSES,
        "img_size": IMG_SIZE,
    }, P.model_path)
    write_json(P.out_dir / "selection.json", choice)
    log(f"\nВибрано: {choice['regime']}, зерно {choice['seed']}, macro-F1 val {choice['F']:.4f}")
    log(f"Середні за режимами: {choice['regime_means']}")
    log(f"Збережено {P.model_path.name}")


if __name__ == "__main__":
    main()
