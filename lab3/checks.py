"""Перевірки пунктів 1–2: поділ, пакет після перетворень, початкова втрата, режими, дослід з BN."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from common import CLASSES, FIG_DIR, NUM_CLASSES  # noqa: E402
from data import denormalize, train_loader  # noqa: E402
from model import REGIMES, TOP_MODULES, TOTAL_PARAMS, build_model, count_params, set_train_mode  # noqa: E402

BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


# ---------------------------------------------------------------- пункт 1

def batch_figure(train_ds, path=FIG_DIR / "batch.png") -> str:
    """Перші 16 зображень першого навчального пакета так, як їх отримує модель."""
    torch.manual_seed(0)
    x, y = next(iter(train_loader(train_ds)))
    fig, axes = plt.subplots(4, 4, figsize=(9, 9.6))
    for ax, img, label in zip(axes.flat, x[:16], y[:16]):
        ax.imshow(denormalize(img).permute(1, 2, 0).numpy())
        ax.set_title(f"{int(label)}: {CLASSES[int(label)]}", fontsize=10)
        ax.axis("off")
    fig.suptitle(
        "Навчальний пакет після RandomResizedCrop(160) + HFlip, обернене нормування "
        f"(тензор {tuple(x.shape[1:])})",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return (
        f"Рисунок `{path.relative_to(path.parent.parent)}`: перші 16 зображень пакета, "
        f"форма пакета {tuple(x.shape)}, діапазон після нормування [{x.min():.2f}; {x.max():.2f}]."
    )


@torch.no_grad()
def initial_loss(train_ds, device) -> tuple[str, dict]:
    """L0 на першому пакеті до жодного кроку для scratch і frozen (зерно 1)."""
    criterion = nn.CrossEntropyLoss()
    ln6 = math.log(NUM_CLASSES)
    rows, data = [], {}
    for regime in ("scratch", "frozen"):
        model, _ = build_model(regime, 1, device)
        loader = train_loader(train_ds)
        set_train_mode(model, regime)
        body = nn.Sequential(*list(model.children())[:-1])
        all_losses, all_n, first = [], [], None
        for x, y in loader:  # 1-й пакет - це L0; решта лише для довідки (середнє за епохою без кроків)
            x, y = x.to(device), y.to(device)
            feats = body(x).flatten(1)
            logits = model.fc(feats)
            all_losses.append(criterion(logits, y).item())
            all_n.append(len(y))
            if first is None:
                first = (feats, logits)
        feats, logits = first
        loss = all_losses[0]
        sigma2 = logits.var(1).mean().item()
        stats = {
            "L0": loss,
            "ln6": ln6,
            "diff": loss - ln6,
            "logit_var_within_sample": sigma2,
            "approx_ln6_plus_half_var": ln6 + 0.5 * (1 - 1 / NUM_CLASSES) * sigma2,
            "feature_norm_mean": feats.norm(dim=1).mean().item(),
            "max_prob_mean": logits.softmax(1).max(1).values.mean().item(),
            "mean_over_all_batches_no_steps": sum(l * n for l, n in zip(all_losses, all_n)) / sum(all_n),
        }
        data[regime] = stats
        rows.append(
            f"| {regime} | {loss:.4f} | {loss - ln6:+.4f} | {stats['feature_norm_mean']:.2f} | "
            f"{sigma2:.3f} | {stats['approx_ln6_plus_half_var']:.4f} | "
            f"{stats['mean_over_all_batches_no_steps']:.4f} |"
        )
        del model
    md = "\n".join([
        f"$\\ln 6 = {ln6:.4f}$.",
        "",
        "| Режим | $L_0$ (1-й пакет) | $L_0 - \\ln 6$ | Норма ознак після GAP | "
        "Дисперсія логітів у прикладі $\\sigma^2$ | $\\ln 6 + \\tfrac12(1-\\tfrac16)\\sigma^2$ | "
        "Довідково: середнє за 38 пакетами без кроків |",
        "| :-- | --: | --: | --: | --: | --: | --: |",
        *rows,
    ])
    return md, data


# ---------------------------------------------------------------- пункт 2

def param_report(device) -> tuple[str, bool]:
    lines = [
        "| Режим | Усього параметрів | requires_grad=True | Очікувано | Групи AdamW (параметрів, lr) | "
        "Заморожених в оптимізаторі | Усі trainable в оптимізаторі | Збіг |",
        "| :-- | --: | --: | --: | :-- | --: | :-: | :-: |",
    ]
    all_ok = True
    for regime, cfg in REGIMES.items():
        model, opt = build_model(regime, 1, device)
        total = count_params(model.parameters())
        trainable = count_params(p for p in model.parameters() if p.requires_grad)
        in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
        frozen_in_opt = sum(1 for p in model.parameters() if id(p) in in_opt and not p.requires_grad)
        all_trainable_in_opt = all(id(p) in in_opt for p in model.parameters() if p.requires_grad)
        groups = "; ".join(
            f"{g['name']}: {count_params(g['params']):,}".replace(",", " ") + f" шт., lr={g['lr']:g}"
            for g in opt.param_groups
        )
        expected_groups = [(m, lr) for m, lr in cfg["groups"]]
        groups_ok = [("+".join(m), lr) for m, lr in expected_groups] == [
            (g["name"], g["lr"]) for g in opt.param_groups
        ] and all(g["weight_decay"] == 1e-4 for g in opt.param_groups)
        ok = (total == TOTAL_PARAMS and trainable == cfg["expected_trainable"] and frozen_in_opt == 0
              and all_trainable_in_opt and groups_ok)
        all_ok &= ok
        fmt = lambda v: f"{v:,}".replace(",", " ")  # noqa: E731
        lines.append(
            f"| {regime} | {fmt(total)} | {fmt(trainable)} | {fmt(cfg['expected_trainable'])} | {groups} | "
            f"{frozen_in_opt} | {'так' if all_trainable_in_opt else 'ні'} | {'так' if ok else 'НІ'} |"
        )
        del model, opt
    lines.append("")
    lines.append("Кількість параметрів - сума `numel()`; в усіх групах `weight_decay=1e-4`.")
    return "\n".join(lines), all_ok


def _snapshot(model: nn.Module) -> dict:
    return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}


def _five_steps(model, opt, regime, train_ds, device, use_set_train_mode=True) -> list[float]:
    if use_set_train_mode:
        set_train_mode(model, regime)
    else:
        model.train()  # навмисна помилка
    criterion = nn.CrossEntropyLoss()
    losses = []
    for step, (x, y) in enumerate(train_loader(train_ds)):
        if step == 5:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def _module_changes(before: dict, after: dict) -> dict:
    """Для кожного модуля верхнього рівня: чи змінився параметр, чи змінилась BN-статистика."""
    result = {}
    for mod in TOP_MODULES:
        keys = [k for k in before if k.split(".")[0] == mod]
        bn_keys = [k for k in keys if k.rsplit(".", 1)[-1] in BN_BUFFERS]
        param_keys = [k for k in keys if k not in bn_keys]
        params_changed = any(not torch.equal(before[k], after[k]) for k in param_keys)
        if bn_keys:
            bn_changed = any(not torch.equal(before[k], after[k]) for k in bn_keys)
        else:
            bn_changed = None
        result[mod] = (params_changed, bn_changed)
    return result


def _expected(regime: str, mod: str, has_bn: bool):
    frozen = mod in REGIMES[regime]["eval_modules"]
    return (not frozen, (not frozen) if has_bn else None)


def _yn(v) -> str:
    return "немає BN" if v is None else ("так" if v else "ні")


def change_report(train_ds, device) -> tuple[str, bool]:
    lines = [
        "| Режим | Модуль | Параметри змінилися | Ковзні статистики BN змінилися | Збігається з очікуваним |",
        "| :-- | :-- | :-: | :-: | :-: |",
    ]
    all_ok = True
    for regime in ("frozen", "unfreeze"):
        model, opt = build_model(regime, 1, device)
        before = _snapshot(model)
        _five_steps(model, opt, regime, train_ds, device)
        changes = _module_changes(before, _snapshot(model))
        for mod, (pc, bc) in changes.items():
            ok = (pc, bc) == _expected(regime, mod, bc is not None)
            all_ok &= ok
            lines.append(f"| {regime} | {mod} | {_yn(pc)} | {_yn(bc)} | {'так' if ok else 'НІ'} |")
        del model, opt
    return "\n".join(lines), all_ok


def bn_bug_experiment(train_ds, device) -> tuple[str, dict]:
    """frozen з model.train() замість set_train_mode; порівняння з правильним варіантом."""
    out = {}
    for variant, use_stm in (("set_train_mode", True), ("model.train()", False)):
        model, opt = build_model("frozen", 1, device)
        before = _snapshot(model)
        losses = _five_steps(model, opt, "frozen", train_ds, device, use_set_train_mode=use_stm)
        after = _snapshot(model)
        changes = _module_changes(before, after)
        pre = "layer1.0.bn1."
        out[variant] = {
            "losses": losses,
            "changes": changes,
            "max_abs_d_running_mean": (after[pre + "running_mean"] - before[pre + "running_mean"]).abs().max().item(),
            "max_abs_d_running_var": (after[pre + "running_var"] - before[pre + "running_var"]).abs().max().item(),
            "num_batches_tracked": (int(before[pre + "num_batches_tracked"]), int(after[pre + "num_batches_tracked"])),
            "running_var_rel_change": (
                (after[pre + "running_var"] - before[pre + "running_var"]).abs()
                / before[pre + "running_var"].abs()
            ).max().item(),
        }
        del model, opt

    bug = out["model.train()"]
    lines = [
        "| Модуль | Параметри змінилися | Ковзні статистики BN змінилися |",
        "| :-- | :-: | :-: |",
    ]
    for mod, (pc, bc) in bug["changes"].items():
        lines.append(f"| {mod} | {_yn(pc)} | {_yn(bc)} |")
    lines += [
        "",
        "| Величина для `layer1.0.bn1` після 5 кроків | set_train_mode | model.train() (помилка) |",
        "| :-- | --: | --: |",
        f"| max \\|Δ running_mean\\| | {out['set_train_mode']['max_abs_d_running_mean']:.6g} | "
        f"{bug['max_abs_d_running_mean']:.6g} |",
        f"| max \\|Δ running_var\\| | {out['set_train_mode']['max_abs_d_running_var']:.6g} | "
        f"{bug['max_abs_d_running_var']:.6g} |",
        f"| max відносна зміна running_var | {out['set_train_mode']['running_var_rel_change']:.4g} | "
        f"{bug['running_var_rel_change']:.4g} |",
        f"| num_batches_tracked до → після | {out['set_train_mode']['num_batches_tracked'][0]} → "
        f"{out['set_train_mode']['num_batches_tracked'][1]} | {bug['num_batches_tracked'][0]} → "
        f"{bug['num_batches_tracked'][1]} |",
        "",
        "| Крок | Втрата пакета, set_train_mode | Втрата пакета, model.train() |",
        "| --: | --: | --: |",
    ]
    for i, (a, b) in enumerate(zip(out["set_train_mode"]["losses"], bug["losses"]), 1):
        lines.append(f"| {i} | {a:.4f} | {b:.4f} |")
    return "\n".join(lines), out
