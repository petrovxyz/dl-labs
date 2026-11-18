"""Модель, режими навчання, set_train_mode, оптимізатор, цикл навчання й оцінка."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from common import EPOCHS, NUM_CLASSES, TRAIN_BATCH, WEIGHT_DECAY

TOP_MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]

# Режими: ваги, модулі, що навчаються, групи AdamW (модулі, швидкість навчання), модулі в eval().
REGIMES = {
    "scratch": {
        "weights": None,
        "trainable": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"],
        "groups": [(["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"], 3e-3)],
        "eval_modules": [],
        "expected_trainable": 11_179_590,
    },
    "frozen": {
        "weights": "IMAGENET1K_V1",
        "trainable": ["fc"],
        "groups": [(["fc"], 3e-3)],
        "eval_modules": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
        "expected_trainable": 3_078,
    },
    "unfreeze": {
        "weights": "IMAGENET1K_V1",
        "trainable": ["layer4", "fc"],
        "groups": [(["fc"], 3e-3), (["layer4"], 3e-4)],
        "eval_modules": ["conv1", "bn1", "layer1", "layer2", "layer3"],
        "expected_trainable": 8_396_806,
    },
}
TOTAL_PARAMS = 11_179_590


def build_model(regime: str, seed: int, device: torch.device):
    """Зерна → модель → нова голова → заморожування → пристрій і AdamW."""
    cfg = REGIMES[regime]
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = ResNet18_Weights.IMAGENET1K_V1 if cfg["weights"] else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(512, NUM_CLASSES)
    for name, module in model.named_children():
        trainable = name in cfg["trainable"]
        for p in module.parameters():
            p.requires_grad_(trainable)
    # Оптимізатор - після заміни fc і лише з параметрами, що навчаються.
    model.to(device)
    groups = []
    for modules, lr in cfg["groups"]:
        params = [p for m in modules for p in getattr(model, m).parameters()]
        groups.append({"params": params, "lr": lr, "name": "+".join(modules)})
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    return model, optimizer


def frozen_modules(model: nn.Module, regime: str) -> list[nn.Module]:
    """Модулі, що під час навчання режиму мають бути в eval()."""
    return [getattr(model, name) for name in REGIMES[regime]["eval_modules"]]


def set_train_mode(model: nn.Module, frozen_modules: list[nn.Module]) -> None:
    """model.train(), потім eval() для заморожених модулів; викликається на початку кожної епохи.

    Інакше BN заморожених модулів оновлює ковзні статистики, хоча requires_grad=False.
    """
    model.train()
    for m in frozen_modules:
        m.eval()


def count_params(params) -> int:
    return sum(p.numel() for p in params)


def model_settings(regime: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    """Налаштування моделі, режиму й оптимізатора, включно зі значеннями за замовчуванням."""
    cfg = REGIMES[regime]
    bn = model.bn1
    groups = []
    for g in optimizer.param_groups:
        hp = {k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items() if k != "params"}
        hp["num_params"] = count_params(g["params"])
        groups.append(hp)
    return {
        "regime": regime,
        "architecture": "torchvision.models.resnet18",
        "weights": "ResNet18_Weights.IMAGENET1K_V1" if cfg["weights"] else None,
        "fc": "torch.nn.Linear(512, 6), стандартна ініціалізація",
        "zero_init_residual": False,
        "trainable_modules": cfg["trainable"],
        "eval_modules_during_training": cfg["eval_modules"],
        "total_params": count_params(model.parameters()),
        "trainable_params": count_params(p for p in model.parameters() if p.requires_grad),
        "batchnorm": {"momentum": bn.momentum, "eps": bn.eps, "affine": bn.affine,
                      "track_running_stats": bn.track_running_stats},
        "optimizer": {"name": "torch.optim.AdamW", "param_groups": groups},
        "loss": {"name": "torch.nn.CrossEntropyLoss", "reduction": "mean", "label_smoothing": 0.0,
                 "weight": None},
        "lr_scheduler": None,
        "grad_clipping": None,
        "mixed_precision": False,
        "dtype": "float32",
        "early_stopping": False,
        "epoch_selection_by_validation": False,
    }


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def macro_f1(labels: np.ndarray, logits: np.ndarray) -> float:
    return float(f1_score(labels, logits.argmax(1), average="macro", labels=list(range(NUM_CLASSES))))


def train_run(regime: str, seed: int, train_ds, val_loader, device: torch.device, log=print):
    """Один запуск: 8 епох, validation після кожної. Повертає (model, optimizer, history, val_logits, seconds)."""
    from data import train_loader  # уникаємо циклічного імпорту

    model, optimizer = build_model(regime, seed, device)
    frozen = frozen_modules(model, regime)
    loader = train_loader(train_ds)
    criterion = nn.CrossEntropyLoss()
    n = len(train_ds)
    history = {"train_loss": [], "val_macro_f1": [], "steps": 0}

    train_seconds = 0.0
    sync(device)
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        sync(device)
        t0 = time.perf_counter()
        set_train_mode(model, frozen)
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)  # втрата пакета до кроку, зважена розміром пакета
            history["steps"] += 1
        sync(device)
        train_seconds += time.perf_counter() - t0
        val_logits, val_labels = predict_logits(model, val_loader, device)
        f1 = macro_f1(val_labels, val_logits)
        history["train_loss"].append(total / n)
        history["val_macro_f1"].append(f1)
        log(f"  {regime} seed{seed} епоха {epoch}: втрата train {total / n:.4f}, macro-F1 val {f1:.4f}")
    expected_steps = EPOCHS * math.ceil(n / TRAIN_BATCH)  # 8 × 38 = 304
    assert history["steps"] == expected_steps, (history["steps"], expected_steps)
    sync(device)
    seconds = {"train_steps": train_seconds, "total_with_validation": time.perf_counter() - start}
    return model, optimizer, history, val_logits, seconds


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
