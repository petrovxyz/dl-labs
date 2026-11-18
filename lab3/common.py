"""Спільні константи й службові функції."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torchvision

LAB_DIR = Path(__file__).resolve().parent


class Paths:
    """Тека результатів: split.csv, runs/, lab3_model.pt, outputs/, figures/ (за замовчуванням – тека коду)."""

    def __init__(self, root: Path):
        self.set_root(root)

    def set_root(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.split_csv = self.root / "split.csv"
        self.runs_dir = self.root / "runs"
        self.fig_dir = self.root / "figures"
        self.out_dir = self.root / "outputs"
        self.model_path = self.root / "lab3_model.pt"


P = Paths(LAB_DIR)

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

EPOCHS = 8
TRAIN_BATCH = 32
EVAL_BATCH = 64
WEIGHT_DECAY = 1e-4

# Сім запусків: (номер, режим, зерно).
RUNS = [
    (1, "scratch", 1),
    (2, "frozen", 1),
    (3, "frozen", 2),
    (4, "frozen", 3),
    (5, "unfreeze", 1),
    (6, "unfreeze", 2),
    (7, "unfreeze", 3),
]


def run_dir(regime: str, seed: int) -> Path:
    return P.runs_dir / f"{regime}_seed{seed}"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda: {torch.cuda.get_device_name(device)}"
    if device.type == "mps":
        return f"mps: Apple {platform.machine()} ({_mac_chip()})"
    return f"cpu: {platform.processor() or platform.machine()}"


def _mac_chip() -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def git_info() -> dict:
    """Хеш коміту і незакомічені зміни коду (*.py, pyproject.toml, uv.lock) цієї теки."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=LAB_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty_files = git(
            "status", "--porcelain", "--", "*.py", "pyproject.toml", "uv.lock"
        ).splitlines()
    except Exception as exc:  # noqa: BLE001
        return {"commit": None, "dirty": None, "error": str(exc)}
    return {"commit": commit, "dirty": bool(dirty_files), "dirty_files": dirty_files}


def env_info(device: torch.device) -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "platform": platform.platform(),
        "device": device_name(device),
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_run(records: list[dict]) -> dict:
    """Вибір за validation: режим з найбільшим F̄, у ньому запуск з найбільшим F.

    records: [{"regime", "seed", "F"}]; за рівних F перевага меншому зерну.
    """
    by_regime: dict[str, list[dict]] = {}
    for r in records:
        by_regime.setdefault(r["regime"], []).append(r)
    means = {reg: sum(r["F"] for r in rs) / len(rs) for reg, rs in by_regime.items()}
    best_regime = max(means, key=lambda reg: means[reg])
    best = max(sorted(by_regime[best_regime], key=lambda r: r["seed"]), key=lambda r: r["F"])
    return {"regime": best_regime, "seed": best["seed"], "F": best["F"], "regime_means": means}
