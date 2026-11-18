"""Поділ, набір даних, перетворення й завантажувачі."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import CLASSES, EVAL_BATCH, IMG_SIZE, MEAN, STD, SPLIT_CSV, TRAIN_BATCH

TRAIN_PER_CLASS = 200
PARTS = ("train", "validation", "test")

# Контрольна таблиця з умови: (кількість, сума числових імен файлів).
CONTROL = {
    "buildings": {"train": (200, 1_996_761), "validation": (218, 4_837_323), "test": (219, 4_837_559)},
    "forest": {"train": (200, 2_032_625), "validation": (237, 5_276_962), "test": (237, 5_291_326)},
    "glacier": {"train": (200, 2_183_923), "validation": (276, 6_135_652), "test": (277, 6_157_250)},
    "mountain": {"train": (200, 1_950_345), "validation": (262, 5_789_535), "test": (263, 5_849_765)},
    "sea": {"train": (200, 1_987_613), "validation": (255, 5_655_068), "test": (255, 5_668_011)},
    "street": {"train": (200, 1_841_817), "validation": (250, 5_541_845), "test": (251, 5_547_533)},
}
CONTROL_TOTAL = {"train": (1200, 11_993_084), "validation": (1498, 33_236_385), "test": (1502, 33_351_444)}


def train_dir(data_root: Path) -> Path:
    return Path(data_root) / "seg_train" / "seg_train"


def test_dir(data_root: Path) -> Path:
    return Path(data_root) / "seg_test" / "seg_test"


def sorted_files(folder: Path) -> list[str]:
    """Імена файлів теки, відсортовані як рядки (як sorted() і ImageFolder)."""
    return sorted(p.name for p in folder.iterdir() if p.is_file())


def build_split(data_root: Path) -> list[dict]:
    """Поділ строго за процедурою з умови. Повертає рядки split.csv."""
    rows: list[dict] = []

    # train: один генератор на всі класи по черзі.
    rng = np.random.default_rng(0)
    for label in CLASSES:
        files = sorted_files(train_dir(data_root) / label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        for i in idx[:TRAIN_PER_CLASS]:
            rows.append({"file": files[i], "label": label, "part": "train"})

    # validation і test: новий генератор, теж один на всі класи по черзі.
    rng = np.random.default_rng(0)
    for label in CLASSES:
        files = sorted_files(test_dir(data_root) / label)
        n_k = len(files)
        idx = np.arange(n_k)
        rng.shuffle(idx)
        cut = n_k // 2
        for i in idx[:cut]:
            rows.append({"file": files[i], "label": label, "part": "validation"})
        for i in idx[cut:]:
            rows.append({"file": files[i], "label": label, "part": "test"})
    return rows


def write_split(rows: list[dict], path: Path = SPLIT_CSV) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "part"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_split(path: Path = SPLIT_CSV) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def part_rows(rows: list[dict], part: str) -> list[dict]:
    """Рядки однієї частини в порядку split.csv."""
    return [r for r in rows if r["part"] == part]


def control_table(rows: list[dict]) -> dict:
    """{клас: {частина: (кількість, сума числових імен)}} плюс рядок 'Разом'."""
    table = {c: {p: [0, 0] for p in PARTS} for c in CLASSES}
    for r in rows:
        cell = table[r["label"]][r["part"]]
        cell[0] += 1
        cell[1] += int(Path(r["file"]).stem)
    total = {p: [sum(table[c][p][0] for c in CLASSES), sum(table[c][p][1] for c in CLASSES)] for p in PARTS}
    result = {c: {p: tuple(table[c][p]) for p in PARTS} for c in CLASSES}
    result["Разом"] = {p: tuple(total[p]) for p in PARTS}
    return result


def check_split(rows: list[dict]) -> tuple[bool, str]:
    """Звірка з контрольною таблицею і перевірка, що жоден файл не входить у дві частини."""
    table = control_table(rows)
    expected = dict(CONTROL)
    expected["Разом"] = CONTROL_TOTAL

    fmt = lambda cell: f"{cell[0]} / {cell[1]:,}".replace(",", " ")  # noqa: E731
    lines = [
        "| Клас | train | validation | test | Збіг |",
        "| :-- | --: | --: | --: | :-: |",
    ]
    all_ok = True
    for name in [*CLASSES, "Разом"]:
        ok = all(table[name][p] == tuple(expected[name][p]) for p in PARTS)
        all_ok &= ok
        cells = []
        for p in PARTS:
            mark = "" if table[name][p] == tuple(expected[name][p]) else f" (очікувано {fmt(expected[name][p])})"
            cells.append(fmt(table[name][p]) + mark)
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {'так' if ok else 'НІ'} |")

    names = [r["file"] for r in rows]
    unique = len(set(names)) == len(names)
    keys = [(r["label"], r["file"]) for r in rows]
    overlap = {p: set() for p in PARTS}
    for r in rows:
        overlap[r["part"]].add((r["label"], r["file"]))
    pairs = {
        f"{a} ∩ {b}": len(overlap[a] & overlap[b])
        for i, a in enumerate(PARTS)
        for b in PARTS[i + 1:]
    }
    disjoint = all(v == 0 for v in pairs.values()) and len(set(keys)) == len(keys)

    lines.append("")
    lines.append(f"Усі 18 клітинок і підсумки збігаються з контрольною таблицею: **{'так' if all_ok else 'НІ'}**.")
    lines.append(
        "Перетини частин: " + ", ".join(f"{k} = {v}" for k, v in pairs.items())
        + f"; усі імена файлів унікальні: {'так' if unique else 'ні'}."
    )
    return all_ok and disjoint and unique, "\n".join(lines)


# ---------------------------------------------------------------- перетворення

def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def transform_settings() -> dict:
    """Повні налаштування перетворень, включно зі значеннями за замовчуванням."""
    rrc = transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0))
    flip = transforms.RandomHorizontalFlip()
    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
    return {
        "train": [
            {"name": "RandomResizedCrop", "size": list(rrc.size), "scale": list(rrc.scale),
             "ratio": [float(x) for x in rrc.ratio], "interpolation": rrc.interpolation.value,
             "antialias": rrc.antialias},
            {"name": "RandomHorizontalFlip", "p": flip.p},
            {"name": "ToTensor"},
            {"name": "Normalize", "mean": list(MEAN), "std": list(STD)},
        ],
        "validation_test": [
            {"name": "Resize", "size": list(resize.size), "interpolation": resize.interpolation.value,
             "max_size": resize.max_size, "antialias": resize.antialias},
            {"name": "ToTensor"},
            {"name": "Normalize", "mean": list(MEAN), "std": list(STD)},
        ],
        "test_time_augmentation": False,
    }


# ---------------------------------------------------------------- набір даних

class SceneDataset(Dataset):
    """Зображення частини поділу в порядку split.csv.

    Зображення відкриваються через Pillow з .convert("RGB") один раз і тримаються в пам'яті
    (у незміненому вигляді); випадкові перетворення застосовуються заново при кожному зверненні.
    """

    def __init__(self, rows: list[dict], data_root: Path, transform):
        self.rows = rows
        self.transform = transform
        self.labels = [CLASSES.index(r["label"]) for r in rows]
        self.images = []
        for r in rows:
            folder = train_dir(data_root) if r["part"] == "train" else test_dir(data_root)
            with Image.open(folder / r["label"] / r["file"]) as im:
                self.images.append(im.convert("RGB"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        return self.transform(self.images[i]), self.labels[i]


def load_part(data_root: Path, part: str, rows: list[dict] | None = None) -> SceneDataset:
    if part == "test":
        raise RuntimeError("test завантажується лише командою аналізу (load_test_set)")
    rows = read_split() if rows is None else rows
    transform = train_transform() if part == "train" else eval_transform()
    return SceneDataset(part_rows(rows, part), data_root, transform)


def load_test_set(data_root: Path) -> SceneDataset:
    """Єдина точка завантаження test; викликається лише з analysis.py."""
    return SceneDataset(part_rows(read_split(), "test"), data_root, eval_transform())


def train_loader(ds: SceneDataset) -> DataLoader:
    # Без власного generator чи sampler: перемішування й аугментації беруть випадковість
    # із глобального генератора PyTorch, засіяного під час побудови моделі.
    return DataLoader(ds, batch_size=TRAIN_BATCH, shuffle=True, drop_last=False, num_workers=0)


def eval_loader(ds: SceneDataset) -> DataLoader:
    return DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=0)


def loader_settings() -> dict:
    return {
        "train": {"batch_size": TRAIN_BATCH, "shuffle": True, "drop_last": False, "num_workers": 0,
                  "generator": None, "sampler": None},
        "validation_test": {"batch_size": EVAL_BATCH, "shuffle": False, "drop_last": False,
                            "num_workers": 0, "order": "як у split.csv"},
    }


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)
