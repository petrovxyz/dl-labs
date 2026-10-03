"""Виведення таблиць у консоль (rich) і в markdown."""
from dataclasses import dataclass, field
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

OK = "✓ так"
FAIL = "✗ ні"


def verdict(passed: bool) -> str:
    return OK if passed else FAIL


@dataclass
class Section:
    title: str
    headers: list[str]
    rows: list[list[str]]
    note: str = ""
    left_cols: set[int] = field(default_factory=lambda: {0})


def _style(cell: str) -> str:
    if cell == OK:
        return f"[bold green]{cell}[/]"
    if cell == FAIL:
        return f"[bold red]{cell}[/]"
    return cell


def show(console: Console, section: Section) -> None:
    table = Table(
        title=section.title,
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold",
        caption=section.note or None,
    )
    for i, header in enumerate(section.headers):
        table.add_column(header, justify="left" if i in section.left_cols else "right")
    for row in section.rows:
        table.add_row(*[_style(cell) for cell in row])
    console.print(table)
    console.print()


def to_markdown(section: Section) -> str:
    lines = [f"**{section.title}**", ""]
    lines.append("| " + " | ".join(section.headers) + " |")
    aligns = [":--" if i in section.left_cols else "--:" for i in range(len(section.headers))]
    lines.append("| " + " | ".join(aligns) + " |")
    for row in section.rows:
        lines.append("| " + " | ".join(row) + " |")
    if section.note:
        lines += ["", f"_{section.note}_"]
    return "\n".join(lines) + "\n"


def save_markdown(path: Path, heading: str, blocks: list) -> None:
    """blocks: Section або str (довільний markdown-текст)."""
    parts = [f"# {heading}\n"]
    for block in blocks:
        parts.append(to_markdown(block) if isinstance(block, Section) else block + "\n")
    path.write_text("\n".join(parts), encoding="utf-8")
