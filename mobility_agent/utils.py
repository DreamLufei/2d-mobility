from __future__ import annotations

import os
from typing import Any, Iterable, TypeVar


T = TypeVar("T")


def dedupe_keep_order(items: Iterable[T] | None) -> list[T]:
    if items is None:
        return []
    seen: set[object] = set()
    result: list[T] = []
    for item in items:
        marker = item
        try:
            if marker in seen:
                continue
            seen.add(marker)
        except TypeError:
            marker = repr(item)
            if marker in seen:
                continue
            seen.add(marker)
        result.append(item)
    return result


def summarize_poscar(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"path": path, "exists": False, "atom_count": 0}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if len(lines) < 8:
        return {"path": path, "exists": True, "atom_count": 0, "warning": "poscar_too_short"}
    scale = float(lines[1].split()[0])
    lattice = []
    for idx in range(2, 5):
        parts = [float(v) for v in lines[idx].split()[:3]]
        lattice.append([scale * v for v in parts])
    species_line = lines[5].split()
    counts_line = lines[6].split()
    if all(part.replace("-", "").isdigit() for part in counts_line):
        counts = [int(v) for v in counts_line]
    else:
        counts = [int(v) for v in lines[7].split()]
    atom_count = int(sum(counts))
    return {
        "path": path,
        "exists": True,
        "atom_count": atom_count,
        "species": species_line,
        "counts": counts,
        "lattice": lattice,
    }