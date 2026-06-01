from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from pymatgen.core import Structure


def _load_potcar_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("POTCAR_MAP 必须是 JSON dict: {\"Cd\": \"Cd\", ...}")
    return {str(k): str(v) for k, v in data.items()}


def _species_order(structure: Structure) -> List[str]:
    order: List[str] = []
    seen = set()
    for site in structure:
        el = site.specie.symbol
        if el not in seen:
            order.append(el)
            seen.add(el)
    return order


def build_potcar(
    structure: Structure,
    *,
    potcar_root: str,
    dest_path: str,
    potcar_map_path: Optional[str] = None,
) -> List[str]:
    """按元素顺序拼接 POTCAR。

    默认规则：使用 potcar_root/<symbol>/POTCAR
    可用 POTCAR_MAP (json) 覆盖: {"Ti": "Ti_pv", "Ba": "Ba_sv"}

    返回实际使用的 potcar 子目录名列表。
    """
    if not potcar_root:
        raise ValueError("potcar_root 为空")

    mapping = _load_potcar_map(potcar_map_path)
    elements = _species_order(structure)

    used: List[str] = []
    chunks: List[str] = []

    for el in elements:
        sub = mapping.get(el, el)
        potcar_file = os.path.join(potcar_root, sub, "POTCAR")
        if not os.path.exists(potcar_file):
            raise FileNotFoundError(f"未找到 POTCAR: {potcar_file} (元素 {el})")
        with open(potcar_file, "r", encoding="utf-8", errors="ignore") as f:
            chunks.append(f.read())
        used.append(sub)

    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(chunks))

    return used
