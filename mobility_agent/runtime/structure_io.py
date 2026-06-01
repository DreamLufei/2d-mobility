from __future__ import annotations

import os
from typing import Any, Dict

from pymatgen.core import Structure

from .normalize import normalize_mongo_extended_json


def structure_from_mongo_doc(doc: Dict[str, Any]) -> Structure:
    if "structure" not in doc:
        raise ValueError("文档缺少 structure 字段")

    struct_dict = normalize_mongo_extended_json(doc["structure"])
    # 兼容 pymatgen as_dict 格式
    if not isinstance(struct_dict, dict):
        raise ValueError("structure 字段不是 dict")

    try:
        return Structure.from_dict(struct_dict)
    except Exception as e:
        raise ValueError(f"Structure.from_dict 失败: {e}") from e


def write_poscar(structure: Structure, dest_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    # 保持数据库结构的晶格/坐标，不做额外变换
    poscar_str = structure.to(fmt="poscar")
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(poscar_str)
