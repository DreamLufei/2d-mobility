from __future__ import annotations

from typing import Any


_EXT_NUMBER_KEYS = (
    "$numberDouble",
    "$numberInt",
    "$numberLong",
    "$numberDecimal",
)


def normalize_mongo_extended_json(obj: Any) -> Any:
    """把 MongoDB Extended JSON(如 $numberDouble/$oid) 递归还原为 Python 原生类型。"""
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, list):
        return [normalize_mongo_extended_json(x) for x in obj]

    if isinstance(obj, dict):
        # 单键的扩展 JSON
        if len(obj) == 1:
            (k, v), = obj.items()
            if k in _EXT_NUMBER_KEYS:
                try:
                    return float(v)
                except Exception:
                    return v
            if k == "$oid":
                return str(v)
            if k == "$date":
                # 可能是 ISO 字符串或 {"$numberLong": "..."}
                return normalize_mongo_extended_json(v)

        return {str(k): normalize_mongo_extended_json(v) for k, v in obj.items()}

    return obj
