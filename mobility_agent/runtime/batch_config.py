from __future__ import annotations

import os
from dataclasses import dataclass

from ..env import ensure_project_env_loaded


@dataclass(frozen=True)
class BatchConfig:
    mongo_uri: str
    mongo_db: str
    mongo_collection: str

    batch_tag: str
    runs_root: str

    # POTCAR 生成方式：默认用 vaspkit -task 103
    potcar_method: str
    vaspkit_cmd: str
    vaspkit_task: int

    potcar_root: str | None
    potcar_map_path: str | None

    retry_failed: bool
    running_stale_s: int
    deprecation_warnings: tuple[str, ...] = ()


def load_config() -> BatchConfig:
    ensure_project_env_loaded()
    mongo_uri = os.environ.get("MONGO_URI", "").strip()
    if not mongo_uri:
        raise RuntimeError(
            "缺少环境变量 MONGO_URI。建议写入 .env/.env.local，或手动 export。"
        )

    mongo_db = os.environ.get("MONGO_DB", "materials_database").strip()
    mongo_collection = os.environ.get("MONGO_COLLECTION", "Vertical_NM_Sample_20").strip()

    batch_tag = os.environ.get("BATCH_TAG", "batch_run").strip()
    # 默认：以当前工作目录作为运行根目录（例如你在 test_mobality 目录下运行）
    runs_root = os.environ.get("RUNS_ROOT", os.getcwd()).strip()

    potcar_method = os.environ.get("POTCAR_METHOD", "vaspkit").strip().lower()
    vaspkit_cmd = os.environ.get("VASPKIT_CMD", "vaspkit").strip()
    vaspkit_task = int(os.environ.get("VASPKIT_TASK", "103") or "103")

    potcar_root = os.environ.get("POTCAR_ROOT")
    potcar_map_path = os.environ.get("POTCAR_MAP")

    retry_failed = (os.environ.get("RETRY_FAILED", "0").strip() in {"1", "true", "True"})

    # 如果上次异常退出，可能留下 running 状态；超过该时间视为可重新领取。
    running_stale_s = int(os.environ.get("RUNNING_STALE_S", str(12 * 3600)) or str(12 * 3600))
    deprecation_warnings: list[str] = []
    if os.environ.get("MOBALITY_SCRIPT") is not None:
        deprecation_warnings.append("deprecated_env_ignored:MOBALITY_SCRIPT")
    if os.environ.get("VASP_TIMEOUT_S") is not None:
        deprecation_warnings.append("deprecated_env_ignored:VASP_TIMEOUT_S")

    return BatchConfig(
        mongo_uri=mongo_uri,
        mongo_db=mongo_db,
        mongo_collection=mongo_collection,
        batch_tag=batch_tag,
        runs_root=os.path.abspath(runs_root),
        potcar_method=potcar_method,
        vaspkit_cmd=vaspkit_cmd,
        vaspkit_task=vaspkit_task,
        potcar_root=potcar_root,
        potcar_map_path=potcar_map_path,
        retry_failed=retry_failed,
        running_stale_s=running_stale_s,
        deprecation_warnings=tuple(dict.fromkeys(deprecation_warnings)),
    )
