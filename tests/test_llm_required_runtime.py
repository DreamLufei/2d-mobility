from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace

from mobility_agent.runtime.batch_config import BatchConfig
from mobility_agent.runtime.batch_runner import run_mongo_batch
from mobility_agent.runtime.context import RuntimeContext
from mobility_agent.runtime.runner import run_single_material
from tests.llm_test_utils import build_test_agent_runtime


def _prepare_material_root(tmpdir: str) -> None:
    with open(os.path.join(tmpdir, "POSCAR"), "w", encoding="utf-8") as handle:
        handle.write(
            "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 20.0\nSi\n2\nDirect\n0.0 0.0 0.5\n0.333333 0.666667 0.5\n"
        )
    with open(os.path.join(tmpdir, "POTCAR"), "w", encoding="utf-8") as handle:
        handle.write("FAKE POTCAR\n")


class LLMRequiredRuntimeTests(unittest.TestCase):
    def test_single_material_runtime_fails_fast_without_llm_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = RuntimeContext(
                agent_runtime=replace(build_test_agent_runtime(), llm_model="", specialist_model="", report_model=""),
                dry_run=True,
                store_path=os.path.join(tmpdir, "store.sqlite"),
                compatibility_export_enabled=False,
                compatibility_export_pickle=False,
            )
            with self.assertRaises(RuntimeError):
                run_single_material(runtime=runtime, material_id="llm-missing", root_path=tmpdir, fresh=True)

    def test_batch_runtime_fails_fast_without_llm_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = BatchConfig(
                mongo_uri="mongodb://example",
                mongo_db="db",
                mongo_collection="collection",
                batch_tag="llm-required-batch",
                runs_root=tmpdir,
                potcar_method="concat",
                vaspkit_cmd="vaspkit",
                vaspkit_task=103,
                potcar_root=tmpdir,
                potcar_map_path=None,
                retry_failed=False,
                running_stale_s=3600,
            )
            runtime = RuntimeContext(
                agent_runtime=replace(build_test_agent_runtime(), llm_model="", specialist_model="", report_model=""),
                dry_run=True,
                store_path=os.path.join(tmpdir, "store.sqlite"),
                compatibility_export_enabled=False,
                compatibility_export_pickle=False,
            )
            with self.assertRaises(RuntimeError):
                run_mongo_batch(cfg=cfg, runtime=runtime, thread_id="missing-llm", fresh_materials=True)


if __name__ == "__main__":
    unittest.main()
