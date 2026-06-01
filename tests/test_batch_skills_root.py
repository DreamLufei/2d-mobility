from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from mobility_agent.runtime.batch_config import BatchConfig
from mobility_agent.runtime.batch_runner import _batch_nodes
from mobility_agent.runtime.context import RuntimeContext
from tests.llm_test_utils import build_test_agent_runtime


class BatchSkillsRootTests(unittest.TestCase):
    def test_batch_supervisor_uses_runtime_skills_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = BatchConfig(
                mongo_uri="mongodb://example",
                mongo_db="db",
                mongo_collection="collection",
                batch_tag="test-batch",
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
                agent_runtime=build_test_agent_runtime(),
                hitl_policy="non_interactive_skip_on_timeout",
                dry_run=True,
                store_path=os.path.join(tmpdir, "batch_store.sqlite"),
                compatibility_export_enabled=False,
                compatibility_export_pickle=False,
                skills_root=os.path.join(tmpdir, "custom-skills"),
            )
            with patch("mobility_agent.runtime.batch_runner.BatchSupervisorAgent") as supervisor_cls:
                _batch_nodes(cfg=cfg, runtime=runtime, fresh_materials=True)
            self.assertEqual(supervisor_cls.call_args.args[1], os.path.abspath(runtime.skills_root))


if __name__ == "__main__":
    unittest.main()
