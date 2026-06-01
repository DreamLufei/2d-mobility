from __future__ import annotations

from typing import Any

from .base import SkillAwareAgent
from .schemas import BatchSummary


class BatchSupervisorAgent(SkillAwareAgent):
    agent_name = "batch_supervisor"

    def summarize(self, *, outcomes: list[dict[str, Any]]) -> BatchSummary:
        from .reporter import ReporterAgent

        return ReporterAgent(self.runtime, self.skills_root).summarize_batch(outcomes=outcomes)
