from __future__ import annotations

from enum import Enum


class DecisionEngine(str, Enum):
    LLM_REQUIRED = "llm_required"


# Compatibility alias for older imports. The runtime no longer supports
# multiple planner modes; the only canonical decision engine is LLM-required.
PlannerMode = DecisionEngine


class StatusLabel(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    LOW_CONFIDENCE = "low_confidence"


class DecisionKind(str, Enum):
    ADMISSION = "admission"
    RECOVERY = "recovery"
    REFINEMENT = "refinement"
    VALIDATION = "validation"
