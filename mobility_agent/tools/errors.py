from __future__ import annotations


class MobilityRuntimeError(RuntimeError):
    """Base runtime exception for orchestration-level failures."""


class StageDependencyError(MobilityRuntimeError):
    """Raised when a stage is invoked without its declared inputs."""


class UnsupportedRecoveryActionError(MobilityRuntimeError):
    """Raised when a recovery decision chooses an unsupported action."""


class ManualFixValidationError(MobilityRuntimeError):
    """Raised when a manual-fix resume instruction is invalid."""


class CheckpointRestoreError(MobilityRuntimeError):
    """Raised when LangGraph checkpoint recovery is expected but durable state is unavailable."""
