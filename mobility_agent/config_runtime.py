from __future__ import annotations

import os
from dataclasses import dataclass

from .env import ensure_project_env_loaded
from .enums import DecisionEngine


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip() or default


def _env_csv(name: str, *aliases: str) -> tuple[str, ...]:
    for candidate in (name, *aliases):
        value = os.environ.get(candidate)
        if value is None:
            continue
        items = [part.strip() for part in value.split(",")]
        return tuple(part for part in items if part)
    return ()


def _env_int(name: str, default: int, *aliases: str) -> int:
    for candidate in (name, *aliases):
        value = os.environ.get(candidate)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return default


def _env_float(name: str, default: float, *aliases: str) -> float:
    for candidate in (name, *aliases):
        value = os.environ.get(candidate)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return default


def normalize_llm_provider(value: str | None, *, default: str = "openai") -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return str(default or "openai").strip().lower() or "openai"
    if raw in {"openai_compatible", "openai-compatible"}:
        return "openai"
    return raw


@dataclass(frozen=True)
class DecisionToggles:
    enable_admission_decision: bool = True
    enable_recovery_decision: bool = True
    enable_refinement_decision: bool = True
    enable_validation_decision: bool = True
    enable_confidence_scoring: bool = True
    allow_continue_with_warning: bool = True


@dataclass(frozen=True)
class AgentRuntimeConfig:
    decision_engine: DecisionEngine = DecisionEngine.LLM_REQUIRED
    llm_provider: str = "openai"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = ""
    llm_use_responses_api: bool = False
    llm_reasoning_effort: str = ""
    llm_provider_order: tuple[str, ...] = ()
    llm_provider_only: tuple[str, ...] = ()
    llm_provider_ignore: tuple[str, ...] = ()
    llm_provider_sort: str = ""
    llm_provider_allow_fallbacks: bool = True
    llm_provider_require_parameters: bool = False
    decision_provider: str = ""
    decision_base_url: str | None = None
    decision_api_key: str | None = None
    decision_model: str = ""
    specialist_model: str = ""
    report_model: str = ""
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 2
    llm_temperature: float = 0.0
    llm_max_tokens: int = 600
    orchestrator_provider: str = ""
    orchestrator_model: str = ""
    orchestrator_temperature: float = 0.0
    orchestrator_max_tokens: int = 800
    orchestrator_timeout_seconds: int = 45
    planner_provider: str = ""
    planner_model: str = ""
    planner_temperature: float = 0.0
    planner_max_tokens: int = 800
    planner_timeout_seconds: int = 45
    recovery_provider: str = ""
    recovery_model: str = ""
    recovery_temperature: float = 0.0
    recovery_max_tokens: int = 700
    recovery_timeout_seconds: int = 35
    critic_provider: str = ""
    critic_model: str = ""
    critic_temperature: float = 0.0
    critic_max_tokens: int = 700
    critic_timeout_seconds: int = 45
    physics_judge_provider: str = ""
    physics_judge_model: str = ""
    physics_judge_temperature: float = 0.0
    physics_judge_max_tokens: int = 700
    physics_judge_timeout_seconds: int = 45
    cost_guardian_provider: str = ""
    cost_guardian_model: str = ""
    cost_guardian_temperature: float = 0.0
    cost_guardian_max_tokens: int = 600
    cost_guardian_timeout_seconds: int = 30
    reporter_provider: str = ""
    reporter_model: str = ""
    reporter_temperature: float = 0.0
    reporter_max_tokens: int = 700
    reporter_timeout_seconds: int = 30
    executor_provider: str = ""
    executor_model: str = ""
    executor_temperature: float = 0.0
    executor_max_tokens: int = 500
    executor_timeout_seconds: int = 25
    decision_log_level: str = "standard"
    max_refinement_rounds: int = 1
    strain_min_points: int = 5
    strain_target_points: int = 9
    fit_r2_threshold: float = 0.90
    e1_sigma_threshold: float = 0.5
    c2d_sigma_threshold: float = 10.0
    gap_min_for_admission: float = 0.05
    allow_continue_with_warning: bool = True
    human_review_enabled: bool = True
    human_review_timeout_seconds: int = 300
    human_review_poll_seconds: int = 5
    human_review_default_action: str = "skip_material"
    human_review_email_enabled: bool = False
    human_review_email_to: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    smtp_use_tls: bool = True
    toggles: DecisionToggles = DecisionToggles()

    @property
    def llm_timeout_s(self) -> int:
        return int(self.llm_timeout_seconds)

    def resolve_role_config(self, role: str | None) -> dict[str, object]:
        normalized = str(role or "specialist").strip().lower()
        if normalized == "report":
            normalized = "reporter"
        critical_roles = {"planner", "orchestrator", "critic", "physics_judge", "cost_guardian", "recovery"}
        prefer_decision_stack = normalized in critical_roles and bool(
            str(self.decision_model or "").strip() or str(self.decision_provider or "").strip()
        )
        default_provider = self.decision_provider if prefer_decision_stack else self.llm_provider
        provider = normalize_llm_provider(str(getattr(self, f"{normalized}_provider", "") or default_provider or ""))
        model = str(getattr(self, f"{normalized}_model", "") or "").strip()
        if not model:
            if prefer_decision_stack:
                model = str(self.decision_model or self.specialist_model or self.llm_model or "").strip()
            elif normalized == "reporter":
                model = str(self.report_model or self.specialist_model or self.llm_model or "").strip()
            elif normalized in {"orchestrator", "planner", "critic", "physics_judge"}:
                model = str(self.specialist_model or self.llm_model or "").strip()
            elif normalized in {"recovery", "cost_guardian", "executor"}:
                model = str(self.specialist_model or self.llm_model or "").strip()
            else:
                model = str(self.specialist_model or self.llm_model or "").strip()
        base_url = self.decision_base_url if prefer_decision_stack and self.decision_base_url else self.llm_base_url
        api_key = self.decision_api_key if prefer_decision_stack and self.decision_api_key else self.llm_api_key
        temperature = float(getattr(self, f"{normalized}_temperature", self.llm_temperature) or self.llm_temperature)
        max_tokens = int(getattr(self, f"{normalized}_max_tokens", self.llm_max_tokens) or self.llm_max_tokens)
        timeout_seconds = int(
            getattr(self, f"{normalized}_timeout_seconds", self.llm_timeout_seconds) or self.llm_timeout_seconds
        )
        return {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout_seconds": timeout_seconds,
            "use_responses_api": bool(self.llm_use_responses_api),
            "reasoning_effort": str(self.llm_reasoning_effort or "").strip(),
            "provider_preferences": {
                "order": list(self.llm_provider_order),
                "only": list(self.llm_provider_only),
                "ignore": list(self.llm_provider_ignore),
                "sort": str(self.llm_provider_sort or "").strip(),
                "allow_fallbacks": bool(self.llm_provider_allow_fallbacks),
                "require_parameters": bool(self.llm_provider_require_parameters),
            },
        }

    def validate_llm_required(self) -> None:
        from .agents.llm_client import validate_runtime_llm_config

        validate_runtime_llm_config(self)

    @classmethod
    def from_env(cls) -> "AgentRuntimeConfig":
        ensure_project_env_loaded()
        raw_planner_mode = str(os.environ.get("PLANNER_MODE") or "").strip()
        if raw_planner_mode:
            raise RuntimeError(
                "PLANNER_MODE is no longer supported. "
                "This runtime has a single LLM-required decision philosophy; remove PLANNER_MODE from your environment."
            )
        raw_llm_enabled = str(os.environ.get("LLM_ENABLED") or "").strip()
        if raw_llm_enabled:
            raise RuntimeError(
                "LLM_ENABLED is no longer supported. "
                "This runtime is LLM-required; remove LLM_ENABLED and configure LLM_PROVIDER, LLM_BASE_URL, "
                "LLM_API_KEY, and LLM_MODEL instead."
            )
        allow_warning = _env_bool("ALLOW_CONTINUE_WITH_WARNING", True)
        toggles = DecisionToggles(
            enable_admission_decision=_env_bool("ENABLE_ADMISSION_DECISION", True),
            enable_recovery_decision=_env_bool("ENABLE_RECOVERY_DECISION", True),
            enable_refinement_decision=_env_bool("ENABLE_REFINEMENT_DECISION", True),
            enable_validation_decision=_env_bool("ENABLE_VALIDATION_DECISION", True),
            enable_confidence_scoring=_env_bool("ENABLE_CONFIDENCE_SCORING", True),
            allow_continue_with_warning=allow_warning,
        )
        cfg = cls(
            decision_engine=DecisionEngine.LLM_REQUIRED,
            llm_provider=normalize_llm_provider(_env_str("LLM_PROVIDER", "openai")),
            llm_base_url=(_env_str("LLM_BASE_URL") or None),
            llm_api_key=(_env_str("LLM_API_KEY") or None),
            llm_model=_env_str("LLM_MODEL", ""),
            llm_use_responses_api=_env_bool("LLM_USE_RESPONSES_API", False),
            llm_reasoning_effort=_env_str("LLM_REASONING_EFFORT", ""),
            llm_provider_order=_env_csv("LLM_PROVIDER_ORDER"),
            llm_provider_only=_env_csv("LLM_PROVIDER_ONLY"),
            llm_provider_ignore=_env_csv("LLM_PROVIDER_IGNORE"),
            llm_provider_sort=_env_str("LLM_PROVIDER_SORT", ""),
            llm_provider_allow_fallbacks=_env_bool("LLM_PROVIDER_ALLOW_FALLBACKS", True),
            llm_provider_require_parameters=_env_bool("LLM_PROVIDER_REQUIRE_PARAMETERS", False),
            decision_provider=normalize_llm_provider(
                _env_str("DECISION_LLM_PROVIDER", _env_str("DECISION_PROVIDER", "")),
                default="",
            ),
            decision_base_url=(_env_str("DECISION_LLM_BASE_URL", _env_str("DECISION_BASE_URL")) or None),
            decision_api_key=(_env_str("DECISION_LLM_API_KEY", _env_str("DECISION_API_KEY")) or None),
            decision_model=_env_str("DECISION_LLM_MODEL", _env_str("DECISION_MODEL", "")),
            specialist_model=_env_str("SPECIALIST_MODEL", ""),
            report_model=_env_str("REPORT_MODEL", ""),
            llm_timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 30, "LLM_TIMEOUT_S"),
            llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", 600),
            orchestrator_provider=_env_str("ORCHESTRATOR_PROVIDER", ""),
            orchestrator_model=_env_str("ORCHESTRATOR_MODEL", ""),
            orchestrator_temperature=_env_float("ORCHESTRATOR_TEMPERATURE", 0.0),
            orchestrator_max_tokens=_env_int("ORCHESTRATOR_MAX_TOKENS", 800),
            orchestrator_timeout_seconds=_env_int("ORCHESTRATOR_TIMEOUT_SECONDS", 45),
            planner_provider=_env_str("PLANNER_PROVIDER", ""),
            planner_model=_env_str("PLANNER_MODEL", ""),
            planner_temperature=_env_float("PLANNER_TEMPERATURE", 0.0),
            planner_max_tokens=_env_int("PLANNER_MAX_TOKENS", 800),
            planner_timeout_seconds=_env_int("PLANNER_TIMEOUT_SECONDS", 45),
            recovery_provider=_env_str("RECOVERY_PROVIDER", ""),
            recovery_model=_env_str("RECOVERY_MODEL", ""),
            recovery_temperature=_env_float("RECOVERY_TEMPERATURE", 0.0),
            recovery_max_tokens=_env_int("RECOVERY_MAX_TOKENS", 700),
            recovery_timeout_seconds=_env_int("RECOVERY_TIMEOUT_SECONDS", 35),
            critic_provider=_env_str("CRITIC_PROVIDER", ""),
            critic_model=_env_str("CRITIC_MODEL", ""),
            critic_temperature=_env_float("CRITIC_TEMPERATURE", 0.0),
            critic_max_tokens=_env_int("CRITIC_MAX_TOKENS", 700),
            critic_timeout_seconds=_env_int("CRITIC_TIMEOUT_SECONDS", 45),
            physics_judge_provider=_env_str("PHYSICS_JUDGE_PROVIDER", ""),
            physics_judge_model=_env_str("PHYSICS_JUDGE_MODEL", ""),
            physics_judge_temperature=_env_float("PHYSICS_JUDGE_TEMPERATURE", 0.0),
            physics_judge_max_tokens=_env_int("PHYSICS_JUDGE_MAX_TOKENS", 700),
            physics_judge_timeout_seconds=_env_int("PHYSICS_JUDGE_TIMEOUT_SECONDS", 45),
            cost_guardian_provider=_env_str("COST_GUARDIAN_PROVIDER", ""),
            cost_guardian_model=_env_str("COST_GUARDIAN_MODEL", ""),
            cost_guardian_temperature=_env_float("COST_GUARDIAN_TEMPERATURE", 0.0),
            cost_guardian_max_tokens=_env_int("COST_GUARDIAN_MAX_TOKENS", 600),
            cost_guardian_timeout_seconds=_env_int("COST_GUARDIAN_TIMEOUT_SECONDS", 30),
            reporter_provider=_env_str("REPORTER_PROVIDER", ""),
            reporter_model=_env_str("REPORTER_MODEL", ""),
            reporter_temperature=_env_float("REPORTER_TEMPERATURE", 0.0),
            reporter_max_tokens=_env_int("REPORTER_MAX_TOKENS", 700),
            reporter_timeout_seconds=_env_int("REPORTER_TIMEOUT_SECONDS", 30),
            executor_provider=_env_str("EXECUTOR_PROVIDER", ""),
            executor_model=_env_str("EXECUTOR_MODEL", ""),
            executor_temperature=_env_float("EXECUTOR_TEMPERATURE", 0.0),
            executor_max_tokens=_env_int("EXECUTOR_MAX_TOKENS", 500),
            executor_timeout_seconds=_env_int("EXECUTOR_TIMEOUT_SECONDS", 25),
            decision_log_level=_env_str("DECISION_LOG_LEVEL", "standard"),
            max_refinement_rounds=_env_int("MAX_REFINEMENT_ROUNDS", 1),
            strain_min_points=_env_int("STRAIN_MIN_POINTS", 5),
            strain_target_points=_env_int("STRAIN_TARGET_POINTS", 9),
            fit_r2_threshold=_env_float("FIT_R2_THRESHOLD", 0.90),
            e1_sigma_threshold=_env_float("E1_SIGMA_THRESHOLD", 0.5),
            c2d_sigma_threshold=_env_float("C2D_SIGMA_THRESHOLD", 10.0),
            gap_min_for_admission=_env_float("GAP_MIN_FOR_ADMISSION", 0.05),
            allow_continue_with_warning=allow_warning,
            human_review_enabled=_env_bool("ENABLE_HUMAN_REVIEW", True),
            human_review_timeout_seconds=_env_int("HUMAN_REVIEW_TIMEOUT_SECONDS", 300, "HUMAN_REVIEW_TIMEOUT_S"),
            human_review_poll_seconds=_env_int("HUMAN_REVIEW_POLL_SECONDS", 5, "HUMAN_REVIEW_POLL_S"),
            human_review_default_action=_env_str("HUMAN_REVIEW_DEFAULT_ACTION", "skip_material"),
            human_review_email_enabled=_env_bool("ENABLE_EMAIL_NOTIFICATIONS", False),
            human_review_email_to=(_env_str("EMAIL_NOTIFY_TO") or None),
            smtp_host=(_env_str("SMTP_HOST") or None),
            smtp_port=_env_int("SMTP_PORT", 587),
            smtp_username=(_env_str("SMTP_USERNAME") or None),
            smtp_password=(_env_str("SMTP_PASSWORD") or None),
            smtp_from=(_env_str("SMTP_FROM") or None),
            smtp_use_tls=_env_bool("SMTP_USE_TLS", True),
            toggles=toggles,
        )
        cfg.validate_llm_required()
        return cfg
