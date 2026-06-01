import { useEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from "react";
import { toPng } from "html-to-image";

type JobSummary = {
  job_id: string;
  job_type: string;
  job_role: string;
  display_name: string;
  material_id?: string | null;
  batch_tag?: string | null;
  root_path: string;
  workdir?: string | null;
  runtime_run_status: string;
  control_plane_status: string;
  final_acceptance?: string | null;
  quality_grade?: string | null;
  current_stage?: string | null;
  error_summary?: string | null;
  child_job_ids?: string[];
  last_progress_line?: string | null;
  last_state_updated_at?: string | null;
  last_heartbeat_at?: string | null;
};

type JobDetail = JobSummary & {
  pid?: number | null;
  pgid?: number | null;
  thread_id?: string | null;
  hitl_pending: boolean;
  wait_reason?: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  state: Record<string, unknown>;
  artifacts: Record<string, string>;
  summary: Record<string, unknown>;
  logs?: LogsResponse;
  timeline?: TimelineEvent[];
  children?: JobSummary[];
  failure_taxonomy?: Record<string, number>;
};

type TimelineEvent = {
  timestamp: string;
  event_type: string;
  current_stage?: string;
  runtime_run_status?: string;
  selected_action_family?: string;
  selected_capability?: string;
  stage_status?: string;
  latest_error?: string | null;
};

type LogsResponse = {
  total: number;
  offset: number;
  limit: number;
  lines: string[];
};

type RuntimeSettings = {
  service_preset: string;
  mobility_db_uri: string;
  llm_provider: string;
  llm_base_url: string;
  llm_model: string;
  llm_api_key_present: boolean;
  llm_api_key_preview?: string | null;
  embedding_model: string;
  embedding_base_url: string;
  embedding_api_key_present: boolean;
  embedding_api_key_preview?: string | null;
  wiki_qa_model: string;
  agentic_policy_enabled: boolean;
  policy_allowlist_mode: string;
  policy_retrieval_top_k: number;
  policy_trace_enabled: boolean;
  rag_top_k: number;
  rag_chunk_size: number;
  rag_chunk_overlap: number;
  rag_reindex_batch_size: number;
  hitl_policy: string;
  human_review_timeout_seconds: number;
  human_review_default_action: string;
  enable_email_notifications: boolean;
  email_notify_to: string;
  email_dry_run: boolean;
  smtp_host: string;
  smtp_port: number;
  smtp_use_tls: boolean;
  smtp_username: string;
  smtp_from: string;
  smtp_password_present: boolean;
  smtp_password_preview?: string | null;
};

type ArtifactPreviewMap = Record<string, unknown>;

type Route =
  | { kind: "home" }
  | { kind: "wiki" }
  | { kind: "jobs" }
  | { kind: "job"; jobId: string };

const apiBase = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? "";
const STAGE_ORDER = ["prepare", "relax", "scf", "band", "effective_mass", "strain_loop", "mobility", "validation", "final_report"] as const;
const AGENTIC_ARTIFACT_KEYS = ["parameter_plan_path", "retrieval_trace_path", "recovery_diagnosis_path", "final_summary_path", "validation_report_path", "material_outcome_path"] as const;

function normalizeLlmProvider(value: string | null | undefined): string {
  const raw = String(value || "").trim().toLowerCase();
  if (!raw) {
    return "openai";
  }
  return raw === "openai_compatible" ? "openai" : raw;
}

function routeFromLocation(): Route {
  const segments = window.location.pathname.split("/").filter(Boolean);
  const appIndex = segments[0] === "app" ? 1 : 0;
  const rest = segments.slice(appIndex);
  if (rest.length === 0) {
    return { kind: "home" };
  }
  if (rest[0] === "wiki") {
    return { kind: "wiki" };
  }
  if (rest[0] === "jobs" && rest.length === 1) {
    return { kind: "jobs" };
  }
  if (rest[0] === "jobs" && rest[1]) {
    return { kind: "job", jobId: rest[1] };
  }
  return { kind: "home" };
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers ?? {}),
    },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

function wsUrl(path: string): string {
  const origin = apiBase || window.location.origin;
  const url = new URL(origin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = path;
  url.search = "";
  return url.toString();
}

function navigate(path: string): void {
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

function formatTime(value?: string | null): string {
  if (!value) return "n/a";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function summarizeCounts(jobs: JobSummary[]): Record<string, number> {
  return jobs.reduce<Record<string, number>>((acc, job) => {
    acc[job.runtime_run_status] = (acc[job.runtime_run_status] ?? 0) + 1;
    return acc;
  }, {});
}

function displayStageName(value: string): string {
  return value.replaceAll("_", " ");
}

function toneForStatus(value?: string | null): string {
  const normalized = String(value ?? "").toLowerCase();
  if (["failed", "aborted", "cancelled", "disconnected"].includes(normalized)) return "danger";
  if (["completed", "success", "archived"].includes(normalized)) return "success";
  if (["running", "live", "needs_human", "waiting_external"].includes(normalized)) return "active";
  return "neutral";
}

function qualityBadge(job: { final_acceptance?: string | null; quality_grade?: string | null }): { label: string; tone: string } | null {
  const quality = String(job.quality_grade ?? "").trim();
  const acceptance = String(job.final_acceptance ?? "").trim();
  if (!quality && !acceptance) return null;
  if (quality) {
    if (quality === "high_confidence") return { label: `quality:${quality}`, tone: "success" };
    if (quality === "warning_usable" || quality === "low_confidence") return { label: `quality:${quality}`, tone: "warning" };
    return { label: `quality:${quality}`, tone: "neutral" };
  }
  if (acceptance === "pass" || acceptance === "accepted") return { label: `accept:${acceptance}`, tone: "success" };
  if (["pass_with_warning", "accepted_with_warning", "fail", "rejected"].includes(acceptance)) {
    return { label: `accept:${acceptance}`, tone: "warning" };
  }
  return { label: `accept:${acceptance}`, tone: "neutral" };
}

function inferStageStatus(job: JobDetail, stage: string): string {
  const workflow = (job.state?.workflow as Record<string, unknown> | undefined) ?? {};
  const stageStatus = (workflow.stage_status as Record<string, string> | undefined) ?? {};
  const explicit = stageStatus[stage];
  if (explicit) return explicit;
  if (job.current_stage === stage && ["running", "needs_human", "waiting_external"].includes(job.runtime_run_status)) return "active";
  if (job.current_stage === "final_report" && stage !== "final_report" && job.runtime_run_status === "completed") return "success";
  return "pending";
}

function artifactLabel(name: string): string {
  return name.replace(/_path$/u, "").replaceAll("_", " ");
}

function compactPath(path: string): string {
  const parts = path.split("/");
  return parts.slice(Math.max(0, parts.length - 3)).join("/");
}

function PresentationActions({
  targetRef,
  presentation,
  onToggle,
}: {
  targetRef: RefObject<HTMLDivElement>;
  presentation: boolean;
  onToggle: () => void;
}) {
  const exportPng = async () => {
    if (!targetRef.current) return;
    const dataUrl = await toPng(targetRef.current, { cacheBust: true, pixelRatio: 2 });
    const link = document.createElement("a");
    link.href = dataUrl;
    link.download = "script-new-console.png";
    link.click();
  };

  return (
    <div className="presentation-actions">
      <button onClick={onToggle}>{presentation ? "Exit Presentation" : "Presentation Mode"}</button>
      <button onClick={exportPng}>Export PNG</button>
      <button onClick={() => window.print()}>Export PDF</button>
    </div>
  );
}

function Card({
  title,
  subtitle,
  children,
  tone,
  className,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  tone?: "default" | "accent" | "warning";
  className?: string;
}) {
  return (
    <section className={`card ${tone ?? "default"} ${className ?? ""}`.trim()}>
      <header className="card-header">
        <div>
          <h3>{title}</h3>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
      </header>
      <div className="card-body">{children}</div>
    </section>
  );
}

function JsonView({ value }: { value: unknown }) {
  return <pre className="json-view">{JSON.stringify(value ?? {}, null, 2)}</pre>;
}

function ShellSidebar({
  jobs,
  route,
  health,
}: {
  jobs: JobSummary[];
  route: Route;
  health: Record<string, unknown>;
}) {
  const recent = jobs.slice(0, 10);
  return (
    <aside className="shell-sidebar">
      <div className="sidebar-brand">
        <button className="brand" onClick={() => navigate("/app/")}>script_new</button>
        <p>Agent workspace for mobility runs, recovery, and evidence-backed decisions.</p>
      </div>
      <div className="sidebar-nav">
        <button className={route.kind === "home" ? "active" : ""} onClick={() => navigate("/app/")}>Home</button>
        <button className={route.kind === "wiki" ? "active" : ""} onClick={() => navigate("/app/wiki")}>Wiki</button>
        <button className={route.kind === "jobs" ? "active" : ""} onClick={() => navigate("/app/jobs")}>Overview</button>
      </div>
      <div className="sidebar-health">
        <div>
          <span>Backend</span>
          <strong>{String(health.backend ?? "unknown")}</strong>
        </div>
        <div>
          <span>Active jobs</span>
          <strong>{String(health.active_job_count ?? 0)}</strong>
        </div>
        <div>
          <span>WS clients</span>
          <strong>{String(health.websocket_client_count ?? 0)}</strong>
        </div>
      </div>
      <div className="sidebar-section">
        <h4>Recent Runs</h4>
        <div className="sidebar-job-list">
          {recent.map((job) => {
            const quality = qualityBadge(job);
            return (
            <button
              className={`sidebar-job ${route.kind === "job" && route.jobId === job.job_id ? "selected" : ""}`.trim()}
              key={job.job_id}
              onClick={() => navigate(`/app/jobs/${job.job_id}`)}
            >
              <div className="sidebar-job-top">
                <span>{job.display_name}</span>
                <div className="pill-row">
                  <span className={`mini-pill ${toneForStatus(job.runtime_run_status)}`}>{job.runtime_run_status}</span>
                  {quality ? <span className={`mini-pill ${quality.tone}`}>{quality.label}</span> : null}
                </div>
              </div>
              <small>{job.current_stage ?? "n/a"} · {formatTime(job.last_heartbeat_at)}</small>
            </button>
            );
          })}
        </div>
      </div>
    </aside>
  );
}

function FlowRail({ job }: { job: JobDetail }) {
  return (
    <div className="flow-rail">
      {STAGE_ORDER.map((stage, index) => {
        const status = inferStageStatus(job, stage);
        return (
          <div className={`flow-stage ${status}`} key={stage}>
            <div className="flow-node">{index + 1}</div>
            <div className="flow-copy">
              <strong>{displayStageName(stage)}</strong>
              <span>{status}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ArtifactPreviewPanel({ job }: { job: JobDetail }) {
  const [previews, setPreviews] = useState<ArtifactPreviewMap>({});
  const [previewError, setPreviewError] = useState<string | null>(null);

  useEffect(() => {
    const keys = AGENTIC_ARTIFACT_KEYS.filter((name) => Boolean(job.artifacts?.[name]));
    if (keys.length === 0) {
      setPreviews({});
      return;
    }
    let cancelled = false;
    Promise.all(
      keys.map(async (name) => {
        const payload = await apiFetch<unknown>(`/api/jobs/${job.job_id}/artifact-json/${encodeURIComponent(name)}`);
        return [name, payload] as const;
      }),
    )
      .then((items) => {
        if (cancelled) return;
        setPreviews(Object.fromEntries(items));
        setPreviewError(null);
      })
      .catch((error: Error) => {
        if (cancelled) return;
        setPreviewError(error.message);
      });
    return () => {
      cancelled = true;
    };
  }, [job.artifacts, job.job_id]);

  const parameterPlan = previews.parameter_plan_path as Record<string, unknown> | undefined;
  const recoveryDiagnosis = previews.recovery_diagnosis_path as Record<string, unknown> | undefined;
  const retrievalTrace = previews.retrieval_trace_path as Array<Record<string, unknown>> | undefined;
  const finalSummary = previews.final_summary_path as Record<string, unknown> | undefined;

  return (
    <Card title="Agent Workspace" subtitle="What the runtime retrieved, planned, and concluded">
      {previewError ? <div className="inline-banner">{previewError}</div> : null}
      <div className="insight-grid">
        <div className="insight-card">
          <h4>Parameter Plan</h4>
          {parameterPlan && Object.keys(parameterPlan).length ? (
            <div className="mini-list">
              {Object.entries(parameterPlan).map(([stage, plan]) => {
                const record = plan as Record<string, unknown>;
                const overrides = (record.incar_overrides as Record<string, unknown> | undefined) ?? {};
                return (
                  <div className="mini-item" key={stage}>
                    <strong>{displayStageName(stage)}</strong>
                    <span>{Object.keys(overrides).length} INCAR overrides</span>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="empty-note">No parameter plan preview yet. Dry-run jobs usually keep this empty.</p>
          )}
        </div>
        <div className="insight-card">
          <h4>Recovery Diagnosis</h4>
          {recoveryDiagnosis && Object.keys(recoveryDiagnosis).length ? (
            <div className="mini-list">
              <div className="mini-item">
                <strong>{String(recoveryDiagnosis.recommended_action ?? "n/a")}</strong>
                <span>recommended action</span>
              </div>
              {Array.isArray(recoveryDiagnosis.hypotheses)
                ? recoveryDiagnosis.hypotheses.slice(0, 3).map((item, index) => (
                    <div className="mini-item" key={`${item}-${index}`}>
                      <strong>Hypothesis {index + 1}</strong>
                      <span>{String(item)}</span>
                    </div>
                  ))
                : null}
            </div>
          ) : (
            <p className="empty-note">No recovery diagnosis recorded for this run.</p>
          )}
        </div>
        <div className="insight-card span-2">
          <h4>Retrieved Evidence</h4>
          {retrievalTrace && retrievalTrace.length ? (
            <div className="evidence-stack">
              {retrievalTrace.slice(-6).reverse().map((item, index) => {
                const record = item as Record<string, unknown>;
                return (
                  <div className="evidence-card" key={`${record.stage ?? "stage"}-${index}`}>
                    <div className="evidence-meta">
                      <span>{String(record.stage ?? "stage")}</span>
                      <span>{String(record.kind ?? "trace")}</span>
                    </div>
                    <pre>{JSON.stringify(record, null, 2)}</pre>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="empty-note">No retrieval trace exported yet.</p>
          )}
        </div>
        {finalSummary && Object.keys(finalSummary).length ? (
          <div className="insight-card span-2">
            <h4>Final Summary</h4>
            <pre className="mini-json">{JSON.stringify(finalSummary, null, 2)}</pre>
          </div>
        ) : null}
      </div>
    </Card>
  );
}

function RuntimeSettingsCard({
  settings,
  onSaved,
}: {
  settings: RuntimeSettings | null;
  onSaved: (settings: RuntimeSettings) => void;
}) {
  const [form, setForm] = useState({
    service_preset: "custom",
    mobility_db_uri: "",
    llm_provider: "openai",
    llm_base_url: "",
    llm_model: "",
    llm_api_key: "",
    clear_llm_api_key: false,
    embedding_model: "",
    embedding_base_url: "",
    embedding_api_key: "",
    clear_embedding_api_key: false,
    wiki_qa_model: "",
    agentic_policy_enabled: true,
    policy_allowlist_mode: "restricted",
    policy_retrieval_top_k: 5,
    policy_trace_enabled: true,
    rag_top_k: 6,
    rag_chunk_size: 1200,
    rag_chunk_overlap: 180,
    rag_reindex_batch_size: 64,
    hitl_policy: "interactive",
    human_review_timeout_seconds: 300,
    human_review_default_action: "skip_material",
    enable_email_notifications: false,
    email_notify_to: "",
    email_dry_run: true,
    smtp_host: "",
    smtp_port: 587,
    smtp_use_tls: true,
    smtp_username: "",
    smtp_from: "",
    smtp_password: "",
    clear_smtp_password: false,
  });
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!settings) return;
    setForm({
      service_preset: settings.service_preset || "custom",
      mobility_db_uri: settings.mobility_db_uri || "",
      llm_provider: normalizeLlmProvider(settings.llm_provider),
      llm_base_url: settings.llm_base_url || "",
      llm_model: settings.llm_model || "",
      llm_api_key: "",
      clear_llm_api_key: false,
      embedding_model: settings.embedding_model || "",
      embedding_base_url: settings.embedding_base_url || "",
      embedding_api_key: "",
      clear_embedding_api_key: false,
      wiki_qa_model: settings.wiki_qa_model || settings.llm_model || "",
      agentic_policy_enabled: settings.agentic_policy_enabled,
      policy_allowlist_mode: settings.policy_allowlist_mode || "restricted",
      policy_retrieval_top_k: settings.policy_retrieval_top_k || 5,
      policy_trace_enabled: settings.policy_trace_enabled,
      rag_top_k: settings.rag_top_k || 6,
      rag_chunk_size: settings.rag_chunk_size || 1200,
      rag_chunk_overlap: settings.rag_chunk_overlap || 180,
      rag_reindex_batch_size: settings.rag_reindex_batch_size || 64,
      hitl_policy: settings.hitl_policy || "interactive",
      human_review_timeout_seconds: settings.human_review_timeout_seconds || 300,
      human_review_default_action: settings.human_review_default_action || "skip_material",
      enable_email_notifications: settings.enable_email_notifications,
      email_notify_to: settings.email_notify_to || "",
      email_dry_run: settings.email_dry_run,
      smtp_host: settings.smtp_host || "",
      smtp_port: settings.smtp_port || 587,
      smtp_use_tls: settings.smtp_use_tls,
      smtp_username: settings.smtp_username || "",
      smtp_from: settings.smtp_from || "",
      smtp_password: "",
      clear_smtp_password: false,
    });
  }, [settings]);

  const applyPreset = (preset: string) => {
    setForm((current) => {
      let llmBaseUrl = current.llm_base_url;
      let llmProvider = normalizeLlmProvider(current.llm_provider);
      let llmModel = current.llm_model;
      if (preset === "openrouter") {
        llmBaseUrl = "https://openrouter.ai/api/v1";
        llmProvider = "openai";
      } else if (preset === "zhipu") {
        llmBaseUrl = "https://open.bigmodel.cn/api/paas/v4/";
        llmProvider = "openai";
        if (!llmModel || current.service_preset === "openrouter" || current.service_preset === "custom") {
          llmModel = "glm-5.1";
        }
      } else if (preset === "gemini") {
        llmBaseUrl = "https://generativelanguage.googleapis.com/v1beta/openai/";
        llmProvider = "openai";
        if (
          !llmModel
          || current.service_preset === "openrouter"
          || current.service_preset === "custom"
          || current.service_preset === "zhipu"
        ) {
          llmModel = "gemini-2.5-flash";
        }
      }
      return {
        ...current,
        service_preset: preset,
        llm_provider: llmProvider,
        llm_base_url: llmBaseUrl,
        llm_model: llmModel,
      };
    });
  };

  const save = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const payload: Record<string, unknown> = {
        mobility_db_uri: form.mobility_db_uri,
        llm_provider: form.llm_provider,
        llm_base_url: form.llm_base_url,
        llm_model: form.llm_model,
        embedding_model: form.embedding_model,
        embedding_base_url: form.embedding_base_url,
        wiki_qa_model: form.wiki_qa_model,
        agentic_policy_enabled: form.agentic_policy_enabled,
        policy_allowlist_mode: form.policy_allowlist_mode,
        policy_retrieval_top_k: form.policy_retrieval_top_k,
        policy_trace_enabled: form.policy_trace_enabled,
        rag_top_k: form.rag_top_k,
        rag_chunk_size: form.rag_chunk_size,
        rag_chunk_overlap: form.rag_chunk_overlap,
        rag_reindex_batch_size: form.rag_reindex_batch_size,
        hitl_policy: form.hitl_policy,
        human_review_timeout_seconds: form.human_review_timeout_seconds,
        human_review_default_action: form.human_review_default_action,
        enable_email_notifications: form.enable_email_notifications,
        email_notify_to: form.email_notify_to,
        email_dry_run: form.email_dry_run,
        smtp_host: form.smtp_host,
        smtp_port: form.smtp_port,
        smtp_use_tls: form.smtp_use_tls,
        smtp_username: form.smtp_username,
        smtp_from: form.smtp_from,
      };
      if (form.llm_api_key.trim()) {
        payload.llm_api_key = form.llm_api_key.trim();
      }
      if (form.clear_llm_api_key) {
        payload.clear_llm_api_key = true;
      }
      if (form.embedding_api_key.trim()) {
        payload.embedding_api_key = form.embedding_api_key.trim();
      }
      if (form.clear_embedding_api_key) {
        payload.clear_embedding_api_key = true;
      }
      if (form.smtp_password.trim()) {
        payload.smtp_password = form.smtp_password.trim();
      }
      if (form.clear_smtp_password) {
        payload.clear_smtp_password = true;
      }
      const saved = await apiFetch<RuntimeSettings>("/api/settings/runtime", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      onSaved(saved);
      setMessage("Runtime settings saved. New jobs will use these values.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "failed_to_save_runtime_settings");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card
      title="Runtime Settings"
      subtitle="Configure Postgres, LLM, embeddings, RAG indexing, and SMTP before launching new jobs"
      className="settings-card"
    >
      <div className="preset-row">
        {[
          { key: "openrouter", label: "OpenRouter" },
          { key: "zhipu", label: "Zhipu GLM" },
          { key: "gemini", label: "Gemini AI Studio" },
          { key: "custom", label: "Custom" },
        ].map((item) => (
          <button
            key={item.key}
            className={form.service_preset === item.key ? "active" : ""}
            onClick={() => applyPreset(item.key)}
            type="button"
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="settings-section">
        <h4>Database And LLM API</h4>
        <div className="settings-grid">
          <label className="span-2">
            Mobility DB URI
            <input value={form.mobility_db_uri} onChange={(event) => setForm({ ...form, mobility_db_uri: event.target.value })} />
          </label>
          <label>
            Provider
            <input value={form.llm_provider} onChange={(event) => setForm({ ...form, llm_provider: event.target.value })} />
          </label>
          <label>
            Base URL
            <input value={form.llm_base_url} onChange={(event) => setForm({ ...form, llm_base_url: event.target.value })} />
          </label>
          <label>
            Main model
            <input value={form.llm_model} onChange={(event) => setForm({ ...form, llm_model: event.target.value })} />
          </label>
          <label className="span-2">
            API Key
            <input
              type="password"
              placeholder={settings?.llm_api_key_present ? "Leave blank to keep the saved key" : "Paste a new API key"}
              value={form.llm_api_key}
              onChange={(event) => setForm({ ...form, llm_api_key: event.target.value, clear_llm_api_key: false })}
            />
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={form.clear_llm_api_key}
              onChange={(event) => setForm({ ...form, clear_llm_api_key: event.target.checked, llm_api_key: "" })}
            />
            Clear saved API key
          </label>
        </div>
        <p className="field-hint">
          DB URI: {settings?.mobility_db_uri || "not set"}.
        </p>
        <p className="field-hint">
          Saved key: {settings?.llm_api_key_present ? settings.llm_api_key_preview ?? "present" : "not set"}.
          The key is stored server-side in the repo `.env.local`.
        </p>
      </div>

      <div className="settings-section">
        <h4>Agentic Policy And RAG</h4>
        <div className="settings-grid">
          <label>
            Embedding model
            <input value={form.embedding_model} onChange={(event) => setForm({ ...form, embedding_model: event.target.value })} />
          </label>
          <label>
            Embedding base URL
            <input value={form.embedding_base_url} onChange={(event) => setForm({ ...form, embedding_base_url: event.target.value })} />
          </label>
          <label>
            Wiki QA model
            <input value={form.wiki_qa_model} onChange={(event) => setForm({ ...form, wiki_qa_model: event.target.value })} />
          </label>
          <label className="span-2">
            Embedding API Key
            <input
              type="password"
              placeholder={settings?.embedding_api_key_present ? "Leave blank to keep the saved key" : "Paste a new embedding API key"}
              value={form.embedding_api_key}
              onChange={(event) => setForm({ ...form, embedding_api_key: event.target.value, clear_embedding_api_key: false })}
            />
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={form.clear_embedding_api_key}
              onChange={(event) => setForm({ ...form, clear_embedding_api_key: event.target.checked, embedding_api_key: "" })}
            />
            Clear saved embedding API key
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={form.agentic_policy_enabled}
              onChange={(event) => setForm({ ...form, agentic_policy_enabled: event.target.checked })}
            />
            Enable agentic parameter planning
          </label>
          <label>
            Allowlist mode
            <select value={form.policy_allowlist_mode} onChange={(event) => setForm({ ...form, policy_allowlist_mode: event.target.value })}>
              <option value="restricted">restricted</option>
            </select>
          </label>
          <label>
            Retrieval top-k
            <input
              type="number"
              min={1}
              max={20}
              value={form.policy_retrieval_top_k}
              onChange={(event) => setForm({ ...form, policy_retrieval_top_k: Number(event.target.value || 1) })}
            />
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={form.policy_trace_enabled}
              onChange={(event) => setForm({ ...form, policy_trace_enabled: event.target.checked })}
            />
            Export policy traces
          </label>
          <label>
            RAG top-k
            <input
              type="number"
              min={1}
              max={20}
              value={form.rag_top_k}
              onChange={(event) => setForm({ ...form, rag_top_k: Number(event.target.value || 1) })}
            />
          </label>
          <label>
            Chunk size
            <input
              type="number"
              min={200}
              value={form.rag_chunk_size}
              onChange={(event) => setForm({ ...form, rag_chunk_size: Number(event.target.value || 200) })}
            />
          </label>
          <label>
            Chunk overlap
            <input
              type="number"
              min={0}
              value={form.rag_chunk_overlap}
              onChange={(event) => setForm({ ...form, rag_chunk_overlap: Number(event.target.value || 0) })}
            />
          </label>
          <label>
            Reindex batch size
            <input
              type="number"
              min={1}
              value={form.rag_reindex_batch_size}
              onChange={(event) => setForm({ ...form, rag_reindex_batch_size: Number(event.target.value || 1) })}
            />
          </label>
        </div>
        <p className="field-hint">
          Embedding key: {settings?.embedding_api_key_present ? settings.embedding_api_key_preview ?? "present" : "not set"}.
        </p>
      </div>

      <div className="settings-section">
        <h4>HITL And Email</h4>
        <div className="settings-grid">
          <label>
            HITL policy
            <select value={form.hitl_policy} onChange={(event) => setForm({ ...form, hitl_policy: event.target.value })}>
              <option value="interactive">interactive</option>
              <option value="non_interactive_skip_on_timeout">non_interactive_skip_on_timeout</option>
              <option value="non_interactive_abort_on_timeout">non_interactive_abort_on_timeout</option>
            </select>
          </label>
          <label>
            Human timeout seconds
            <input
              type="number"
              min={0}
              value={form.human_review_timeout_seconds}
              onChange={(event) => setForm({ ...form, human_review_timeout_seconds: Number(event.target.value || 0) })}
            />
          </label>
          <label>
            Timeout default action
            <select
              value={form.human_review_default_action}
              onChange={(event) => setForm({ ...form, human_review_default_action: event.target.value })}
            >
              <option value="skip_material">skip_material</option>
              <option value="abort_task">abort_task</option>
              <option value="retry_current_stage">retry_current_stage</option>
            </select>
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={form.enable_email_notifications}
              onChange={(event) => setForm({ ...form, enable_email_notifications: event.target.checked })}
            />
            Enable email notifications
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={form.email_dry_run}
              onChange={(event) => setForm({ ...form, email_dry_run: event.target.checked })}
            />
            Email dry run
          </label>
          <label className="span-2">
            Notify to
            <input value={form.email_notify_to} onChange={(event) => setForm({ ...form, email_notify_to: event.target.value })} />
          </label>
          <label>
            SMTP host
            <input value={form.smtp_host} onChange={(event) => setForm({ ...form, smtp_host: event.target.value })} />
          </label>
          <label>
            SMTP port
            <input type="number" min={1} value={form.smtp_port} onChange={(event) => setForm({ ...form, smtp_port: Number(event.target.value || 0) })} />
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={form.smtp_use_tls}
              onChange={(event) => setForm({ ...form, smtp_use_tls: event.target.checked })}
            />
            SMTP TLS
          </label>
          <label>
            SMTP username
            <input value={form.smtp_username} onChange={(event) => setForm({ ...form, smtp_username: event.target.value })} />
          </label>
          <label>
            SMTP from
            <input value={form.smtp_from} onChange={(event) => setForm({ ...form, smtp_from: event.target.value })} />
          </label>
          <label className="span-2">
            SMTP password
            <input
              type="password"
              placeholder={settings?.smtp_password_present ? "Leave blank to keep the saved SMTP password" : "Paste a new SMTP password"}
              value={form.smtp_password}
              onChange={(event) => setForm({ ...form, smtp_password: event.target.value, clear_smtp_password: false })}
            />
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={form.clear_smtp_password}
              onChange={(event) => setForm({ ...form, clear_smtp_password: event.target.checked, smtp_password: "" })}
            />
            Clear saved SMTP password
          </label>
        </div>
        <p className="field-hint">
          Saved SMTP password: {settings?.smtp_password_present ? settings.smtp_password_preview ?? "present" : "not set"}.
        </p>
      </div>

      <div className="settings-actions">
        <button disabled={busy} onClick={save} type="button">
          {busy ? "Saving..." : "Save Runtime Settings"}
        </button>
        {message ? <span className="status-note">{message}</span> : null}
      </div>
    </Card>
  );
}

function WikiPage({
  settings,
  health,
  onReindexStarted,
}: {
  settings: RuntimeSettings | null;
  health: Record<string, unknown>;
  onReindexStarted: (jobId: string) => void;
}) {
  const [query, setQuery] = useState("What does ISMEAR control in VASP, and when should I change it?");
  const [topK, setTopK] = useState(6);
  const [busy, setBusy] = useState(false);
  const [reindexBusy, setReindexBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [answer, setAnswer] = useState<{ answer: string; citations: Array<Record<string, unknown>>; retrieval_metadata?: Record<string, unknown> } | null>(null);

  const runQuery = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const payload = await apiFetch<{ answer: string; citations: Array<Record<string, unknown>>; retrieval_metadata?: Record<string, unknown> }>("/api/wiki/query", {
        method: "POST",
        body: JSON.stringify({
          query,
          top_k: topK,
          corpora: ["vasp_wiki"],
        }),
      });
      setAnswer(payload);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "wiki_query_failed");
    } finally {
      setBusy(false);
    }
  };

  const startReindex = async () => {
    setReindexBusy(true);
    setMessage(null);
    try {
      const payload = await apiFetch<JobDetail>("/api/wiki/reindex", {
        method: "POST",
        body: JSON.stringify({ mode: "incremental", include_all_pages: false }),
      });
      onReindexStarted(payload.job_id);
      setMessage("Wiki reindex job started.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "wiki_reindex_failed");
    } finally {
      setReindexBusy(false);
    }
  };

  return (
    <div className="stack">
      <Card title="Wiki RAG" subtitle="Query the Postgres-backed VASP Wiki index and trigger control-plane reindex jobs." tone="accent">
        <div className="settings-grid">
          <label className="span-2">
            Question
            <textarea value={query} onChange={(event) => setQuery(event.target.value)} rows={4} />
          </label>
          <label>
            Top-k
            <input type="number" min={1} max={20} value={topK} onChange={(event) => setTopK(Number(event.target.value || 1))} />
          </label>
          <label>
            Collection health
            <input readOnly value={String((health.wiki as Record<string, unknown> | undefined)?.status ?? "unknown")} />
          </label>
        </div>
        <div className="button-row">
          <button onClick={runQuery} disabled={busy}>{busy ? "Querying..." : "Run Query"}</button>
          <button onClick={startReindex} disabled={reindexBusy}>{reindexBusy ? "Starting..." : "Reindex Wiki"}</button>
        </div>
        <p className="field-hint">
          DB: {settings?.mobility_db_uri || "not set"} | Embeddings: {settings?.embedding_model || "not set"} | QA model: {settings?.wiki_qa_model || settings?.llm_model || "not set"}
        </p>
        {message ? <p className="field-hint">{message}</p> : null}
      </Card>

      {answer ? (
        <Card title="Answer" subtitle={String(answer.retrieval_metadata?.collection_name ?? "VASP Wiki")}>
          <div className="card-body stack">
            <div className="markdown-prose">
              <pre className="log-view">{answer.answer}</pre>
            </div>
            <div className="artifact-preview-grid">
              {answer.citations.map((citation, index) => (
                <div className="artifact-preview-card" key={`${String(citation.chunk_id ?? index)}`}>
                  <h4>{String(citation.title ?? "source")}</h4>
                  <p>{String(citation.heading ?? "")}</p>
                  <p>{String(citation.snippet ?? "")}</p>
                  <small>{String(citation.url ?? "")}</small>
                </div>
              ))}
            </div>
          </div>
        </Card>
      ) : null}
    </div>
  );
}

function OverviewPage({
  jobs,
  refresh,
  runtimeSettings,
  onRuntimeSettingsSaved,
}: {
  jobs: JobSummary[];
  refresh: () => Promise<void>;
  runtimeSettings: RuntimeSettings | null;
  onRuntimeSettingsSaved: (settings: RuntimeSettings) => void;
}) {
  const [singleForm, setSingleForm] = useState({
    display_name: "",
    root_path: "",
    material_id: "",
    dry_run: true,
    dry_run_fail_stages: "",
    fresh: true,
  });
  const [batchForm, setBatchForm] = useState({
    display_name: "",
    batch_tag: "demo_batch",
    runs_root: "",
    mongo_uri: "",
    mongo_db: "materials_database",
    mongo_collection: "Vertical_NM_Sample_20",
    dry_run: true,
    fresh_materials: true,
  });
  const [busy, setBusy] = useState<string | null>(null);
  const counts = useMemo(() => summarizeCounts(jobs), [jobs]);

  const submitSingle = async () => {
    setBusy("single");
    try {
      const detail = await apiFetch<JobDetail>("/api/jobs/single", {
        method: "POST",
        body: JSON.stringify({
          display_name: singleForm.display_name || undefined,
          root_path: singleForm.root_path,
          material_id: singleForm.material_id || undefined,
          fresh: singleForm.fresh,
          runtime: {
            dry_run: singleForm.dry_run,
            dry_run_fail_stages: singleForm.dry_run_fail_stages,
          },
        }),
      });
      navigate(`/app/jobs/${detail.job_id}`);
    } finally {
      setBusy(null);
      await refresh();
    }
  };

  const submitBatch = async () => {
    setBusy("batch");
    try {
      const detail = await apiFetch<JobDetail>("/api/jobs/batch", {
        method: "POST",
        body: JSON.stringify({
          display_name: batchForm.display_name || undefined,
          fresh_materials: batchForm.fresh_materials,
          config: {
            batch_tag: batchForm.batch_tag,
            runs_root: batchForm.runs_root,
            mongo_uri: batchForm.mongo_uri,
            mongo_db: batchForm.mongo_db,
            mongo_collection: batchForm.mongo_collection,
            potcar_method: "vaspkit",
            vaspkit_cmd: "vaspkit",
            vaspkit_task: 103,
            retry_failed: false,
            running_stale_s: 43200,
          },
          runtime: {
            dry_run: batchForm.dry_run,
          },
        }),
      });
      navigate(`/app/jobs/${detail.job_id}`);
    } finally {
      setBusy(null);
      await refresh();
    }
  };

  return (
    <div className="page-grid">
      <RuntimeSettingsCard settings={runtimeSettings} onSaved={onRuntimeSettingsSaved} />

      <Card title="Launch Single Material" subtitle="Canonical runner, no duplicated workflow" tone="accent" className="launch-card">
        <label>
          Display name
          <input value={singleForm.display_name} onChange={(event) => setSingleForm({ ...singleForm, display_name: event.target.value })} />
        </label>
        <label>
          Root path
          <input value={singleForm.root_path} onChange={(event) => setSingleForm({ ...singleForm, root_path: event.target.value })} />
        </label>
        <label>
          Material ID
          <input value={singleForm.material_id} onChange={(event) => setSingleForm({ ...singleForm, material_id: event.target.value })} />
        </label>
        <label>
          Dry-run fail stages
          <input
            placeholder="scf,band"
            value={singleForm.dry_run_fail_stages}
            onChange={(event) => setSingleForm({ ...singleForm, dry_run_fail_stages: event.target.value })}
          />
        </label>
        <div className="toggle-row">
          <label><input type="checkbox" checked={singleForm.dry_run} onChange={(event) => setSingleForm({ ...singleForm, dry_run: event.target.checked })} />Dry run</label>
          <label><input type="checkbox" checked={singleForm.fresh} onChange={(event) => setSingleForm({ ...singleForm, fresh: event.target.checked })} />Fresh</label>
        </div>
        <button disabled={busy === "single" || !singleForm.root_path} onClick={submitSingle}>Start Single Run</button>
      </Card>

      <Card title="Launch Batch" subtitle="Parent-child orchestration on top of the same runtime" className="launch-card">
        <label>
          Display name
          <input value={batchForm.display_name} onChange={(event) => setBatchForm({ ...batchForm, display_name: event.target.value })} />
        </label>
        <label>
          Batch tag
          <input value={batchForm.batch_tag} onChange={(event) => setBatchForm({ ...batchForm, batch_tag: event.target.value })} />
        </label>
        <label>
          Runs root
          <input value={batchForm.runs_root} onChange={(event) => setBatchForm({ ...batchForm, runs_root: event.target.value })} />
        </label>
        <label>
          Mongo URI
          <input value={batchForm.mongo_uri} onChange={(event) => setBatchForm({ ...batchForm, mongo_uri: event.target.value })} />
        </label>
        <label>
          Mongo DB
          <input value={batchForm.mongo_db} onChange={(event) => setBatchForm({ ...batchForm, mongo_db: event.target.value })} />
        </label>
        <label>
          Mongo Collection
          <input value={batchForm.mongo_collection} onChange={(event) => setBatchForm({ ...batchForm, mongo_collection: event.target.value })} />
        </label>
        <div className="toggle-row">
          <label><input type="checkbox" checked={batchForm.dry_run} onChange={(event) => setBatchForm({ ...batchForm, dry_run: event.target.checked })} />Dry run</label>
          <label><input type="checkbox" checked={batchForm.fresh_materials} onChange={(event) => setBatchForm({ ...batchForm, fresh_materials: event.target.checked })} />Fresh materials</label>
        </div>
        <button disabled={busy === "batch" || !batchForm.runs_root || !batchForm.mongo_uri} onClick={submitBatch}>Start Batch Run</button>
      </Card>

      <Card title="Runtime Snapshot" subtitle="What the control plane believes right now" className="snapshot-card">
        <div className="stat-grid">
          {Object.entries(counts).map(([key, value]) => (
            <div className="stat-chip" key={key}>
              <strong>{value}</strong>
              <span>{key}</span>
            </div>
          ))}
        </div>
      </Card>

      <div className="job-list">
        {jobs.map((job) => {
          const quality = qualityBadge(job);
          return (
          <button className="job-card" key={job.job_id} onClick={() => navigate(`/app/jobs/${job.job_id}`)}>
            <div className="job-card-top">
              <span className={`pill ${job.runtime_run_status}`}>{job.runtime_run_status}</span>
              <span className={`pill ghost ${job.control_plane_status}`}>{job.control_plane_status}</span>
              {quality ? <span className={`pill ${quality.tone}`}>{quality.label}</span> : null}
            </div>
            <h3>{job.display_name}</h3>
            <p>{job.job_role === "batch_parent" ? `Batch ${job.batch_tag ?? ""}` : job.material_id ?? job.job_type}</p>
            <dl>
              <div><dt>Stage</dt><dd>{job.current_stage ?? "n/a"}</dd></div>
              <div><dt>Heartbeat</dt><dd>{formatTime(job.last_heartbeat_at)}</dd></div>
            </dl>
            {job.error_summary ? <p className="error-text">{job.error_summary}</p> : null}
          </button>
          );
        })}
      </div>
    </div>
  );
}

function SingleJobActions({ job, refresh }: { job: JobDetail; refresh: () => Promise<void> }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [hitl, setHitl] = useState({ action: "retry_current_stage", reason: "" });
  const [eventForm, setEventForm] = useState({
    event_type: "resume_requested",
    job_id: "",
    target_capability: "",
    action_family: "run_capability",
    error_summary: "",
  });

  const cancel = async () => {
    setBusy("cancel");
    try {
      await apiFetch(`/api/jobs/${job.job_id}/cancel`, { method: "POST" });
    } finally {
      setBusy(null);
      await refresh();
    }
  };

  const submitHitl = async () => {
    setBusy("hitl");
    try {
      await apiFetch(`/api/jobs/${job.job_id}/hitl/respond`, {
        method: "POST",
        body: JSON.stringify(hitl),
      });
    } finally {
      setBusy(null);
      await refresh();
    }
  };

  const submitEvent = async () => {
    setBusy("event");
    try {
      await apiFetch(`/api/jobs/${job.job_id}/events/resume`, {
        method: "POST",
        body: JSON.stringify({
          thread_id: job.thread_id,
          event: {
            event_type: eventForm.event_type,
            job_id: eventForm.job_id || undefined,
            target_capability: eventForm.target_capability || undefined,
            action_family: eventForm.action_family || undefined,
            error_summary: eventForm.error_summary || undefined,
          },
        }),
      });
    } finally {
      setBusy(null);
      await refresh();
    }
  };

  return (
    <Card title="Control Actions" subtitle="Cancel, respond to HITL, or inject an external resume event">
      <div className="action-grid">
        <button disabled={busy === "cancel"} onClick={cancel}>Cancel Process Group</button>
      </div>
      <div className="form-slab">
        <h4>HITL Response</h4>
        <label>
          Action
          <select value={hitl.action} onChange={(event) => setHitl({ ...hitl, action: event.target.value })}>
            <option value="retry_current_stage">retry_current_stage</option>
            <option value="skip_material">skip_material</option>
            <option value="abort_task">abort_task</option>
          </select>
        </label>
        <label>
          Reason
          <input value={hitl.reason} onChange={(event) => setHitl({ ...hitl, reason: event.target.value })} />
        </label>
        <button disabled={busy === "hitl"} onClick={submitHitl}>Submit HITL Response</button>
      </div>
      <div className="form-slab">
        <h4>External Event Resume</h4>
        <label>
          Event type
          <select value={eventForm.event_type} onChange={(event) => setEventForm({ ...eventForm, event_type: event.target.value })}>
            <option value="resume_requested">resume_requested</option>
            <option value="job_completed">job_completed</option>
            <option value="job_failed">job_failed</option>
            <option value="job_timeout">job_timeout</option>
            <option value="artifact_missing">artifact_missing</option>
          </select>
        </label>
        <label>
          Job ID
          <input value={eventForm.job_id} onChange={(event) => setEventForm({ ...eventForm, job_id: event.target.value })} />
        </label>
        <label>
          Target capability
          <input value={eventForm.target_capability} onChange={(event) => setEventForm({ ...eventForm, target_capability: event.target.value })} />
        </label>
        <label>
          Action family
          <input value={eventForm.action_family} onChange={(event) => setEventForm({ ...eventForm, action_family: event.target.value })} />
        </label>
        <label>
          Error summary
          <input value={eventForm.error_summary} onChange={(event) => setEventForm({ ...eventForm, error_summary: event.target.value })} />
        </label>
        <button disabled={busy === "event"} onClick={submitEvent}>Inject External Event</button>
      </div>
    </Card>
  );
}

function JobDetailPage({ job, refresh }: { job: JobDetail; refresh: () => Promise<void> }) {
  const isBatch = job.job_role === "batch_parent";
  const state = job.state ?? {};
  const stageStatus = (state.workflow as Record<string, unknown> | undefined)?.stage_status as Record<string, string> | undefined;
  const recentTimeline = (job.timeline ?? []).slice().reverse();
  const groupedArtifacts = Object.entries(job.artifacts ?? {}).sort(([left], [right]) => left.localeCompare(right));
  const quality = qualityBadge(job);

  return (
    <div className="detail-workspace">
      <div className="detail-stack">
        <Card title={job.display_name} subtitle={`${job.job_role} • ${job.job_id}`} tone="accent" className="hero-card">
          <div className="hero-headline">
            <div>
              <div className="pill-row">
                <span className={`pill ${job.runtime_run_status}`}>{job.runtime_run_status}</span>
                <span className={`pill ghost ${job.control_plane_status}`}>{job.control_plane_status}</span>
                {quality ? <span className={`pill ${quality.tone}`}>{quality.label}</span> : null}
                {job.hitl_pending ? <span className="pill needs_human">hitl pending</span> : null}
              </div>
              <h2>{job.current_stage ? displayStageName(job.current_stage) : "Waiting to start"}</h2>
              <p>{job.wait_reason ?? job.error_summary ?? "The canonical runtime is streaming state into this workspace."}</p>
            </div>
            <div className="hero-metrics">
              <div><span>Current stage</span><strong>{job.current_stage ?? "n/a"}</strong></div>
              <div><span>PGID</span><strong>{job.pgid ?? "n/a"}</strong></div>
              <div><span>Started</span><strong>{formatTime(job.started_at)}</strong></div>
              <div><span>Heartbeat</span><strong>{formatTime(job.last_heartbeat_at)}</strong></div>
            </div>
          </div>
          <dl className="meta-grid">
            <div><dt>Root</dt><dd>{job.root_path}</dd></div>
            <div><dt>Workdir</dt><dd>{job.workdir ?? "n/a"}</dd></div>
            <div><dt>Thread</dt><dd>{job.thread_id ?? "n/a"}</dd></div>
            <div><dt>Last update</dt><dd>{formatTime(job.last_state_updated_at)}</dd></div>
          </dl>
        </Card>

        <Card title="Process Flow" subtitle="Canonical stages with live status, so you can see the workflow at a glance">
          <FlowRail job={job} />
        </Card>

        <Card title="Stage Ledger" subtitle="Snapshot of stage contract state and downstream progress">
          <div className="stage-grid">
            {STAGE_ORDER.map((stage) => (
              <div className={`stage-chip ${toneForStatus(inferStageStatus(job, stage))}`} key={stage}>
                <span>{displayStageName(stage)}</span>
                <strong>{inferStageStatus(job, stage)}</strong>
              </div>
            ))}
            {Object.entries(stageStatus ?? {})
              .filter(([stage]) => !STAGE_ORDER.includes(stage as (typeof STAGE_ORDER)[number]))
              .map(([stage, status]) => (
                <div className={`stage-chip ${toneForStatus(status)}`} key={stage}>
                  <span>{displayStageName(stage)}</span>
                  <strong>{status}</strong>
                </div>
              ))}
          </div>
        </Card>

        <Card title="Live Event Stream" subtitle="A readable event feed instead of forcing you to stare at raw logs">
          <div className="event-feed">
            {recentTimeline.map((item, index) => (
              <div className={`event-card ${toneForStatus(item.runtime_run_status ?? item.stage_status)}`} key={`${item.timestamp}-${index}`}>
                <div className="event-topline">
                  <strong>{item.event_type}</strong>
                  <span>{formatTime(item.timestamp)}</span>
                </div>
                <p>
                  {item.current_stage ? displayStageName(item.current_stage) : "n/a"} · {item.runtime_run_status ?? item.stage_status ?? "pending"}
                </p>
                {item.selected_action_family || item.selected_capability ? (
                  <small>
                    {item.selected_action_family ?? "action"} {item.selected_capability ? `→ ${item.selected_capability}` : ""}
                  </small>
                ) : null}
                {item.latest_error ? <small className="error-text">{item.latest_error}</small> : null}
              </div>
            ))}
          </div>
        </Card>

        <ArtifactPreviewPanel job={job} />

        {isBatch ? (
          <Card title="Batch Parent-Child View" subtitle="Children are canonical single-material runs, not a separate workflow">
            <div className="taxonomy-grid">
              {Object.entries(job.failure_taxonomy ?? {}).map(([key, value]) => (
                <div className="stat-chip" key={key}>
                  <strong>{value}</strong>
                  <span>{key}</span>
                </div>
              ))}
            </div>
            <div className="child-list">
              {(job.children ?? []).map((child) => (
                <button className="child-row" key={child.job_id} onClick={() => navigate(`/app/jobs/${child.job_id}`)}>
                  <span>{child.display_name}</span>
                  <span>{child.current_stage ?? "n/a"}</span>
                  <span>{child.runtime_run_status}</span>
                </button>
              ))}
            </div>
          </Card>
        ) : (
          <SingleJobActions job={job} refresh={refresh} />
        )}
      </div>

      <div className="detail-rail">
        <Card title="Artifacts" subtitle="Downloadable runtime outputs and trace files">
          <div className="artifact-grid">
            {groupedArtifacts.map(([name, path]) => (
              <a key={name} className="artifact-link" href={`${apiBase}/api/jobs/${job.job_id}/download/${encodeURIComponent(name)}`}>
                <span>{artifactLabel(name)}</span>
                <small>{compactPath(path)}</small>
              </a>
            ))}
          </div>
        </Card>

        <Card title="Runtime Logs" subtitle="Human-readable progress stream">
          <pre className="log-view">{(job.logs?.lines ?? []).join("\n")}</pre>
        </Card>

        <Card title="Runtime State" subtitle="Raw state is still available when you need the exact payload">
          <JsonView value={job.state} />
        </Card>
      </div>
    </div>
  );
}

export default function App() {
  const [route, setRoute] = useState<Route>(() => routeFromLocation());
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [selectedJob, setSelectedJob] = useState<JobDetail | null>(null);
  const [health, setHealth] = useState<Record<string, unknown>>({});
  const [runtimeSettings, setRuntimeSettings] = useState<RuntimeSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const presentation = new URLSearchParams(window.location.search).get("presentation") === "1";
  const exportRef = useRef<HTMLDivElement>(null);

  const refreshJobs = async () => {
    const [jobsPayload, healthPayload, settingsPayload] = await Promise.all([
      apiFetch<JobSummary[]>("/api/jobs"),
      apiFetch<Record<string, unknown>>("/api/health"),
      apiFetch<RuntimeSettings>("/api/settings/runtime"),
    ]);
    setJobs(jobsPayload);
    setHealth(healthPayload);
    setRuntimeSettings(settingsPayload);
  };

  const refreshDetail = async (jobId: string) => {
    const detail = await apiFetch<JobDetail>(`/api/jobs/${jobId}`);
    setSelectedJob(detail);
  };

  useEffect(() => {
    const onPopState = () => setRoute(routeFromLocation());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    refreshJobs().catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    if (route.kind !== "job") {
      setSelectedJob(null);
      return;
    }
    refreshDetail(route.jobId).catch((err: Error) => setError(err.message));
  }, [route]);

  useEffect(() => {
    if (route.kind === "job") {
      const socket = new WebSocket(wsUrl(`/ws/jobs/${route.jobId}`));
      socket.onmessage = (event) => {
        const payload = JSON.parse(event.data) as { detail?: JobDetail };
        if (payload.detail) {
          setSelectedJob(payload.detail);
        }
      };
      socket.onerror = () => setError("job detail websocket disconnected");
      return () => socket.close();
    }

    const socket = new WebSocket(wsUrl("/ws/jobs"));
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data) as { jobs?: JobSummary[] };
      if (payload.jobs) {
        setJobs(payload.jobs);
      }
    };
    socket.onerror = () => setError("overview websocket disconnected");
    return () => socket.close();
  }, [route]);

  const togglePresentation = () => {
    const url = new URL(window.location.href);
    if (presentation) {
      url.searchParams.delete("presentation");
    } else {
      url.searchParams.set("presentation", "1");
    }
    window.history.replaceState({}, "", url.toString());
    setRoute(routeFromLocation());
  };

  return (
    <div className={`app-shell ${presentation ? "presentation" : ""}`}>
      <div className="shell-frame">
        <ShellSidebar jobs={jobs} route={route} health={health} />

        <div className="shell-main">
          <header className="app-header">
            <div>
              <h1>
                {route.kind === "job" && selectedJob
                  ? selectedJob.display_name
                  : route.kind === "jobs"
                    ? "Overview"
                    : route.kind === "wiki"
                      ? "VASP Wiki"
                      : "Control Plane"}
              </h1>
              <p>
                {route.kind === "job"
                  ? "A workspace view of stages, events, recovery, and agentic evidence."
                  : route.kind === "wiki"
                    ? "Shared pgvector-backed Wiki retrieval and question answering inside the same control plane."
                    : "Recoverable, explainable, monitorable research-agent control plane."}
              </p>
            </div>
            <PresentationActions targetRef={exportRef} presentation={presentation} onToggle={togglePresentation} />
          </header>

          {error ? <div className="banner error">{error}</div> : null}

          <main className="app-main" ref={exportRef}>
            {route.kind === "home" ? (
              <div className="landing-grid">
                <Card title="Scientific Runtime Control Plane" subtitle="A cleaner AI-workspace view built around the canonical LangGraph runtime" tone="accent">
                  <p>
                    Launch runs, watch the canonical stage flow, inspect agentic parameter choices, and intervene when
                    HITL or external resume events are needed.
                  </p>
                  <div className="hero-metrics">
                    <div><span>Backend</span><strong>{String(health.backend ?? "unknown")}</strong></div>
                    <div><span>Registry</span><strong>{String(health.registry_ok ?? "unknown")}</strong></div>
                    <div><span>Active Jobs</span><strong>{String(health.active_job_count ?? 0)}</strong></div>
                    <div><span>WS Clients</span><strong>{String(health.websocket_client_count ?? 0)}</strong></div>
                  </div>
                  <button onClick={() => navigate("/app/jobs")}>Open Overview</button>
                </Card>
                <Card title="Workspace Pattern" subtitle="The UI is now optimized around flow, evidence, and intervention">
                  <ul className="feature-list">
                    <li>A persistent sidebar keeps recent jobs visible, so the console feels more like a real workspace.</li>
                    <li>Stage flow, event feed, artifacts, and runtime state now live in separate reading zones.</li>
                    <li>Agentic traces such as parameter plans and retrieval evidence can be surfaced inline.</li>
                    <li>The deterministic physics backbone stays visible without burying you in raw JSON by default.</li>
                  </ul>
                </Card>
                <Card title="Recent Jobs" subtitle="Jump straight into live or archived runs">
                  <div className="child-list">
                    {jobs.slice(0, 8).map((job) => (
                      <button className="child-row" key={job.job_id} onClick={() => navigate(`/app/jobs/${job.job_id}`)}>
                        <span>{job.display_name}</span>
                        <span>{job.current_stage ?? "n/a"}</span>
                        <span>{job.runtime_run_status}</span>
                      </button>
                    ))}
                  </div>
                </Card>
              </div>
            ) : null}

            {route.kind === "jobs" ? (
              <OverviewPage
                jobs={jobs}
                refresh={refreshJobs}
                runtimeSettings={runtimeSettings}
                onRuntimeSettingsSaved={setRuntimeSettings}
              />
            ) : null}

            {route.kind === "wiki" ? (
              <WikiPage
                settings={runtimeSettings}
                health={health}
                onReindexStarted={(jobId) => navigate(`/app/jobs/${jobId}`)}
              />
            ) : null}

            {route.kind === "job" && selectedJob ? (
              <JobDetailPage job={selectedJob} refresh={() => refreshDetail(selectedJob.job_id)} />
            ) : null}
          </main>
        </div>
      </div>
    </div>
  );
}
