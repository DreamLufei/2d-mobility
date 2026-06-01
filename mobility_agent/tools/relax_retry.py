from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .vasp_common import classify_vasp_failure_text, read_vasp_failure_log_text


# === Public contract ===
# - Only used by relax stage (01_relax under main flow and strain subfolders).
# - Detects ZBRENT fatal banner from merged tail -F of: VASP stdout file(s), slurm-*.out, *.log
# - On detection, retries up to 3 times.


ZBRENT_BANNER_RE = re.compile(
    r"EEEEE[\s\S]*?ZBRENT: fatal error in bracketing[\s\S]*?please rerun with smaller EDIFF, or copy CONTCAR to POSCAR and continue[\s\S]*?I REFUSE TO CONTINUE WITH THIS SICK JOB \.\.\. BYE!!!",
    re.MULTILINE,
)


class RelaxRetryFatal(RuntimeError):
    """Fatal error after exhausting retry budget."""

    def __init__(self, message: str, *, summary: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.summary = dict(summary or {})


@dataclass
class RelaxCheckpoint:
    material_id: str
    retry_n: int
    last_backup_file: Optional[str]

    @staticmethod
    def path(workdir: str) -> str:
        return os.path.join(workdir, ".relax_checkpoint")

    @classmethod
    def load(cls, workdir: str) -> Optional["RelaxCheckpoint"]:
        p = cls.path(workdir)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return cls(
                material_id=str(obj.get("material_id", "")),
                retry_n=int(obj.get("retry_n", 0) or 0),
                last_backup_file=obj.get("last_backup_file"),
            )
        except Exception:
            return None

    def save(self, workdir: str) -> None:
        p = self.path(workdir)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "material_id": self.material_id,
                    "retry_n": int(self.retry_n),
                    "last_backup_file": self.last_backup_file,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        os.replace(tmp, p)


def relax_retry_enabled(*, cli_no_relax_retry: bool = False) -> bool:
    if cli_no_relax_retry:
        return False
    v = os.environ.get("RELAX_RETRY", "true").strip().lower()
    return v not in {"0", "false", "no", "off"}


def detect_zbrent_banner(text: str) -> bool:
    return bool(ZBRENT_BANNER_RE.search(text))


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_retry_log(
    *,
    log_path: str,
    material_id: str,
    retry_n: int,
    error_snippet: str,
    backup_file: Optional[str],
) -> None:
    Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
    backup = backup_file or ""
    line = f"[UTC] {_utc_ts()} | {material_id} | {retry_n} | {error_snippet} | {backup}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def _kill_process_group(proc: subprocess.Popen, *, term_wait_s: float = 5.0) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None

    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            pass

        deadline = time.time() + float(term_wait_s)
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.1)

        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            pass

    try:
        proc.wait(timeout=2)
    except Exception:
        pass


def _backup_and_promote_contcar(workdir: str, retry_n: int) -> str:
    contcar = os.path.join(workdir, "CONTCAR")
    if (not os.path.exists(contcar)) or os.path.getsize(contcar) <= 0:
        raise FileNotFoundError(f"CONTCAR 不存在: {contcar}")

    backup = os.path.join(workdir, f"POSCAR.retry{retry_n}.bak")
    if os.path.exists(backup):
        suffix = 1
        while os.path.exists(f"{backup}.{suffix}"):
            suffix += 1
        backup = f"{backup}.{suffix}"
    import shutil

    poscar = os.path.join(workdir, "POSCAR")
    if os.path.exists(poscar):
        shutil.copy2(poscar, backup)
    else:
        shutil.copy2(contcar, backup)
    shutil.copy2(contcar, poscar)

    return backup


def _maybe_resume_from_checkpoint(workdir: str) -> tuple[int, Optional[str]]:
    ck = RelaxCheckpoint.load(workdir)
    if not ck:
        return 0, None

    if ck.retry_n >= 3:
        return int(ck.retry_n), ck.last_backup_file

    return int(ck.retry_n), ck.last_backup_file


def _tighten_ediff_in_incar(workdir: str, *, factor: float = 0.1, min_ediff: float = 1e-7) -> tuple[Optional[float], Optional[float]]:
    incar = os.path.join(workdir, "INCAR")
    if not os.path.exists(incar):
        return None, None

    try:
        with open(incar, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return None, None

    ediff_re = re.compile(r"^(\s*EDIFF\s*=\s*)([^\s#]+)(.*)$", re.IGNORECASE)
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    new_lines: list[str] = []
    replaced = False

    for line in lines:
        m = ediff_re.match(line.rstrip("\n"))
        if not m:
            new_lines.append(line)
            continue
        try:
            old_value = float(m.group(2))
        except Exception:
            new_lines.append(line)
            continue
        new_value = max(min_ediff, old_value * factor)
        suffix = m.group(3) if m.group(3).endswith("\n") else m.group(3) + "\n"
        new_lines.append(f"{m.group(1)}{new_value:.1e}{suffix}")
        replaced = True

    if not replaced:
        return None, None

    try:
        with open(incar, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    except Exception:
        return old_value, None
    return old_value, new_value


def _make_retry_summary(
    *,
    stage: str,
    error_type: str,
    trigger_pattern: str,
    retries_used: int,
    max_retries: int,
    recommended_action: str,
    applied_action: str,
    final_outcome: str,
    backup_file: Optional[str] = None,
    actions: Optional[list[dict[str, Any]]] = None,
    error_summary: Optional[str] = None,
) -> dict[str, Any]:
    summary = {
        "stage": stage,
        "error_type": error_type,
        "trigger_pattern": trigger_pattern,
        "retries_used": int(retries_used),
        "max_retries": int(max_retries),
        "recommended_action": recommended_action,
        "applied_action": applied_action,
        "final_outcome": final_outcome,
    }
    if backup_file:
        summary["backup_file"] = backup_file
        summary["previous_poscar_backup_path"] = backup_file
    if error_summary:
        summary["error_summary"] = error_summary
    if actions:
        summary["actions"] = actions
    return summary


def _classify_abortable_vasp_failure(workdir: str, *, merged_text: str = "", returncode: int | None = None) -> Optional[dict[str, str]]:
    text = "\n".join([str(merged_text or ""), read_vasp_failure_log_text(workdir)])
    error_type, trigger_pattern = classify_vasp_failure_text(text)
    if error_type not in {"runner_environment_failure", "chgcar_compatibility_failure"}:
        return None
    return {
        "error_type": error_type,
        "trigger_pattern": trigger_pattern if returncode is None else f"{trigger_pattern}; RETURNCODE_{returncode}",
        "error_summary": f"{error_type}: {trigger_pattern}",
    }


async def _tail_follow_one(path: str, *, stop_event: asyncio.Event, out_q: asyncio.Queue[str]) -> None:
    """Follow a single file with tail -F and push lines into out_q."""

    Path(path).touch(exist_ok=True)
    cmd = ["tail", "-n", "0", "-F", path]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    assert proc.stdout is not None
    try:
        while True:
            if stop_event.is_set():
                break

            read_task = asyncio.create_task(proc.stdout.readline())
            stop_task = asyncio.create_task(stop_event.wait())
            done, pending = await asyncio.wait(
                {read_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for t in pending:
                t.cancel()

            if stop_task in done:
                read_task.cancel()
                break

            line = read_task.result()
            if not line:
                await asyncio.sleep(0.05)
                continue

            try:
                out_q.put_nowait(line.decode("utf-8", errors="replace"))
            except Exception:
                pass
    finally:
        try:
            proc.terminate()
        except Exception:
            pass

        try:
            await asyncio.wait_for(proc.wait(), timeout=1.5)
        except Exception:
            pass

        if proc.returncode is None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.5)
            except Exception:
                pass


def _discover_monitor_files(workdir: str) -> list[str]:
    files: list[str] = []

    # VASP stdout redirected here by default script
    files.append(os.path.join(workdir, "sout"))

    # slurm output (if submitted in slurm wrapper by user)
    for p in sorted(Path(workdir).glob("slurm-*.out")):
        files.append(str(p))

    # any extra logs
    for p in sorted(Path(workdir).glob("*.log")):
        files.append(str(p))

    # de-dup preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        uniq.append(f)
    return uniq


def _default_convergence_ok(workdir: str) -> bool:
    osz = os.path.join(workdir, "OSZICAR")
    if not os.path.exists(osz):
        return False
    try:
        with open(osz, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return bool(lines and ("F=" in lines[-1]))
    except Exception:
        return False


def run_relax_vasp_with_retry(
    *,
    workdir: str,
    material_id: str,
    vasp_cmd: str,
    retry_log_path: str,
    enabled: bool,
    check_convergence: bool,
    convergence_ok: Callable[[str], bool] = _default_convergence_ok,
    max_retries: int = 3,
    retry_on_nonzero_exit: bool = True,
    retry_on_nonconvergence: bool = True,
    extra_monitored_files: Optional[Iterable[str]] = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Run relax stage with retry.

    Returns: (ok, backups_created, recovery_summary)

    Notes:
    - Uses tail -F to monitor merged logs.
    - On banner detection, kills process group, backs up CONTCAR, promotes to POSCAR, and retries.
    - Optionally also retries on non-zero exit and/or nonconvergence (by promoting CONTCAR -> POSCAR).
    """

    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)

    # resume support
    retry_n0, _last = _maybe_resume_from_checkpoint(workdir)

    if retry_n0 >= max_retries:
        append_retry_log(
            log_path=retry_log_path,
            material_id=material_id,
            retry_n=retry_n0,
            error_snippet="SKIP_AFTER_3_RETRIES",
            backup_file=_last,
        )
        raise RelaxRetryFatal(
            "SKIP_AFTER_3_RETRIES",
            summary=_make_retry_summary(
                stage="relax",
                error_type="retry_limit_reached",
                trigger_pattern="SKIP_AFTER_3_RETRIES",
                retries_used=retry_n0,
                max_retries=max_retries,
                recommended_action="skip_material",
                applied_action="skip_material",
                final_outcome="failed",
                backup_file=_last,
            ),
        )

    backups: list[str] = []
    recovery_actions: list[dict[str, Any]] = []

    def _start() -> subprocess.Popen:
        # make sure sout exists so tail -F can follow
        Path(os.path.join(workdir, "sout")).touch(exist_ok=True)
        return subprocess.Popen(
            vasp_cmd,
            shell=True,
            cwd=workdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

    if not enabled:
        proc = _start()
        proc.wait()
        if proc.returncode != 0:
            abortable = _classify_abortable_vasp_failure(workdir, returncode=proc.returncode)
            if abortable is not None:
                return False, backups, _make_retry_summary(
                    stage="relax",
                    error_type=abortable["error_type"],
                    trigger_pattern=abortable["trigger_pattern"],
                    retries_used=0,
                    max_retries=max_retries,
                    recommended_action="repair_execution_context",
                    applied_action="abort_stage_retry",
                    final_outcome="failed",
                    actions=[{"action": "abort_stage_retry", **abortable}],
                    error_summary=abortable["error_summary"],
                )
            return False, backups, _make_retry_summary(
                stage="relax",
                error_type="nonzero_exit",
                trigger_pattern=f"RETURNCODE_{proc.returncode}",
                retries_used=0,
                max_retries=max_retries,
                recommended_action="retry",
                applied_action="retry",
                final_outcome="failed",
            )
        if check_convergence and (not convergence_ok(workdir)):
            return False, backups, _make_retry_summary(
                stage="relax",
                error_type="nonconverged",
                trigger_pattern="NONCONVERGED",
                retries_used=0,
                max_retries=max_retries,
                recommended_action="retry",
                applied_action="retry",
                final_outcome="failed",
            )
        return True, backups, _make_retry_summary(
            stage="relax",
            error_type="none",
            trigger_pattern="normal_completion",
            retries_used=0,
            max_retries=max_retries,
                recommended_action="continue",
                applied_action="continue",
                final_outcome="success",
            )

    for retry_n in range(retry_n0, max_retries + 1):
        # --- sync polling monitor (robust & deterministic) ---
        merged_text = ""
        banner_hit = False
        buf_limit = 120_000
        offsets: dict[str, int] = {}

        # Match `tail -n 0 -F` semantics: ignore existing file contents at attempt start,
        # and only react to new content appended during this attempt.
        # NOTE: initialize offsets BEFORE starting the subprocess to reduce race window.
        Path(os.path.join(workdir, "sout")).touch(exist_ok=True)
        init_paths = _discover_monitor_files(workdir)
        if extra_monitored_files:
            for p in extra_monitored_files:
                pp = os.path.join(workdir, str(p)) if (not os.path.isabs(str(p))) else str(p)
                init_paths.append(pp)
        for p in init_paths:
            try:
                offsets[p] = int(os.path.getsize(p))
            except Exception:
                offsets[p] = 0

        proc = _start()

        def _read_new(p: str) -> str:
            try:
                st = os.stat(p)
                size = int(st.st_size)
                off = int(offsets.get(p, 0))
                # file truncated/rotated
                if size < off:
                    off = 0
                with open(p, "rb") as f:
                    f.seek(off)
                    data_b = f.read()
                offsets[p] = off + len(data_b)
                return data_b.decode("utf-8", errors="replace")
            except Exception:
                return ""

        while proc.poll() is None:
            paths = _discover_monitor_files(workdir)
            if extra_monitored_files:
                for p in extra_monitored_files:
                    pp = os.path.join(workdir, str(p)) if (not os.path.isabs(str(p))) else str(p)
                    paths.append(pp)
            # de-dup preserve order
            seen: set[str] = set()
            uniq_paths: list[str] = []
            for p in paths:
                if p in seen:
                    continue
                seen.add(p)
                uniq_paths.append(p)

            for p in uniq_paths:
                merged_text += _read_new(p)
                if len(merged_text) > buf_limit:
                    merged_text = merged_text[-buf_limit:]
                if detect_zbrent_banner(merged_text):
                    banner_hit = True
                    break
            if banner_hit:
                break
            time.sleep(0.2)

        # If banner hit, terminate early; do not wait for natural completion.
        if banner_hit:
            _kill_process_group(proc, term_wait_s=5.0)

        # Ensure process has ended before we inspect return code / outputs.
        proc.wait()

        # If banner not hit, decide success or retry based on exit code / convergence
        if not banner_hit:
            ok_exit = (proc.returncode == 0)
            ok_conv = (not check_convergence) or bool(convergence_ok(workdir))

            if ok_exit and ok_conv:
                return True, backups, _make_retry_summary(
                    stage="relax",
                    error_type="none",
                    trigger_pattern="normal_completion",
                retries_used=len(backups),
                max_retries=max_retries,
                recommended_action="continue",
                applied_action=(recovery_actions[-1]["action"] if recovery_actions else "continue"),
                final_outcome="success",
                backup_file=(backups[-1] if backups else None),
                actions=recovery_actions,
            )

            failure_reason = None
            failure_type = "unknown_failure"
            if (not ok_exit) and retry_on_nonzero_exit:
                abortable = _classify_abortable_vasp_failure(workdir, merged_text=merged_text, returncode=proc.returncode)
                if abortable is not None:
                    return False, backups, _make_retry_summary(
                        stage="relax",
                        error_type=abortable["error_type"],
                        trigger_pattern=abortable["trigger_pattern"],
                        retries_used=len(backups),
                        max_retries=max_retries,
                        recommended_action="repair_execution_context",
                        applied_action="abort_stage_retry",
                        final_outcome="failed",
                        backup_file=(backups[-1] if backups else None),
                        actions=recovery_actions + [{"action": "abort_stage_retry", **abortable}],
                        error_summary=abortable["error_summary"],
                    )
                failure_reason = f"RETURNCODE_{proc.returncode}"
                failure_type = "nonzero_exit"
            if (not ok_conv) and retry_on_nonconvergence:
                failure_reason = failure_reason or "NONCONVERGED"
                failure_type = "nonconverged"

            # If we are not configured to retry on this kind of failure, fail fast.
            if failure_reason is None:
                return False, backups, _make_retry_summary(
                    stage="relax",
                    error_type=failure_type,
                    trigger_pattern=(f"RETURNCODE_{proc.returncode}" if not ok_exit else "NONCONVERGED"),
                    retries_used=len(backups),
                    max_retries=max_retries,
                    recommended_action="abort_workflow",
                    applied_action="abort_workflow",
                    final_outcome="failed",
                    backup_file=(backups[-1] if backups else None),
                    actions=recovery_actions,
                )

            next_retry_n = retry_n + 1
            if next_retry_n > max_retries:
                append_retry_log(
                    log_path=retry_log_path,
                    material_id=material_id,
                    retry_n=retry_n,
                    error_snippet=failure_reason,
                    backup_file=None,
                )
                ck = RelaxCheckpoint(
                    material_id=material_id,
                    retry_n=max_retries,
                    last_backup_file=(backups[-1] if backups else None),
                )
                ck.save(workdir)
                raise RelaxRetryFatal(
                    "RETRY_LIMIT_REACHED",
                    summary=_make_retry_summary(
                        stage="relax",
                        error_type="retry_limit_reached",
                        trigger_pattern=failure_reason,
                        retries_used=max_retries,
                        max_retries=max_retries,
                        recommended_action="skip_material",
                        applied_action="skip_material",
                        final_outcome="failed",
                        backup_file=(backups[-1] if backups else None),
                        actions=recovery_actions,
                    ),
                )

            # Promote CONTCAR -> POSCAR and retry (if CONTCAR exists)
            try:
                backup_file = _backup_and_promote_contcar(workdir, next_retry_n)
            except FileNotFoundError:
                return False, backups, _make_retry_summary(
                    stage="relax",
                    error_type="missing_output",
                    trigger_pattern="CONTCAR_MISSING",
                    retries_used=len(backups),
                    max_retries=max_retries,
                    recommended_action="skip_material",
                    applied_action="skip_material",
                    final_outcome="failed",
                    actions=recovery_actions,
                )

            ediff_before, ediff_after = _tighten_ediff_in_incar(workdir)

            backups.append(backup_file)
            recovery_actions.append(
                {
                    "retry_n": next_retry_n,
                    "error_type": failure_type,
                    "trigger_pattern": failure_reason,
                    "action": "copy_contcar_to_poscar_and_retry",
                    "backup_file": backup_file,
                    "previous_poscar_backup_path": backup_file,
                    "promoted_from": "CONTCAR",
                    "ediff_before": ediff_before,
                    "ediff_after": ediff_after,
                }
            )

            ck = RelaxCheckpoint(material_id=material_id, retry_n=next_retry_n, last_backup_file=backup_file)
            ck.save(workdir)

            append_retry_log(
                log_path=retry_log_path,
                material_id=material_id,
                retry_n=next_retry_n,
                error_snippet=failure_reason,
                backup_file=backup_file,
            )

            # retry
            continue

        # banner detected => retry flow

        # error_snippet: last few lines
        snippet_lines = merged_text.splitlines()[-12:]
        error_snippet = " / ".join([s.strip() for s in snippet_lines if s.strip()])
        if len(error_snippet) > 500:
            error_snippet = error_snippet[-500:]

        next_retry_n = retry_n + 1
        if next_retry_n > max_retries:
            append_retry_log(
                log_path=retry_log_path,
                material_id=material_id,
                retry_n=retry_n,
                error_snippet=error_snippet or "ZBRENT_FATAL",
                backup_file=None,
            )
            ck = RelaxCheckpoint(material_id=material_id, retry_n=max_retries, last_backup_file=(backups[-1] if backups else None))
            ck.save(workdir)
            raise RelaxRetryFatal(
                "RETRY_LIMIT_REACHED",
                summary=_make_retry_summary(
                    stage="relax",
                    error_type="retry_limit_reached",
                    trigger_pattern=(error_snippet or "ZBRENT_FATAL"),
                    retries_used=max_retries,
                    max_retries=max_retries,
                    recommended_action="skip_material",
                    applied_action="skip_material",
                    final_outcome="failed",
                    backup_file=(backups[-1] if backups else None),
                    actions=recovery_actions,
                ),
            )

        backup_file = _backup_and_promote_contcar(workdir, next_retry_n)
        ediff_before, ediff_after = _tighten_ediff_in_incar(workdir)
        backups.append(backup_file)
        recovery_actions.append(
            {
                "retry_n": next_retry_n,
                "error_type": "rerun_with_smaller_ediff_or_copy_contcar",
                "trigger_pattern": (error_snippet or "ZBRENT_FATAL"),
                "action": "copy_contcar_to_poscar_and_retry",
                "backup_file": backup_file,
                "previous_poscar_backup_path": backup_file,
                "promoted_from": "CONTCAR",
                "ediff_before": ediff_before,
                "ediff_after": ediff_after,
            }
        )

        ck = RelaxCheckpoint(material_id=material_id, retry_n=next_retry_n, last_backup_file=backup_file)
        ck.save(workdir)

        append_retry_log(
            log_path=retry_log_path,
            material_id=material_id,
            retry_n=next_retry_n,
            error_snippet=error_snippet or "ZBRENT_FATAL",
            backup_file=backup_file,
        )

        # continue loop to re-submit

    # loop should have returned or raised above
    raise RelaxRetryFatal(
        "RETRY_LIMIT_REACHED",
        summary=_make_retry_summary(
            stage="relax",
            error_type="retry_limit_reached",
            trigger_pattern="RETRY_LIMIT_REACHED",
            retries_used=max_retries,
            max_retries=max_retries,
            recommended_action="skip_material",
            applied_action="skip_material",
            final_outcome="failed",
            backup_file=(backups[-1] if backups else None),
            actions=recovery_actions,
        ),
    )  # pragma: no cover
