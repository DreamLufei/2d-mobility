from __future__ import annotations

import math
import os
import shutil
import subprocess
from typing import Any, Optional


_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_CHGCAR_COMPATIBLE_INCAR_KEYS = {
    "ENCUT",
    "PREC",
    "LREAL",
    "LASPH",
    "ADDGRID",
    "ISPIN",
    "LSORBIT",
    "NGX",
    "NGY",
    "NGZ",
    "NGXF",
    "NGYF",
    "NGZF",
}
_RECOVERY_POLICY_ACTIONS = {"retry_capability", "rerun_from_capability", "repair_execution_context"}
_RUNNER_ENVIRONMENT_PATTERNS = (
    "mpirun: command not found",
    "mpiexec: command not found",
    "vasp_std: command not found",
    "vasp_gam: command not found",
    "vasp_ncl: command not found",
    "mpirun: not found",
    "mpiexec: not found",
    "vasp_std: not found",
    "vasp_gam: not found",
    "vasp_ncl: not found",
    "error while loading shared libraries",
    "cannot open shared object file",
    "ld_library_path",
    "libmpi",
    "libmkl",
    "libifcore",
    "libimf",
    "orted: command not found",
)
_CHGCAR_COMPATIBILITY_PATTERNS = (
    "chgcar",
    "charge density",
    "fft grid",
    "ngxf",
    "ngyf",
    "ngzf",
    "number of data items",
)


def reuse_completed_vasp_stages_enabled() -> bool:
    return str(os.environ.get("MOBILITY_REUSE_COMPLETED_VASP_STAGES") or "").strip().lower() in _TRUE_ENV_VALUES


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {}


def _stage_names(stage: str, stage_aliases: tuple[str, ...] | list[str] | set[str] | None = None) -> set[str]:
    names = {str(stage or "").strip()}
    names.update(str(item or "").strip() for item in (stage_aliases or ()))
    return {item for item in names if item}


def _payload_has_failed_stage(payload: dict[str, Any], names: set[str]) -> bool:
    workflow = _as_dict(payload.get("workflow"))
    stage_status = _as_dict(workflow.get("stage_status"))
    if any(str(stage_status.get(name) or "").strip() == "failed" for name in names):
        return True

    retry_counts = _as_dict(workflow.get("retry_counts"))
    for name in names:
        try:
            if int(retry_counts.get(name, 0) or 0) > 0:
                return True
        except Exception:
            pass

    diagnostics = _as_dict(payload.get("diagnostics"))
    recovery_summary = _as_dict(diagnostics.get("recovery_summary"))
    recovery_stage = str(recovery_summary.get("stage") or recovery_summary.get("current_stage") or "").strip()
    if recovery_stage in names and (recovery_summary.get("error_type") or recovery_summary.get("error_summary")):
        return True

    execution = _as_dict(payload.get("execution"))
    latest_observation = _as_dict(execution.get("latest_execution_observation"))
    if not latest_observation:
        latest_observation = _as_dict(_as_dict(payload.get("blackboard")).get("latest_execution_observation"))
    observation_stage = str(latest_observation.get("target_capability") or latest_observation.get("stage") or "").strip()
    observation_status = str(latest_observation.get("status") or "").strip()
    if observation_stage in names and observation_status in {"failed", "error"}:
        return True

    for item in list(execution.get("failure_history") or []):
        entry = _as_dict(item)
        failure_stage = str(entry.get("target_capability") or entry.get("stage") or "").strip()
        if failure_stage in names:
            return True

    return False


def policy_stage_planning_allowed(
    state_payload: dict[str, Any] | None,
    stage: str,
    *,
    stage_aliases: tuple[str, ...] | list[str] | set[str] | None = None,
) -> bool:
    """Allow RAG/policy parameter patches only after a concrete failed stage."""

    payload = _as_dict(state_payload)
    names = _stage_names(stage, stage_aliases)
    if not payload or not names:
        return False
    execution = _as_dict(payload.get("execution"))
    current_action = _as_dict(execution.get("current_action"))
    action_family = str(current_action.get("action_family") or "").strip()
    if action_family not in _RECOVERY_POLICY_ACTIONS:
        return False
    target = str(current_action.get("target_capability") or current_action.get("capability") or "").strip()
    if target and target not in names:
        return False
    return _payload_has_failed_stage(payload, names)


def _tail_text(path: str, *, max_bytes: int = 60_000) -> str:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            data = handle.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def read_vasp_failure_log_text(cwd: str, *, max_bytes_per_file: int = 60_000) -> str:
    candidates = [
        os.path.join(cwd, "sout"),
        os.path.join(cwd, "vasp.out"),
        os.path.join(cwd, "stdout"),
        os.path.join(cwd, "stderr"),
        os.path.join(cwd, "vasp_subprocess_stdout.log"),
        os.path.join(cwd, "vasp_subprocess_stderr.log"),
    ]
    try:
        for name in sorted(os.listdir(cwd)):
            if name.endswith((".out", ".err", ".log")):
                candidates.append(os.path.join(cwd, name))
    except Exception:
        pass
    chunks: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        text = _tail_text(path, max_bytes=max_bytes_per_file)
        if text:
            chunks.append(f"\n--- {os.path.basename(path)} ---\n{text}")
    return "\n".join(chunks)


def classify_vasp_failure_text(text: str) -> tuple[str, str]:
    value = str(text or "").lower()
    for pattern in _RUNNER_ENVIRONMENT_PATTERNS:
        if pattern in value:
            return "runner_environment_failure", pattern
    if "command not found" in value and any(token in value for token in ("mpirun", "mpiexec", "vasp")):
        return "runner_environment_failure", "command_not_found"
    if "no such file or directory" in value and any(token in value for token in ("mpirun", "mpiexec", "vasp_std", "vasp_gam", "vasp_ncl", "lib")):
        return "runner_environment_failure", "missing_executable_or_library"
    if any(pattern in value for pattern in _CHGCAR_COMPATIBILITY_PATTERNS) and any(token in value for token in ("mismatch", "incompatible", "wrong", "different", "error", "dimension")):
        return "chgcar_compatibility_failure", "chgcar_grid_or_dimension_mismatch"
    return "unknown_failure", "unknown_vasp_failure"


def summarize_vasp_failure(cwd: str, *, stage: str, default_error: str, returncode: int | None = None) -> dict[str, Any]:
    if returncode is None:
        try:
            with open(os.path.join(cwd, "vasp_returncode.txt"), "r", encoding="utf-8", errors="ignore") as handle:
                returncode = int(str(handle.read()).strip())
        except Exception:
            returncode = None
    text = read_vasp_failure_log_text(cwd)
    error_type, trigger_pattern = classify_vasp_failure_text(text)
    if error_type == "unknown_failure" and returncode is not None:
        error_type = "nonzero_exit"
        trigger_pattern = f"RETURNCODE_{returncode}"
    stage_label = str(stage or "vasp").upper()
    if error_type == "runner_environment_failure":
        message = f"{stage_label} runner/environment failure: {trigger_pattern}"
        recommended_action = "repair_execution_context"
    elif error_type == "chgcar_compatibility_failure":
        message = f"{stage_label} CHGCAR compatibility failure: {trigger_pattern}"
        recommended_action = "repair_execution_context"
    elif error_type == "nonzero_exit":
        message = f"{stage_label} nonzero exit: {trigger_pattern}"
        recommended_action = "retry_capability"
    else:
        message = default_error
        recommended_action = "retry_capability"
    return {
        "stage": stage,
        "error_type": error_type,
        "trigger_pattern": trigger_pattern,
        "error_summary": message,
        "recommended_action": recommended_action,
        "applied_action": "abort_stage" if recommended_action == "repair_execution_context" else "retry",
        "final_outcome": "failed",
    }


def read_chgcar_compatible_incar_overrides(base_dir: str) -> dict[str, Any]:
    scf_incar_path = os.path.join(base_dir, "02_scf", "INCAR")
    if not os.path.exists(scf_incar_path):
        return {}
    try:
        from pymatgen.io.vasp.inputs import Incar

        scf_incar = Incar.from_file(scf_incar_path)
    except Exception:
        return {}
    return {
        key: scf_incar[key]
        for key in sorted(_CHGCAR_COMPATIBLE_INCAR_KEYS)
        if key in scf_incar
    }


def symlink_force(src: str, dst: str) -> None:
    src_abs = os.path.abspath(src)
    dst_abs = os.path.abspath(dst)
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    try:
        if os.path.lexists(dst_abs):
            os.remove(dst_abs)
    except IsADirectoryError:
        shutil.rmtree(dst_abs, ignore_errors=True)
    src_rel = os.path.relpath(src_abs, start=os.path.dirname(dst_abs))
    os.symlink(src_rel, dst_abs)


def prune_dir_keep_files(folder: str, keep: set[str]) -> None:
    try:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if name in keep or name.startswith("CONTCAR.retry.") or os.path.isdir(path):
                continue
            try:
                os.remove(path)
            except Exception:
                pass
    except FileNotFoundError:
        return


def run_vasp(*, cwd: str, vasp_cmd: str, check_convergence: bool = True) -> bool:
    process = subprocess.Popen(
        vasp_cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_b, stderr_b = process.communicate()
    if stdout_b:
        with open(os.path.join(cwd, "vasp_subprocess_stdout.log"), "ab") as handle:
            handle.write(stdout_b)
    if stderr_b:
        with open(os.path.join(cwd, "vasp_subprocess_stderr.log"), "ab") as handle:
            handle.write(stderr_b)
    with open(os.path.join(cwd, "vasp_returncode.txt"), "w", encoding="utf-8") as handle:
        handle.write(str(process.returncode))
    if process.returncode != 0:
        return False
    if check_convergence:
        oszicar_path = os.path.join(cwd, "OSZICAR")
        if os.path.exists(oszicar_path):
            with open(oszicar_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            return bool(lines and "F=" in lines[-1])
    return True


def copy_poscar_potcar(poscar_path: str, potcar_path: str, dest_dir: str) -> None:
    shutil.copy(poscar_path, os.path.join(dest_dir, "POSCAR"))
    shutil.copy(potcar_path, os.path.join(dest_dir, "POTCAR"))


def _load_structure(path: str):
    from pymatgen.core import Structure

    return Structure.from_file(path)


def _load_incar(params: dict[str, Any]):
    from pymatgen.io.vasp.inputs import Incar

    return Incar(params)


def write_incar(path: str, params: dict[str, Any]) -> None:
    incar = _load_incar(params)
    incar.write_file(os.path.join(path, "INCAR"))


def _floor_to_even(n: int) -> int:
    return n if (n % 2 == 0) else (n - 1)


def _kmesh_from_poscar_2d(poscar_path: str, *, target_ka: float = 50.0) -> tuple[int, int, int]:
    struct = _load_structure(poscar_path)
    a, b, _c = struct.lattice.abc
    if a <= 0 or b <= 0:
        raise ValueError(f"POSCAR 晶格长度异常: a={a}, b={b}")
    kx = max(2, _floor_to_even(int(math.floor(float(target_ka) / float(a)))))
    ky = max(2, _floor_to_even(int(math.floor(float(target_ka) / float(b)))))
    return int(kx), int(ky), 1


def write_relax_scf_kpoints(
    dest_dir: str,
    *,
    material_name: str,
    target_ka: float = 50.0,
    gamma_centered: bool = False,
    explicit_mesh: tuple[int, int, int] | None = None,
) -> None:
    poscar_path = os.path.join(dest_dir, "POSCAR")
    if not os.path.exists(poscar_path):
        raise FileNotFoundError(f"POSCAR 不存在: {poscar_path}")
    if explicit_mesh is None:
        kx, ky, kz = _kmesh_from_poscar_2d(poscar_path, target_ka=target_ka)
    else:
        kx, ky, kz = explicit_mesh
    scheme = "G" if gamma_centered else "M"
    kpoints_content = f"""{material_name}
0
{scheme}
{kx} {ky} {kz}
0 0 0
"""
    with open(os.path.join(dest_dir, "KPOINTS"), "w", encoding="utf-8") as f:
        f.write(kpoints_content)


def write_band_kpoints(filepath: str, npoints_per_segment: int = 40) -> None:
    npoints = max(8, int(npoints_per_segment or 40))
    kpoints_content = f"""KPATH: G-X-S-Y-G
{npoints}
Line-Mode
Reciprocal
   0.0000000000   0.0000000000   0.0000000000     GAMMA
   0.5000000000   0.0000000000   0.0000000000     X

   0.5000000000   0.0000000000   0.0000000000     X
   0.5000000000   0.5000000000   0.0000000000     S

   0.5000000000   0.5000000000   0.0000000000     S
   0.0000000000   0.5000000000   0.0000000000     Y

   0.0000000000   0.5000000000   0.0000000000     Y
   0.0000000000   0.0000000000   0.0000000000     GAMMA
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(kpoints_content)


def build_incar_relax(system: str, *, ediff: float, isif: int, lattice_constraints: Optional[str] = None, consider_spin: bool = False, nsw: int = 200, encut: int = 600, nelm: int = 300, ivdw: int = 12) -> dict[str, Any]:
    params = {
        "SYSTEM": system,
        "ISTART": 0,
        "ICHARG": 2,
        "ISMEAR": 0,
        "SIGMA": 0.01,
        "LREAL": "Auto",
        "PREC": "Normal",
        "EDIFF": ediff,
        "ENCUT": encut,
        "NELM": nelm,
        "IVDW": ivdw,
        "ISIF": isif,
        "NSW": nsw,
        "IBRION": 2,
        "POTIM": 0.5,
        "EDIFFG": -0.01,
        "NELMIN": 4,
        "ALGO": "Normal",
        "LCHARG": False,
        "LWAVE": False,
        "LELF": False,
    }
    if lattice_constraints is not None:
        params["LATTICE_CONSTRAINTS"] = lattice_constraints
    if consider_spin:
        params["ISPIN"] = 2
    return params


def build_incar_scf(system: str, *, consider_spin: bool = False, ediff: float = 1e-6, encut: int = 600, nelm: int = 300, ivdw: int = 12, lvtot: bool = False, lvhar: bool = False) -> dict[str, Any]:
    params = {
        "SYSTEM": system,
        "ISTART": 0,
        "ICHARG": 2,
        "ISMEAR": 0,
        "SIGMA": 0.01,
        "LREAL": "Auto",
        "PREC": "Normal",
        "EDIFF": ediff,
        "ENCUT": encut,
        "NELM": nelm,
        "IVDW": ivdw,
        "NELMIN": 4,
        "ALGO": "Normal",
        "LCHARG": True,
        "LWAVE": True,
        "LELF": False,
    }
    if lvtot:
        params["LVTOT"] = True
    if lvhar:
        params["LVHAR"] = True
    if consider_spin:
        params["ISPIN"] = 2
    return params


def build_incar_band(system: str, *, consider_spin: bool = False, ediff: float = 1e-6, encut: int = 600, nelm: int = 300, ivdw: int = 12) -> dict[str, Any]:
    params = {
        "SYSTEM": system,
        "ISTART": 0,
        "ICHARG": 11,
        "ISMEAR": 0,
        "SIGMA": 0.01,
        "LREAL": "Auto",
        "PREC": "Normal",
        "EDIFF": ediff,
        "ENCUT": encut,
        "NELM": nelm,
        "IVDW": ivdw,
        "NELMIN": 4,
        "ALGO": "Normal",
        "LCHARG": False,
        "LWAVE": False,
        "LELF": False,
    }
    if consider_spin:
        params["ISPIN"] = 2
    return params
