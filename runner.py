"""Routing screen and task-level ablation execution."""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import shutil
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import (
    ExperimentConfig,
    SURFACES,
    apply_datamind_environment,
    import_datamind,
    sha256_file,
    sha256_json,
    validate_credentials,
)
from .tasks import load_tasks, required_surfaces, task_manifest
from .workspace import load_workspace_manifest, workspace_manifest_path


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _error(exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "status_code": getattr(exc, "status_code", None),
        "message": str(exc)[:1000],
    }


def condition_sets(required: Iterable[str]) -> list[tuple[str, frozenset[str]]]:
    """Return FULL and all strict non-empty subsets of one task's surfaces."""
    required_set = frozenset(required)
    ordered = sorted(required_set)
    result: list[tuple[str, frozenset[str]]] = [("full", required_set)]
    for size in range(1, len(ordered)):
        for combo in itertools.combinations(ordered, size):
            active = frozenset(combo)
            result.append(("subset_" + "_".join(combo), active))
    return result


@dataclass(frozen=True)
class Condition:
    name: str
    active_surfaces: frozenset[str]
    replicates: int
    allow_empty_control: bool = False


@dataclass(frozen=True)
class TaskPlan:
    task: dict[str, Any]
    mode: str
    group: str
    conditions: tuple[Condition, ...]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _replicates(value: Any, *, fallback: int) -> int:
    result = fallback if value is None else int(value)
    if result <= 0:
        raise ValueError("replicates must be positive")
    return result


def resolve_plan(tasks: list[dict[str, Any]], plan: dict[str, Any]) -> list[TaskPlan]:
    default = plan.get("default") or {"mode": "none"}
    groups = plan.get("groups") or []
    if not isinstance(default, dict) or not isinstance(groups, list):
        raise ValueError("ablation plan default/groups have invalid types")
    global_replicates = _replicates(plan.get("replicates"), fallback=3)
    by_id = {str(task["task_id"]): task for task in tasks}
    assignments: dict[str, tuple[str, dict[str, Any]]] = {}
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"groups[{index}] must be an object")
        name = str(group.get("name") or f"group_{index + 1}")
        group_tasks = group.get("tasks") or []
        if not isinstance(group_tasks, list) or not group_tasks:
            raise ValueError(f"group {name!r} must list tasks")
        for raw_id in group_tasks:
            task_id = str(raw_id)
            if task_id not in by_id:
                raise ValueError(f"group {name!r} references unknown task {task_id!r}")
            if task_id in assignments:
                raise ValueError(f"task {task_id!r} appears in multiple groups")
            assignments[task_id] = (name, group)

    result: list[TaskPlan] = []
    for task in tasks:
        task_id = str(task["task_id"])
        group_name, spec = assignments.get(task_id, ("default", default))
        mode = str(spec.get("mode", "none"))
        if mode not in {"full", "selected", "none"}:
            raise ValueError(f"task {task_id}: unsupported mode {mode!r}")
        required = required_surfaces(task)
        group_replicates = _replicates(spec.get("replicates"), fallback=global_replicates)
        conditions: list[Condition] = []
        if mode == "full":
            conditions = [
                Condition(name=name, active_surfaces=active, replicates=group_replicates)
                for name, active in condition_sets(required)
            ]
        elif mode == "selected":
            raw_conditions = spec.get("conditions") or []
            if not isinstance(raw_conditions, list) or not raw_conditions:
                raise ValueError(f"task {task_id}: selected mode requires conditions")
            names: set[str] = set()
            for index, raw_condition in enumerate(raw_conditions):
                if not isinstance(raw_condition, dict):
                    raise ValueError(f"task {task_id}: condition {index} must be an object")
                name = str(raw_condition.get("name", "")).strip()
                active = frozenset(str(item) for item in (raw_condition.get("active_surfaces") or []))
                allow_empty = bool(raw_condition.get("allow_empty_control", False))
                if not name or name in names:
                    raise ValueError(f"task {task_id}: condition names must be unique")
                if not active <= required or not active <= SURFACES:
                    raise ValueError(
                        f"task {task_id}: {name} active_surfaces must be a subset of {sorted(required)}"
                    )
                if not active and not allow_empty:
                    raise ValueError(f"task {task_id}: empty active_surfaces requires allow_empty_control")
                names.add(name)
                conditions.append(Condition(
                    name=name,
                    active_surfaces=active,
                    replicates=_replicates(raw_condition.get("replicates"), fallback=group_replicates),
                    allow_empty_control=allow_empty,
                ))
        result.append(TaskPlan(task=task, mode=mode, group=group_name, conditions=tuple(conditions)))
    return result


def _source_path(config: ExperimentConfig, manifest: dict[str, Any]) -> Path:
    source = Path(str(manifest["source"]))
    if not source.is_absolute():
        source = config.root / source
    return source.resolve()


def _run_manifest(
    *,
    config: ExperimentConfig,
    phase: str,
    tasks_path: Path,
    tasks: list[dict[str, Any]],
    workspace: dict[str, Any],
    run_id: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_record = task_manifest(tasks, tasks_path)
    if tasks_path.is_relative_to(config.root):
        task_record["source"] = str(tasks_path.relative_to(config.root))
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": phase,
        "created_at": _timestamp(),
        "task_manifest": task_record,
        "workspace_manifest": str(workspace_manifest_path(config).relative_to(config.root)),
        "workspace_sha256": sha256_json(workspace),
        "settings": config.public_manifest(),
        "settings_fingerprint": config.settings_fingerprint,
    }
    if extra:
        manifest.update(extra)
    return manifest


def _check_resume(manifest: dict[str, Any], config: ExperimentConfig, *, plan_hash: str | None = None) -> None:
    if manifest.get("settings_fingerprint") != config.settings_fingerprint:
        raise ValueError("run settings fingerprint differs; create a new run for API changes")
    if plan_hash is not None and manifest.get("plan_sha256") != plan_hash:
        raise ValueError("ablation plan changed; create a new run")


def _read_secret_fingerprint(run_dir: Path) -> str | None:
    path = run_dir / ".secret_fingerprint"
    return path.read_text(encoding="ascii").strip() if path.is_file() else None


def _write_secret_fingerprint(run_dir: Path, config: ExperimentConfig) -> None:
    (run_dir / ".secret_fingerprint").write_text(config.secret_fingerprint + "\n", encoding="ascii")


def _routing_classification(task: dict[str, Any], row: dict[str, Any]) -> str:
    required = required_surfaces(task)
    used = set(row.get("surfaces_used", []))
    if row.get("error"):
        return "runner_error"
    if not row.get("answer"):
        return "model_failure"
    if "graph" not in required:
        return "graph_not_required"
    return "graph_bypass_candidate" if "graph" not in used else "graph_used"


async def _build_system(config: ExperimentConfig, active: frozenset[str]) -> Any:
    import_datamind(config)
    from datamind.agent import build_datamind
    from datamind.config import Settings

    return await build_datamind(Settings(), enable=set(active))


async def _query(system: Any, task: dict[str, Any], *, retries: int) -> dict[str, Any]:
    attempts = 0
    started = time.perf_counter()
    answer: dict[str, Any] = {}
    error: dict[str, Any] | None = None
    while attempts <= retries:
        attempts += 1
        try:
            answer = await system.query(str(task["question"]))
            error = None
            break
        except Exception as exc:  # benchmark runner records failures per task
            error = _error(exc)
            status = getattr(exc, "status_code", None)
            transient = isinstance(exc, (TimeoutError, ConnectionError)) or status in {408, 409, 429, 500, 502, 503, 504}
            if not transient or attempts > retries:
                break
            await asyncio.sleep(min(4.0, 0.5 * (2 ** (attempts - 1))))
    elapsed = time.perf_counter() - started
    answer = answer or {}
    return {
        "answer": answer.get("answer", ""),
        "usage": answer.get("usage", {}),
        "resolved_models": (answer.get("usage") or {}).get("resolved_models", []),
        "iterations": answer.get("iterations"),
        "stop_reason": answer.get("stop_reason"),
        "surfaces_used": answer.get("surfaces_used", []),
        "tool_trace": answer.get("tool_trace", []),
        "evidence": answer.get("evidence", []),
        "error": error,
        "attempts": attempts,
        "latency_s": round(elapsed, 6),
    }


def _row(task: dict[str, Any], *, condition: str, active: frozenset[str], replicate: int, run_id: str, result: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "task_id": str(task["task_id"]),
        "condition": condition,
        "mode": mode,
        "replicate": replicate,
        "active_surfaces": sorted(active),
        "required_surfaces": sorted(required_surfaces(task)),
        "question": task["question"],
        "reference_answer": task.get("reference_answer", ""),
        "metadata": task.get("metadata", {}),
        "answer": result.get("answer", ""),
        "usage": result.get("usage", {}),
        "resolved_models": result.get("resolved_models", []),
        "iterations": result.get("iterations"),
        "stop_reason": result.get("stop_reason"),
        "surfaces_used": result.get("surfaces_used", []),
        "tool_trace": result.get("tool_trace", []),
        "evidence": result.get("evidence", []),
        "error": result.get("error"),
        "attempts": result.get("attempts"),
        "latency_s": result.get("latency_s"),
        "completed_at": _timestamp(),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_screen_summary(run_dir: Path) -> None:
    rows = _load_jsonl(run_dir / "screen" / "results.jsonl")
    counts: dict[str, int] = defaultdict(int)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        classification = str(row.get("classification", "unknown"))
        counts[classification] += 1
        if classification == "graph_bypass_candidate":
            candidates.append({
                "task_id": row.get("task_id"),
                "required_surfaces": row.get("required_surfaces", []),
                "surfaces_used": row.get("surfaces_used", []),
                "answer": row.get("answer", ""),
            })
    summary = {
        "schema_version": 1,
        "rows": len(rows),
        "classifications": dict(sorted(counts.items())),
        "graph_bypass_candidates": candidates,
    }
    _atomic_json(run_dir / "screen" / "routing_screen.json", summary)
    lines = [
        "# Routing Screen",
        "",
        f"Tasks: {len(rows)}",
        "",
        "| Classification | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in sorted(counts.items()))
    (run_dir / "screen" / "routing_screen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import csv
    with (run_dir / "screen" / "routing_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "required_surfaces", "surfaces_used", "answer"])
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(candidate)


async def _run_screen_async(
    config: ExperimentConfig,
    tasks: list[dict[str, Any]],
    *,
    run_id: str,
    run_dir: Path,
    retries: int,
    resume: bool,
) -> dict[str, Any]:
    systems: dict[frozenset[str], Any] = {}
    rows: list[dict[str, Any]] = _load_jsonl(run_dir / "screen" / "results.jsonl") if resume else []
    completed = {str(row.get("task_id")) for row in rows}
    try:
        for task in tasks:
            if str(task["task_id"]) in completed:
                continue
            active = required_surfaces(task)
            if active not in systems:
                systems[active] = await _build_system(config, active)
                await systems[active].warmup()
            result = await _query(systems[active], task, retries=retries)
            row = _row(task, condition="full", active=active, replicate=1, run_id=run_id, result=result, mode="screen")
            row["classification"] = _routing_classification(task, row)
            _append_jsonl(run_dir / "screen" / "results.jsonl", row)
            rows.append(row)
    finally:
        for system in systems.values():
            await system.aclose()
    status = {
        "status": "complete" if len(rows) == len(tasks) else "incomplete",
        "expected_tasks": len(tasks),
        "observed_tasks": len(rows),
        "errors": sum(1 for row in rows if row.get("error")),
        "finished_at": _timestamp(),
    }
    _atomic_json(run_dir / "screen" / "status.json", status)
    _write_screen_summary(run_dir)
    return status


def run_screen(
    config: ExperimentConfig,
    tasks_path: Path,
    selected: list[dict[str, Any]],
    *,
    run_id: str | None = None,
    retries: int = 1,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    if dry_run:
        try:
            workspace = load_workspace_manifest(config)
        except FileNotFoundError:
            workspace = {"status": "not_prepared"}
    else:
        workspace = load_workspace_manifest(config)
    run_id = run_id or f"edu-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_dir = config.output_dir / run_id
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite existing run: {run_dir}; use --resume")
        manifest = _read_json(run_dir / "run_manifest.json")
        if manifest.get("phase") != "screen":
            raise ValueError(f"not a screen run: {run_dir}")
        _check_resume(manifest, config)
        existing_secret = _read_secret_fingerprint(run_dir)
        if existing_secret and existing_secret != config.secret_fingerprint:
            raise ValueError("resume API key differs; create a new run")
        if manifest.get("workspace_sha256") != sha256_json(workspace):
            raise ValueError("resume workspace manifest differs; create a new run")
        expected_ids = manifest.get("task_manifest", {}).get("task_ids", [])
        selected_ids = [str(task["task_id"]) for task in selected]
        if expected_ids != selected_ids:
            raise ValueError("resume task selection differs from the existing screen run")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = _run_manifest(
            config=config, phase="screen", tasks_path=tasks_path, tasks=selected,
            workspace=workspace, run_id=run_id,
        )
        _atomic_json(run_dir / "run_manifest.json", manifest)
    if dry_run:
        _atomic_json(run_dir / "screen" / "status.json", {"status": "dry_run", "expected_tasks": len(selected)})
        return run_dir
    validate_credentials(config)
    apply_datamind_environment(config)
    _write_secret_fingerprint(run_dir, config)
    status = asyncio.run(_run_screen_async(config, selected, run_id=run_id, run_dir=run_dir, retries=retries, resume=resume))
    manifest["status"] = status
    _atomic_json(run_dir / "run_manifest.json", manifest)
    return run_dir


async def _run_ablation_async(
    config: ExperimentConfig,
    plans: list[TaskPlan],
    *,
    run_id: str,
    run_dir: Path,
    retries: int,
) -> list[dict[str, Any]]:
    systems: dict[frozenset[str], Any] = {}
    statuses: list[dict[str, Any]] = []
    try:
        for task_plan in plans:
            for condition in task_plan.conditions:
                active = condition.active_surfaces
                if active not in systems:
                    systems[active] = await _build_system(config, active)
                    await systems[active].warmup()
                for replicate in range(1, condition.replicates + 1):
                    task_id = str(task_plan.task["task_id"])
                    output = run_dir / "ablation" / task_id / condition.name / f"rep{replicate:02d}.jsonl"
                    status_path = output.with_suffix(".status.json")
                    manifest_path = output.with_suffix(".manifest.json")
                    if status_path.is_file() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "complete":
                        statuses.append(json.loads(status_path.read_text(encoding="utf-8")))
                        continue
                    previous = _load_jsonl(output)
                    if previous:
                        existing_row = next((row for row in previous if str(row.get("task_id")) == task_id), None)
                        if existing_row is not None:
                            _atomic_json(status_path, {
                                "status": "complete",
                                "run_id": run_id,
                                "task_id": task_id,
                                "condition": condition.name,
                                "active_surfaces": sorted(active),
                                "replicate": replicate,
                                "error": bool(existing_row.get("error")),
                                "finished_at": existing_row.get("completed_at", _timestamp()),
                            })
                            statuses.append(_read_json(status_path))
                            continue
                    _atomic_json(manifest_path, {
                        "schema_version": 1,
                        "run_id": run_id,
                        "task_id": task_id,
                        "condition": condition.name,
                        "active_surfaces": sorted(active),
                        "mode": task_plan.mode,
                        "replicate": replicate,
                        "settings_fingerprint": config.settings_fingerprint,
                    })
                    result = await _query(systems[active], task_plan.task, retries=retries)
                    row = _row(
                        task_plan.task, condition=condition.name, active=active,
                        replicate=replicate, run_id=run_id, result=result, mode=task_plan.mode,
                    )
                    _append_jsonl(output, row)
                    status = {
                        "status": "complete",
                        "run_id": run_id,
                        "task_id": task_id,
                        "condition": condition.name,
                        "active_surfaces": sorted(active),
                        "replicate": replicate,
                        "error": bool(result.get("error")),
                        "finished_at": _timestamp(),
                    }
                    _atomic_json(status_path, status)
                    statuses.append(status)
    finally:
        for system in systems.values():
            await system.aclose()
    return statuses


def run_ablation(
    config: ExperimentConfig,
    screen_run_dir: Path,
    plan_path: Path,
    *,
    retries: int = 1,
    new_run_id: str | None = None,
) -> Path:
    source_manifest = _read_json(screen_run_dir / "run_manifest.json")
    if source_manifest.get("phase") != "screen":
        raise ValueError(f"not a screen run: {screen_run_dir}")
    validate_credentials(config)
    apply_datamind_environment(config)
    settings_changed = source_manifest.get("settings_fingerprint") != config.settings_fingerprint
    secret_changed = (_read_secret_fingerprint(screen_run_dir) or config.secret_fingerprint) != config.secret_fingerprint
    if (settings_changed or secret_changed) and not new_run_id:
        raise ValueError("run settings fingerprint differs; pass --new-run-id for an API config branch")
    if new_run_id:
        if new_run_id == source_manifest.get("run_id"):
            raise ValueError("new-run-id must differ from the source screen run")
        branch_dir = config.output_dir / new_run_id
        if branch_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing branch run: {branch_dir}")
        branch_dir.mkdir(parents=True)
        source_results = screen_run_dir / "screen" / "results.jsonl"
        if source_results.is_file():
            target_results = branch_dir / "screen" / "results.jsonl"
            target_results.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_results, target_results)
        source_status = screen_run_dir / "screen" / "status.json"
        if source_status.is_file():
            target_status = branch_dir / "screen" / "status.json"
            target_status.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_status, target_status)
        source_manifest = dict(source_manifest)
        source_manifest["parent_run_id"] = source_manifest.get("run_id")
        source_manifest["run_id"] = new_run_id
        source_manifest["settings"] = config.public_manifest()
        source_manifest["settings_fingerprint"] = config.settings_fingerprint
        source_manifest["created_at"] = _timestamp()
        screen_run_dir = branch_dir
        _atomic_json(screen_run_dir / "run_manifest.json", source_manifest)
        _write_secret_fingerprint(screen_run_dir, config)
    else:
        _check_resume(source_manifest, config)
    task_source = _source_path(config, source_manifest["task_manifest"])
    tasks = load_tasks(task_source)
    selected_ids = set(source_manifest["task_manifest"]["task_ids"])
    selected = [task for task in tasks if str(task["task_id"]) in selected_ids]
    plan = _read_json(plan_path)
    plan_hash = sha256_file(plan_path)
    plans = resolve_plan(selected, plan)
    manifest_path = screen_run_dir / "ablation_manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("plan_sha256") != plan_hash:
            raise ValueError("ablation plan changed for this run; create a new screen run")
    else:
        _atomic_json(manifest_path, {
            "schema_version": 1,
            "run_id": source_manifest["run_id"],
            "phase": "ablation",
            "plan": str(plan_path.relative_to(config.root) if plan_path.is_relative_to(config.root) else plan_path),
            "plan_sha256": plan_hash,
            "settings_fingerprint": config.settings_fingerprint,
            "task_ids": sorted(selected_ids),
            "created_at": _timestamp(),
        })
    statuses = asyncio.run(_run_ablation_async(config, plans, run_id=source_manifest["run_id"], run_dir=screen_run_dir, retries=retries))
    _atomic_json(screen_run_dir / "ablation_status.json", {
        "status": "complete",
        "conditions": len(statuses),
        "finished_at": _timestamp(),
    })
    return screen_run_dir
