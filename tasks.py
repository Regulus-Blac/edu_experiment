"""Task loading, validation and deterministic selection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import SURFACES, sha256_file


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(task, dict):
            raise ValueError(f"{path}:{line_no}: task must be an object")
        task_id = str(task.get("task_id", "")).strip()
        question = str(task.get("question", "")).strip()
        reference = str(task.get("reference_answer", "")).strip()
        metadata = task.get("metadata") or {}
        required = metadata.get("required_surfaces") if isinstance(metadata, dict) else None
        if not task_id or task_id in seen:
            raise ValueError(f"{path}:{line_no}: missing or duplicate task_id")
        if not question or not reference:
            raise ValueError(f"{path}:{line_no}: question/reference_answer is empty")
        if not isinstance(required, list) or not required:
            raise ValueError(f"{path}:{line_no}: required_surfaces must be a non-empty list")
        if len(required) != len(set(required)) or not set(required) <= SURFACES:
            raise ValueError(f"{path}:{line_no}: invalid required_surfaces {required!r}")
        seen.add(task_id)
        tasks.append(task)
    if not tasks:
        raise ValueError(f"no tasks found in {path}")
    return tasks


def required_surfaces(task: dict[str, Any]) -> frozenset[str]:
    return frozenset((task.get("metadata") or {}).get("required_surfaces", []))


def select_tasks(
    tasks: list[dict[str, Any]],
    *,
    count: int | None = None,
    task_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if count is not None and count <= 0:
        raise ValueError("count must be positive")
    by_id = {str(task["task_id"]): task for task in tasks}
    if task_ids:
        requested = [item.strip() for item in task_ids if item.strip()]
        missing = [item for item in requested if item not in by_id]
        if missing:
            raise ValueError(f"unknown task IDs: {', '.join(missing)}")
        selected = [by_id[item] for item in requested]
    else:
        selected = tasks[: count if count is not None else None]
    if not selected:
        raise ValueError("task selection is empty")
    return selected


def task_manifest(tasks: list[dict[str, Any]], source_path: Path) -> dict[str, Any]:
    return {
        "source": str(source_path),
        "sha256": sha256_file(source_path),
        "count": len(tasks),
        "task_ids": [str(task["task_id"]) for task in tasks],
        "surface_combinations": {
            "+".join(sorted(required_surfaces(task))): sum(
                1 for item in tasks if required_surfaces(item) == required_surfaces(task)
            )
            for task in tasks
        },
    }
