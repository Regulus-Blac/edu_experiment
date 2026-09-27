"""Offline report generation for screen and ablation runs."""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .runner import TaskPlan, _read_json, resolve_plan
from .tasks import load_tasks


TOKEN = re.compile(r"\d+/\d+|\d+(?:\.\d+)?")
WORD = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}")


def _normal(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", (text or "").lower())


def _numbers(text: str) -> set[str]:
    return {item.rstrip("0").rstrip(".") if "." in item else item for item in TOKEN.findall(text or "")}


def answer_match(reference: str, answer: str) -> dict[str, Any]:
    ref = _normal(reference)
    pred = _normal(answer)
    if not pred:
        return {"correct": False, "method": "empty", "similarity": 0.0, "numeric_recall": 0.0}
    if ref == pred or ref in pred:
        return {"correct": True, "method": "normalized_exact", "similarity": 1.0, "numeric_recall": 1.0}
    ref_numbers = _numbers(reference)
    pred_numbers = _numbers(answer)
    numeric_recall = len(ref_numbers & pred_numbers) / len(ref_numbers) if ref_numbers else 1.0
    similarity = SequenceMatcher(None, ref, pred).ratio()
    ref_words = set(WORD.findall(reference))
    pred_words = set(WORD.findall(answer))
    word_recall = len(ref_words & pred_words) / len(ref_words) if ref_words else 1.0
    correct = numeric_recall == 1.0 and word_recall >= 0.55 and similarity >= 0.35
    return {
        "correct": correct,
        "method": "numeric_and_text" if correct else "not_match",
        "similarity": round(similarity, 4),
        "numeric_recall": round(numeric_recall, 4),
        "word_recall": round(word_recall, 4),
    }


def _rows(run_dir: Path) -> list[dict[str, Any]]:
    paths = []
    screen = run_dir / "screen" / "results.jsonl"
    if screen.is_file():
        paths.append(screen)
    paths.extend(sorted((run_dir / "ablation").rglob("*.jsonl")) if (run_dir / "ablation").is_dir() else [])
    result: list[dict[str, Any]] = []
    for path in paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                row["_artifact"] = str(path)
                row["_line"] = line_no
                result.append(row)
    return result


def _tokens(row: dict[str, Any]) -> tuple[int, int]:
    usage = row.get("usage") or {}
    input_count = usage.get("total_input_tokens", usage.get("input_tokens", usage.get("prompt_tokens", 0))) or 0
    output_count = usage.get("total_output_tokens", usage.get("output_tokens", usage.get("completion_tokens", 0))) or 0
    return int(input_count), int(output_count)


def _task_categories(run_dir: Path, tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    categories = {str(task["task_id"]): {"category": "screen_only", "completion_state": "screen_only", "mode": "none"} for task in tasks}
    plan_path = run_dir / "ablation_manifest.json"
    if not plan_path.is_file():
        return categories
    ablation = _read_json(plan_path)
    plan_file = Path(str(ablation["plan"]))
    if not plan_file.is_absolute():
        plan_file = run_dir.parents[2] / plan_file
    if not plan_file.is_file():
        return categories
    plans = resolve_plan(tasks, _read_json(plan_file))
    for item in plans:
        task_id = str(item.task["task_id"])
        if item.mode == "none":
            continue
        complete = True
        failed = False
        for condition in item.conditions:
            for replicate in range(1, condition.replicates + 1):
                status = run_dir / "ablation" / task_id / condition.name / f"rep{replicate:02d}.status.json"
                if not status.is_file() or _read_json(status).get("status") != "complete":
                    complete = False
                    continue
                rows_path = status.with_name(status.name.replace(".status.json", ".jsonl"))
                rows = []
                if rows_path.is_file():
                    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if any(row.get("error") for row in rows):
                    failed = True
        categories[task_id] = {
            "category": "fully_ablated" if item.mode == "full" and complete and not failed else "partially_ablated",
            "completion_state": "complete" if complete and not failed else ("failed" if failed else "incomplete"),
            "mode": item.mode,
        }
    return categories


def _score(row: dict[str, Any], category: dict[str, Any]) -> dict[str, Any]:
    answer = answer_match(str(row.get("reference_answer", "")), str(row.get("answer", "")))
    required = set(row.get("required_surfaces") or (row.get("metadata") or {}).get("required_surfaces", []))
    used = set(row.get("surfaces_used", []))
    if row.get("error"):
        classification = "runner_error"
    elif not answer["correct"]:
        classification = "model_failure"
    elif "graph" in required and "graph" not in used:
        classification = "graph_bypass_candidate"
    elif required - used:
        classification = "surface_bypass_candidate"
    else:
        classification = "valid_dependency"
    input_tokens, output_tokens = _tokens(row)
    return {
        "task_id": str(row.get("task_id")),
        "condition": row.get("condition"),
        "replicate": row.get("replicate"),
        "category": category["category"],
        "completion_state": category["completion_state"],
        "mode": category["mode"],
        "active_surfaces": sorted(row.get("active_surfaces", [])),
        "required_surfaces": sorted(required),
        "surfaces_used": sorted(used),
        "graph_skipped": "graph" in required and "graph" not in used,
        "classification": classification,
        "answer": answer,
        "error": row.get("error"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_s": row.get("latency_s"),
        "artifact": row.get("_artifact"),
        "line": row.get("_line"),
    }


def _group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    non_errors = [row for row in rows if not row["error"]]
    graph_rows = [row for row in non_errors if "graph" in row["required_surfaces"]]
    correct = sum(1 for row in non_errors if row["answer"]["correct"])
    skipped = sum(1 for row in graph_rows if row["graph_skipped"])
    tasks = {row["task_id"] for row in rows}
    replicate_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        replicate_groups[(row["task_id"], str(row.get("condition")))].append(_normal(str(row.get("answer", ""))))
    repeated = [answers for answers in replicate_groups.values() if len(answers) > 1]
    unstable = sum(1 for answers in repeated if len(set(answers)) > 1)
    return {
        "tasks": len(tasks),
        "runs": len(rows),
        "errors": sum(1 for row in rows if row["error"]),
        "answer_correct": correct,
        "answer_accuracy": round(correct / len(non_errors), 4) if non_errors else None,
        "graph_required_runs": len(graph_rows),
        "graph_skipped_runs": skipped,
        "graph_skip_rate": round(skipped / len(graph_rows), 4) if graph_rows else None,
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "replicate_groups": len(repeated),
        "unstable_replicate_groups": unstable,
        "replicate_stability_rate": round(1 - unstable / len(repeated), 4) if repeated else None,
    }


def build_report(run_dir: Path) -> dict[str, Any]:
    run_manifest = _read_json(run_dir / "run_manifest.json")
    task_source = Path(run_manifest["task_manifest"]["source"])
    if not task_source.is_absolute():
        task_source = run_dir.parents[2] / task_source
    tasks = [task for task in load_tasks(task_source) if str(task["task_id"]) in set(run_manifest["task_manifest"]["task_ids"])]
    categories = _task_categories(run_dir, tasks)
    scored = [_score(row, categories.get(str(row.get("task_id")), {"category": "screen_only", "completion_state": "screen_only", "mode": "none"})) for row in _rows(run_dir)]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_category[row["category"]].append(row)
    task_groups = {
        category: sorted(task_id for task_id, value in categories.items() if value["category"] == category)
        for category in ("fully_ablated", "partially_ablated", "screen_only")
    }
    review: list[dict[str, Any]] = []
    for row in scored:
        if row["classification"] in {"graph_bypass_candidate", "surface_bypass_candidate", "runner_error", "model_failure"}:
            review.append({
                "task_id": row["task_id"],
                "category": row["category"],
                "condition": row["condition"],
                "classification": row["classification"],
                "graph_exclusive_fact": "",
                "other_surface_duplicate": "",
                "manual_decision": "",
                "reviewer": "",
                "notes": "",
            })
    return {
        "schema_version": 1,
        "run_id": run_manifest["run_id"],
        "phase": "screen_and_ablation",
        "task_groups": task_groups,
        "group_stats": {category: _group_stats(rows) for category, rows in sorted(by_category.items())},
        "rows": scored,
        "manual_review_tasks": review,
        "generated_at": _read_json(run_dir / "run_manifest.json").get("created_at"),
    }


def write_report(run_dir: Path) -> tuple[Path, Path, Path]:
    report = build_report(run_dir)
    json_path = run_dir / "final_report.json"
    md_path = run_dir / "final_report.md"
    csv_path = run_dir / "human_review_tasks.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# Edu Experiment Report: `{report['run_id']}`",
        "",
        "| Group | Tasks | Runs | Errors | Accuracy | Graph skip rate | Replicate stability | Input tokens | Output tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category in ("fully_ablated", "partially_ablated", "screen_only"):
        stats = report["group_stats"].get(category, _group_stats([]))
        lines.append(
            f"| {category} | {stats['tasks']} | {stats['runs']} | {stats['errors']} | "
            f"{stats['answer_accuracy']} | {stats['graph_skip_rate']} | {stats['replicate_stability_rate']} | "
            f"{stats['input_tokens']} | {stats['output_tokens']} |"
        )
    lines.extend(["", "## Human Review", "", f"Items requiring review: {len(report['manual_review_tasks'])}", ""])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fields = [
        "task_id", "category", "condition", "classification", "graph_exclusive_fact",
        "other_surface_duplicate", "manual_decision", "reviewer", "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["manual_review_tasks"])
    return json_path, md_path, csv_path
