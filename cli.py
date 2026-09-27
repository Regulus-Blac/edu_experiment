"""Command line entry point for the Edu experiment orchestrator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config, repository_root
from .report import write_report
from .runner import run_ablation, run_screen
from .tasks import load_tasks, select_tasks
from .workspace import prepare_workspace


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"path must be relative to repository root: {value}")
    return root / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m edu_experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="import Edu data and build/freeze a DataMind profile")
    init.add_argument("--settings", default="edu_experiment/settings.json")
    init.add_argument("--clean", action="store_true", help="replace exactly the configured DataMind profile")
    init.add_argument("--reindex", action="store_true", help="build the KB embedding index")

    screen = sub.add_parser("screen", help="run one FULL routing query per selected task")
    screen.add_argument("--settings", default="edu_experiment/settings.json")
    screen.add_argument("--tasks", default="edu_sample/tasks/tasks_big.jsonl")
    screen.add_argument("--count", type=int)
    screen.add_argument("--task-ids", help="comma-separated task IDs")
    screen.add_argument("--run-id")
    screen.add_argument("--retries", type=int, default=1)
    screen.add_argument("--dry-run", action="store_true")
    screen.add_argument("--resume", action="store_true")
    screen.add_argument("--prepare-workspace", action="store_true")
    screen.add_argument("--clean-workspace", action="store_true")
    screen.add_argument("--reindex", action="store_true")

    cont = sub.add_parser("continue", help="execute the selected task-level ablation plan")
    cont.add_argument("--settings", default="edu_experiment/settings.json")
    cont.add_argument("--run-id", required=True, help="screen run ID under the configured output directory")
    cont.add_argument("--new-run-id", help="create a new config branch when switching API settings")
    cont.add_argument("--plan", required=True)
    cont.add_argument("--retries", type=int, default=1)

    report = sub.add_parser("report", help="generate final JSON/Markdown/CSV reports without API calls")
    report.add_argument("--settings", default="edu_experiment/settings.json")
    report.add_argument("--run-id")
    report.add_argument("--run-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repository_root()
    try:
        if args.command == "init":
            config = load_config(args.settings)
            manifest = prepare_workspace(config, clean=args.clean, reindex=args.reindex)
            print(json.dumps({
                "workspace_manifest": str(config.output_dir.parent / "workspace_manifest.json"),
                "source_sha256": manifest["source"]["sha256"],
                "profile": config.profile,
            }, ensure_ascii=False, indent=2))
            return 0

        if args.command == "screen":
            config = load_config(args.settings)
            tasks_path = _relative(root, args.tasks).resolve()
            tasks = load_tasks(tasks_path)
            ids = args.task_ids.split(",") if args.task_ids else None
            selected = select_tasks(tasks, count=args.count, task_ids=ids)
            if args.prepare_workspace:
                prepare_workspace(config, clean=args.clean_workspace, reindex=args.reindex)
            run_dir = run_screen(config, tasks_path, selected, run_id=args.run_id, retries=args.retries, dry_run=args.dry_run, resume=args.resume)
            print(json.dumps({"run_dir": str(run_dir), "tasks": len(selected), "dry_run": args.dry_run}, ensure_ascii=False, indent=2))
            return 0

        if args.command == "continue":
            config = load_config(args.settings)
            run_dir = config.output_dir / args.run_id
            plan_path = _relative(root, args.plan).resolve()
            result = run_ablation(config, run_dir, plan_path, retries=args.retries, new_run_id=args.new_run_id)
            report_paths = write_report(result)
            print(json.dumps({
                "run_dir": str(result),
                "phase": "ablation",
                "final_report": str(report_paths[0]),
                "human_review": str(report_paths[2]),
            }, ensure_ascii=False, indent=2))
            return 0

        if args.command == "report":
            if args.run_dir:
                run_dir = _relative(root, args.run_dir).resolve()
            else:
                config = load_config(args.settings)
                if not args.run_id:
                    raise ValueError("report requires --run-id or --run-dir")
                run_dir = config.output_dir / args.run_id
            paths = write_report(run_dir)
            print(json.dumps({"final_report": str(paths[0]), "markdown": str(paths[1]), "human_review": str(paths[2])}, ensure_ascii=False, indent=2))
            return 0
    except (OSError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
