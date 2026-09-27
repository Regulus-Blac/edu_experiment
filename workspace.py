"""Workspace import and DataMind build/freeze integration."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, apply_datamind_environment, import_datamind, sha256_file, sha256_json


def _source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory in ("kb_docs", "tables", "graph", "sources", "tasks"):
        base = root / directory
        if not base.is_dir():
            continue
        files.extend(
            path for path in sorted(base.rglob("*"))
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
        )
    return files


def source_manifest(config: ExperimentConfig) -> dict[str, Any]:
    files = [
        {
            "path": str(path.relative_to(config.edu_sample)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in _source_files(config.edu_sample)
    ]
    return {
        "schema_version": 1,
        "profile": config.profile,
        "source_root": str(config.edu_sample.relative_to(config.root)),
        "files": files,
        "sha256": sha256_json(files),
    }


def workspace_manifest_path(config: ExperimentConfig) -> Path:
    return config.output_dir.parent / "workspace_manifest.json"


def load_workspace_manifest(config: ExperimentConfig) -> dict[str, Any]:
    path = workspace_manifest_path(config)
    if not path.is_file():
        raise FileNotFoundError(
            f"workspace manifest not found: {path}; run `python -m edu_experiment init` first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _run_import(config: ExperimentConfig, *, clean: bool) -> None:
    importer = config.edu_sample / "import_edu.py"
    if not importer.is_file():
        raise FileNotFoundError(f"Edu importer not found: {importer}")
    command = [
        sys.executable,
        str(importer),
        "--src",
        str(config.edu_sample),
        "--datamind",
        str(config.datamind),
        "--profile",
        config.profile,
    ]
    if clean:
        command.append("--clean")
    subprocess.run(command, cwd=config.root, check=True)


def _run_static_audit(config: ExperimentConfig) -> dict[str, Any]:
    checker = config.edu_sample / "check_static.py"
    if not checker.is_file():
        return {"status": "not_available"}
    completed = subprocess.run(
        [sys.executable, str(checker), "--root", str(config.edu_sample)],
        cwd=config.root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"static leakage audit failed:\n{completed.stdout[-4000:]}\n{completed.stderr[-1000:]}")
    return {"status": "pass", "output": completed.stdout[-4000:]}


async def _build_workspace(config: ExperimentConfig, *, reindex: bool) -> dict[str, Any]:
    import_datamind(config)
    from datamind.agent import build_datamind
    from datamind.config import Settings

    settings = Settings()
    system = await build_datamind(settings, enable={"kb", "db", "graph"})
    try:
        ingest = system.services.ingest
        if ingest is None:
            raise RuntimeError("DataMind ingest service is unavailable")
        build = await ingest.build_start(path=str(config.edu_sample))
        indexed = None
        if reindex:
            tool = system.store.tools.get("kb_reindex")
            if tool is None:
                raise RuntimeError("DataMind kb_reindex tool is unavailable")
            indexed = await tool.handler()
        warmup = await system.warmup()
        frozen = await ingest.build_freeze(build_id=str(build["build_id"]))
        verified = await ingest.build_verify(build_id=str(build["build_id"]))
        if not verified.get("ok"):
            raise RuntimeError(f"workspace verification failed: {verified}")
        return {
            "build": build,
            "indexed": indexed,
            "warmup": warmup,
            "frozen": frozen,
            "verified": verified,
            "profile": config.profile,
            "data_dir": str(settings.data.data_dir.relative_to(config.root)),
            "storage_dir": str(settings.data.storage_dir.relative_to(config.root)),
        }
    finally:
        await system.aclose()


def prepare_workspace(
    config: ExperimentConfig,
    *,
    clean: bool = False,
    reindex: bool = False,
) -> dict[str, Any]:
    """Import Edu data, build/freeze the profile, and record a safe manifest."""
    apply_datamind_environment(config)
    audit = _run_static_audit(config)
    _run_import(config, clean=clean)
    runtime = asyncio.run(_build_workspace(config, reindex=reindex))
    manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source_manifest(config),
        "static_audit": audit,
        "runtime": runtime,
        "settings_fingerprint": config.settings_fingerprint,
    }
    path = workspace_manifest_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
