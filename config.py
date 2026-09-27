"""Configuration and repository path handling for ``edu_experiment``.

The underlying DataMind runtime intentionally keeps its own environment based
settings.  This module is the adapter from the user-facing JSON settings file
to that runtime without writing secrets to artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SURFACES = frozenset({"kb", "db", "graph"})
PLACEHOLDERS = {
    "",
    "changeme",
    "fill-in-api-key",
    "填入推理 api key",
    "填入 embedding api key",
    "your-api-key",
}


def repository_root() -> Path:
    """Return the checkout root independently of the caller's cwd."""
    return Path(__file__).resolve().parents[1]


def _relative_path(root: Path, value: str, field: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{field} must be relative to repository root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes repository root: {value!r}") from exc
    return resolved


@dataclass(frozen=True)
class ExperimentConfig:
    root: Path
    datamind: Path
    edu_sample: Path
    output_dir: Path
    llm: dict[str, Any]
    embedding: dict[str, Any]
    data: dict[str, Any]
    raw: dict[str, Any]

    @property
    def profile(self) -> str:
        return str(self.data.get("profile", "edu_registrar"))

    @property
    def settings_fingerprint(self) -> str:
        """Fingerprint non-secret settings suitable for public manifests."""
        public = {
            "paths": {
                "datamind": str(self.datamind.relative_to(self.root)),
                "edu_sample": str(self.edu_sample.relative_to(self.root)),
                "output_dir": str(self.output_dir.relative_to(self.root)),
            },
            "llm": {
                k: v for k, v in self.llm.items() if k != "api_key"
            },
            "embedding": {
                k: v for k, v in self.embedding.items() if k != "api_key"
            },
            "data": self.data,
        }
        payload = json.dumps(public, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def secret_fingerprint(self) -> str:
        payload = json.dumps({
            "llm": _secret_fingerprint(self.llm.get("api_key")),
            "embedding": _secret_fingerprint(self.embedding.get("api_key")),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def public_manifest(self) -> dict[str, Any]:
        """Return safe settings suitable for run manifests and reports."""
        return {
            "paths": {
                "datamind": str(self.datamind.relative_to(self.root)),
                "edu_sample": str(self.edu_sample.relative_to(self.root)),
                "output_dir": str(self.output_dir.relative_to(self.root)),
            },
            "llm": {
                k: v for k, v in self.llm.items() if k != "api_key"
            },
            "embedding": {
                k: v for k, v in self.embedding.items() if k != "api_key"
            },
            "data": self.data,
            "settings_fingerprint": self.settings_fingerprint,
        }


def _secret_fingerprint(value: Any) -> str:
    raw = str(value or "").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read settings JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("settings JSON must contain an object")
    return value


def load_config(path: str | Path, *, root: Path | None = None) -> ExperimentConfig:
    root = (root or repository_root()).resolve()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    raw = _load_object(config_path.resolve())
    paths = raw.get("paths") or {}
    llm = raw.get("llm") or {}
    embedding = raw.get("embedding") or {}
    data = raw.get("data") or {}
    if not isinstance(paths, dict) or not isinstance(llm, dict) or not isinstance(embedding, dict):
        raise ValueError("paths, llm and embedding must be objects")
    for field in ("api_base", "api_key", "model"):
        if not str(llm.get(field, "")).strip():
            raise ValueError(f"llm.{field} is required")
    for field in ("provider", "api_base", "api_key", "model"):
        if not str(embedding.get(field, "")).strip():
            raise ValueError(f"embedding.{field} is required")
    dm_value = paths.get("datamind") or _find_datamind(root)
    sample_value = paths.get("edu_sample") or "edu_sample"
    output_value = paths.get("output_dir") or "edu_experiment/runs"
    datamind = _relative_path(root, str(dm_value), "paths.datamind")
    edu_sample = _relative_path(root, str(sample_value), "paths.edu_sample")
    output_dir = _relative_path(root, str(output_value), "paths.output_dir")
    if not datamind.is_dir():
        raise FileNotFoundError(f"DataMind directory not found: {datamind}")
    if not edu_sample.is_dir():
        raise FileNotFoundError(f"edu_sample directory not found: {edu_sample}")
    protocol = str(llm.get("protocol", "openai_chat_completions"))
    if protocol not in {"openai_chat_completions", "anthropic"}:
        raise ValueError(f"unsupported llm.protocol: {protocol}")
    if str(embedding.get("provider")) != "openai":
        raise ValueError("embedding.provider must be 'openai' for this experiment")
    return ExperimentConfig(
        root=root,
        datamind=datamind,
        edu_sample=edu_sample,
        output_dir=output_dir,
        llm=dict(llm, protocol=protocol),
        embedding=dict(embedding),
        data=dict(data, profile=str(data.get("profile", "edu_registrar"))),
        raw=raw,
    )


def _find_datamind(root: Path) -> str:
    for name in ("DataMind", "datamind"):
        if (root / name).is_dir():
            return name
    raise FileNotFoundError("no sibling DataMind or datamind directory found")


def validate_credentials(config: ExperimentConfig) -> None:
    for label, section in (("llm", config.llm), ("embedding", config.embedding)):
        key = str(section.get("api_key", "")).strip()
        if key.lower() in {item.lower() for item in PLACEHOLDERS} or "填入" in key:
            raise ValueError(f"{label}.api_key is missing or still a placeholder")


def apply_datamind_environment(config: ExperimentConfig) -> None:
    """Map JSON values to DataMind's process-local nested environment."""
    validate_credentials(config)
    values = {
        "DATAMIND__LLM__API_BASE": config.llm["api_base"],
        "DATAMIND__LLM__API_KEY": config.llm["api_key"],
        "DATAMIND__LLM__PROTOCOL": config.llm["protocol"],
        "DATAMIND__LLM__MODEL": config.llm["model"],
        "DATAMIND__EMBEDDING__PROVIDER": config.embedding["provider"],
        "DATAMIND__EMBEDDING__API_BASE": config.embedding["api_base"],
        "DATAMIND__EMBEDDING__API_KEY": config.embedding["api_key"],
        "DATAMIND__EMBEDDING__MODEL": config.embedding["model"],
        "DATAMIND__DATA__PROFILE": config.profile,
    }
    for key, value in values.items():
        os.environ[key] = str(value)


def import_datamind(config: ExperimentConfig) -> None:
    """Make the sibling DataMind package importable for this process."""
    import sys

    value = str(config.datamind)
    if value not in sys.path:
        sys.path.insert(0, value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
