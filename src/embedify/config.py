"""YAML + env configuration.

Canonical config (``embedify.yaml``)::

    embedders:
      text: {backend: local, model: BAAI/bge-small-en-v1.5}
      code: {backend: endpoint, base_url: http://localhost:11434/v1,
             model: nomic-embed-text, api_key_env: null}
    chunkers:
      text: {max_tokens: 512, overlap: 64}
    sink: {type: lancedb, path: ./embedify.lancedb, table: chunks}
    shared_space: false

Env overrides: ``EMBEDIFY_EMBEDDERS__TEXT__MODEL=...`` (double underscore nesting),
``EMBEDIFY_SINK__TYPE=file`` etc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from embedify.errors import ConfigError

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
FAST_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MULTILINGUAL_MODEL = "intfloat/multilingual-e5-small"


@dataclass
class EmbedderConfig:
    backend: str = "local"  # local | endpoint | hash
    model: str = DEFAULT_LOCAL_MODEL
    base_url: str | None = None
    api_key_env: str | None = "OPENAI_API_KEY"
    model_version: str = "unknown"
    query_prefix: str | None = None  # None = adapter default for known models
    document_prefix: str | None = None
    batch_size: int = 32
    dim: int | None = None  # required for hash / endpoints that don't report it

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EmbedderConfig:
        if not data:
            return cls()
        unknown = set(data) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ConfigError(f"Unknown embedder config keys: {sorted(unknown)}")
        return cls(**data)


@dataclass
class ChunkerConfig:
    strategy: str | None = None  # token | ast | markdown | serialize | native | window
    max_tokens: int = 512
    overlap: int = 64
    # tabular (Phase 2)
    granularity: str = "row"  # row | column | table
    rows_per_chunk: int = 20
    # time series (Phase 2)
    window_size: int = 64
    stride: int = 32
    normalization: str = "zscore"  # zscore | minmax | none
    normalization_scope: str = "series"  # series (default) | window
    # tabular serialization (row strategy)
    include_columns: list[str] | None = None
    exclude_columns: list[str] | None = None
    max_value_length: int = 200

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *,
                  base: ChunkerConfig | None = None) -> ChunkerConfig:
        cfg = base or cls()
        if not data:
            return cfg
        for k, v in data.items():
            if k not in cls.__dataclass_fields__:
                raise ConfigError(f"Unknown chunker config key: {k}")
            setattr(cfg, k, v)
        return cfg


@dataclass
class SinkConfig:
    type: str = "memory"  # memory | file | lancedb | pgvector
    path: str | None = None
    table: str = "chunks"
    connection: str | None = None  # pgvector DSN (Phase 4)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SinkConfig:
        if not data:
            return cls()
        unknown = set(data) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ConfigError(f"Unknown sink config keys: {sorted(unknown)}")
        return cls(**data)


@dataclass
class ArchiveConfig:
    """Safety limits for archive sources (see docs/decisions/0005)."""

    max_files: int = 10_000
    max_member_bytes: int = 100 * 1024 * 1024
    max_total_bytes: int = 1024 * 1024 * 1024
    max_nesting_depth: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ArchiveConfig:
        if not data:
            return cls()
        unknown = set(data) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ConfigError(f"Unknown archive config keys: {sorted(unknown)}")
        return cls(**data)


@dataclass
class EmbedifyConfig:
    embedders: dict[str, EmbedderConfig] = field(default_factory=dict)
    chunkers: dict[str, ChunkerConfig] = field(default_factory=dict)
    sink: SinkConfig = field(default_factory=SinkConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    shared_space: bool = False
    shared_model: str | None = None

    def embedder_for(self, modality: str) -> EmbedderConfig:
        # document falls back to text, code falls back to text
        for key in (modality, "text", "default"):
            if key in self.embedders:
                return self.embedders[key]
        return EmbedderConfig()

    def chunker_for(self, modality: str) -> ChunkerConfig:
        return self.chunkers.get(modality) or self.chunkers.get("text") or ChunkerConfig()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EmbedifyConfig:
        data = data or {}
        embedders = {
            k: EmbedderConfig.from_dict(v) for k, v in (data.get("embedders") or {}).items()
        }
        chunkers = {k: ChunkerConfig.from_dict(v) for k, v in (data.get("chunkers") or {}).items()}
        return cls(
            embedders=embedders,
            chunkers=chunkers,
            sink=SinkConfig.from_dict(data.get("sink")),
            archive=ArchiveConfig.from_dict(data.get("archive")),
            shared_space=bool(data.get("shared_space", False)),
            shared_model=data.get("shared_model"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> EmbedifyConfig:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Config file not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        cfg = cls.from_dict(data if isinstance(data, dict) else {})
        cfg.apply_env_overrides()
        return cfg

    @classmethod
    def load(cls, path: str | Path | None = None) -> EmbedifyConfig:
        if path:
            return cls.from_yaml(path)
        for candidate in ("embedify.yaml", "embedify.yml"):
            if Path(candidate).exists():
                return cls.from_yaml(candidate)
        cfg = cls()
        cfg.apply_env_overrides()
        return cfg

    def apply_env_overrides(self) -> None:
        """Apply EMBEDIFY_SECTION__KEY=value overrides in place."""
        for env_key, value in os.environ.items():
            if not env_key.startswith("EMBEDIFY_"):
                continue
            parts = env_key[len("EMBEDIFY_"):].lower().split("__")
            try:
                if parts[0] == "sink" and len(parts) == 2:
                    setattr(self.sink, parts[1], _coerce(value))
                elif parts[0] == "embedders" and len(parts) == 3:
                    ecfg = self.embedders.setdefault(parts[1], EmbedderConfig())
                    setattr(ecfg, parts[2], _coerce(value))
                elif parts[0] == "shared_space":
                    self.shared_space = bool(_coerce(value))
            except AttributeError as exc:
                raise ConfigError(f"Invalid env override {env_key}: {exc}") from exc


def _coerce(value: str) -> Any:
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        return value
