"""Core types shared across embedify.

Every vector can be traced to an exact source span via ``Locator``,
which is what makes the output usable for RAG citations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class Modality(StrEnum):
    TEXT = "text"
    CODE = "code"
    DOCUMENT = "document"
    TABULAR = "tabular"
    TIMESERIES = "timeseries"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"


class InputType(StrEnum):
    """Hint for asymmetric embedding models.

    ``document`` is used when indexing; ``query`` when searching.
    Adapters map this to instruction prefixes (e.g. BGE), Cohere's
    ``input_type`` parameter, etc.
    """

    QUERY = "query"
    DOCUMENT = "document"


@dataclass(frozen=True)
class Locator:
    """Exact source span a unit came from. Only one family is set per modality."""

    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    row_id: str | None = None
    time_start: datetime | None = None
    time_end: datetime | None = None
    frame_ts: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "row_id": self.row_id,
            "time_start": self.time_start.isoformat() if self.time_start else None,
            "time_end": self.time_end.isoformat() if self.time_end else None,
            "frame_ts": self.frame_ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Locator:
        def _dt(v: Any) -> datetime | None:
            return datetime.fromisoformat(v) if isinstance(v, str) else None

        return cls(
            page=data.get("page"),
            line_start=data.get("line_start"),
            line_end=data.get("line_end"),
            char_start=data.get("char_start"),
            char_end=data.get("char_end"),
            row_id=data.get("row_id"),
            time_start=_dt(data.get("time_start")),
            time_end=_dt(data.get("time_end")),
            frame_ts=data.get("frame_ts"),
        )


def content_hash(text: str, *, namespace: str = "") -> str:
    h = hashlib.sha256()
    if namespace:
        h.update(namespace.encode("utf-8"))
        h.update(b"\0")
    h.update(text.encode("utf-8") if isinstance(text, str) else text)
    return h.hexdigest()


PayloadKind = str  # "text" | "numeric" | "bytes"


@dataclass(frozen=True)
class TextInput:
    """Typed payload: natural-language / serialized text."""

    text: str
    kind: str = "text"


@dataclass(frozen=True)
class NumericInput:
    """Typed payload: a numeric matrix (rows x channels)."""

    values: list[list[float]]
    channels: tuple[str, ...] = ()
    kind: str = "numeric"


@dataclass(frozen=True)
class BytesInput:
    """Typed payload: raw bytes (images, audio clips, video frames)."""

    data: bytes
    mime_type: str | None = None
    kind: str = "bytes"


InputPayload = TextInput | NumericInput | BytesInput


@dataclass
class Unit:
    """One embeddable piece of a source, streamed through the pipeline."""

    content: str
    modality: Modality
    source_uri: str
    locator: Locator = field(default_factory=Locator)
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    # Numeric payload for time-series windows / tabular rows.
    numeric_values: list[list[float]] | None = None
    # Raw-bytes payload for images / audio / video frames (Phase 3).
    content_bytes: bytes | None = None
    mime_type: str | None = None

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = content_hash(self.content)

    @property
    def text_preview(self) -> str:
        return self.content[:500] if isinstance(self.content, str) else ""

    def payload_for(self, accepted: set[str]) -> InputPayload:
        """Typed payload chosen by what the embedder accepts.

        A time-series window routed to a text embedder is serialized to
        text; routed to a numeric embedder it stays a matrix. Images stay
        bytes for vision embedders and degrade to their text label otherwise.
        """
        if "numeric" in accepted and self.numeric_values:
            return NumericInput(
                values=self.numeric_values,
                channels=tuple(str(c) for c in self.metadata.get("channels", [])),
            )
        if "bytes" in accepted and self.content_bytes is not None:
            return BytesInput(data=self.content_bytes, mime_type=self.mime_type)
        return TextInput(text=self.content)


SCHEMA_VERSION = "1.0"  # version of the EmbeddingRecord output contract (docs/schema.md)


@dataclass(frozen=True)
class EmbeddingRecord:
    id: str
    vector: list[float]
    model: str
    model_version: str
    dim: int
    modality: Modality
    source_uri: str
    locator: Locator
    content_hash: str
    text_preview: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "vector": list(self.vector),
            "model": self.model,
            "model_version": self.model_version,
            "dim": self.dim,
            "modality": str(self.modality),
            "source_uri": self.source_uri,
            "locator": self.locator.to_dict(),
            "content_hash": self.content_hash,
            "text_preview": self.text_preview,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingRecord:
        created = data.get("created_at")
        return cls(
            id=data["id"],
            vector=[float(x) for x in data["vector"]],
            model=data["model"],
            model_version=data.get("model_version", "unknown"),
            dim=int(data.get("dim", len(data["vector"]))),
            modality=Modality(data.get("modality", "text")),
            source_uri=data["source_uri"],
            locator=Locator.from_dict(data.get("locator") or {}),
            content_hash=data.get("content_hash", ""),
            text_preview=data.get("text_preview", ""),
            metadata=data.get("metadata") or {},
            created_at=datetime.fromisoformat(created)
            if isinstance(created, str)
            else datetime.now(UTC),
            schema_version=str(data.get("schema_version", "1.0")),
        )


@dataclass
class FileError:
    source_uri: str
    error: str

    def to_dict(self) -> dict[str, str]:
        return {"source_uri": self.source_uri, "error": self.error}


@dataclass(frozen=True)
class SecretFinding:
    """A secret detected *before* embedding. Never carries the secret value."""

    source_uri: str
    kind: str  # e.g. "aws-access-key", "private-key", "secret-file"
    line: int | None = None
    action: str = "excluded"  # the containing file is excluded from embedding

    def to_dict(self) -> dict[str, Any]:
        return {"source_uri": self.source_uri, "kind": self.kind,
                "line": self.line, "action": self.action}


@dataclass
class JobStats:
    units_processed: int = 0
    units_embedded: int = 0
    files_scanned: int = 0
    files_processed: int = 0
    files_failed: int = 0
    files_skipped: int = 0  # secret files, secret-bearing files, unchanged (incremental)
    skipped_by_reason: dict[str, int] = field(default_factory=dict)
    secrets_found: int = 0
    records_deleted: int = 0  # incremental replacement/deletion
    tokens_estimated: int = 0
    elapsed_seconds: float = 0.0

    @property
    def units_per_second(self) -> float:
        return self.units_embedded / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


@dataclass
class JobResult:
    records: list[EmbeddingRecord]
    stats: JobStats
    errors: list[FileError] = field(default_factory=list)
    findings: list[SecretFinding] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.records)
