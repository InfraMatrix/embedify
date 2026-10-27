"""Sinks: where EmbeddingRecords go.

Phase 1: in-memory, file (JSONL / Parquet), and a basic LanceDB sink so
``embedify embed`` followed (in a separate run) by ``embedify query`` works
end to end. pgvector arrives in Phase 4 behind the same interface.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

from embedify.config import SinkConfig
from embedify.errors import SinkError
from embedify.types import EmbeddingRecord


class BaseSink(Protocol):
    def write(self, records: Iterable[EmbeddingRecord]) -> None: ...
    def finalize(self) -> None: ...


class MemorySink:
    def __init__(self) -> None:
        self.records: list[EmbeddingRecord] = []

    def write(self, records: Iterable[EmbeddingRecord]) -> None:
        self.records.extend(records)

    def finalize(self) -> None:
        pass

    # Incremental support (changed files replace, deleted files removed)
    def stored_source_hashes(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for r in self.records:
            out.setdefault(r.source_uri, set()).add(r.content_hash)
        return out

    def delete_source(self, source_uri: str) -> int:
        before = len(self.records)
        self.records = [r for r in self.records if r.source_uri != source_uri]
        return before - len(self.records)


class FileSink:
    """``.jsonl`` anywhere; ``.parquet`` with ``embedify[parquet]``; ``.npz`` vectors+sidecar."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._buffer: list[EmbeddingRecord] = []
        self._fh: Any = None

    def write(self, records: Iterable[EmbeddingRecord]) -> None:
        if self.path.suffix.lower() == ".jsonl":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self._fh is None:
                self._fh = self.path.open("a", encoding="utf-8")
            for r in records:
                self._fh.write(json.dumps(r.to_dict()) + "\n")
            self._fh.flush()
        else:
            self._buffer.extend(records)

    def finalize(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            return
        suffix = self.path.suffix.lower()
        if suffix == ".jsonl":
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".parquet":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise SinkError(
                    "Parquet sink needs pyarrow: pip install 'embedify[parquet]'"
                ) from exc
            flat = []
            for r in self._buffer:
                d = r.to_dict()
                flat.append({**d, "locator": json.dumps(d["locator"]),
                             "metadata": json.dumps(d["metadata"])})
            pq.write_table(pa.table(flat) if flat else pa.table({}), self.path)
        elif suffix == ".npz":
            try:
                import numpy as np
            except ImportError as exc:
                raise SinkError("NPZ sink needs numpy") from exc
            vectors = (
                np.array([r.vector for r in self._buffer], dtype="float32")
                if self._buffer else np.zeros((0, 0), dtype="float32")
            )
            np.savez_compressed(self.path, vectors=vectors)
            with self.path.with_suffix(".jsonl").open("w", encoding="utf-8") as fh:
                for r in self._buffer:
                    fh.write(json.dumps({**r.to_dict(), "vector": []}) + "\n")
        else:
            raise SinkError(f"Unsupported file sink suffix '{suffix}' (use .jsonl/.parquet/.npz)")
        self._buffer = []


def _record_to_row(r: EmbeddingRecord) -> dict[str, Any]:
    d = r.to_dict()
    d["locator"] = json.dumps(d["locator"])
    d["metadata"] = json.dumps(d["metadata"])
    return d


def _row_to_record(row: dict[str, Any]) -> EmbeddingRecord:
    data = dict(row)
    for key in ("locator", "metadata"):
        if isinstance(data.get(key), str):
            data[key] = json.loads(data[key])
    data.pop("_distance", None)
    data.pop("score", None)
    if data.get("created_at") is not None and not isinstance(data["created_at"], str):
        data["created_at"] = data["created_at"].isoformat()
    return EmbeddingRecord.from_dict(data)


class LanceDBSink:
    """Basic LanceDB sink (Phase 1): append-or-create, vector search for query."""

    def __init__(self, path: str | Path, table: str = "chunks", *, mode: str = "append") -> None:
        self.path = str(path)
        self.table_name = table
        self.mode = mode
        self._buffer: list[EmbeddingRecord] = []
        self._db: Any = None

    def _connect(self) -> Any:
        try:
            import lancedb
        except ImportError as exc:
            raise SinkError("LanceDB sink needs: pip install 'embedify[lancedb]'") from exc
        if self._db is None:
            self._db = lancedb.connect(self.path)
        return self._db

    def _table_names(self, db: Any) -> list[str]:
        # Newer lancedb: list_tables() -> ListTablesResponse with .tables: list[str]
        # Older lancedb: table_names() -> list[str]
        list_tables = getattr(db, "list_tables", None)
        if callable(list_tables):
            result = list_tables()
            tables = getattr(result, "tables", result)
            return [str(t) for t in tables]
        return [str(t) for t in db.table_names()]  # older lancedb

    def table_info(self) -> dict[str, Any] | None:
        """Stored {model, model_version, dim} of an existing table, or None."""
        if not Path(self.path).exists():
            return None
        db = self._connect()
        if self.table_name not in self._table_names(db):
            return None
        table = db.open_table(self.table_name)
        row: dict[str, Any] | None = None
        try:
            head = table.head(1)
            rows = head.to_pylist() if hasattr(head, "to_pylist") else list(head)
            row = rows[0] if rows else None
        except Exception:
            try:
                arrow = table.to_arrow()
                rows = arrow.to_pylist()
                row = rows[0] if rows else None
            except Exception:
                row = None
        if not row:
            return None
        return {"model": row.get("model"), "model_version": row.get("model_version"),
                "dim": int(row.get("dim") or len(row.get("vector") or []))}

    # Incremental support (changed files replace, deleted files removed)
    def stored_source_hashes(self) -> dict[str, set[str]]:
        if not Path(self.path).exists():
            return {}
        db = self._connect()
        if self.table_name not in self._table_names(db):
            return {}
        try:
            rows = db.open_table(self.table_name).to_arrow().to_pylist()
        except Exception:
            return {}
        out: dict[str, set[str]] = {}
        for row in rows:
            if row.get("source_uri") is not None:
                out.setdefault(str(row["source_uri"]), set()).add(str(row.get("content_hash")))
        return out

    def delete_source(self, source_uri: str) -> int:
        if not Path(self.path).exists():
            return 0
        db = self._connect()
        if self.table_name not in self._table_names(db):
            return 0
        before = len(self.stored_source_hashes().get(source_uri, set()))
        escaped = source_uri.replace("'", "''")
        db.open_table(self.table_name).delete(f"source_uri = '{escaped}'")
        return before

    def write(self, records: Iterable[EmbeddingRecord]) -> None:
        self._buffer.extend(records)

    def finalize(self) -> None:
        if not self._buffer:
            return
        rows = [_record_to_row(r) for r in self._buffer]
        db = self._connect()
        exists = self.table_name in self._table_names(db)
        if exists and self.mode == "append":
            # Model/dim guard: never mix incomparable vectors in one table.
            info = self.table_info()
            first = self._buffer[0]
            if info and (info["model"] != first.model or info["dim"] != first.dim):
                self._buffer = []
                raise SinkError(
                    f"Refusing to mix models in LanceDB table '{self.table_name}': "
                    f"stored model={info['model']} dim={info['dim']}, "
                    f"new model={first.model} dim={first.dim}. "
                    f"Use a separate table (e.g. table='{first.model}') — vectors from "
                    "different models/dims are not comparable."
                )
            db.open_table(self.table_name).add(rows)
        elif exists and self.mode == "overwrite":
            db.create_table(self.table_name, data=rows, mode="overwrite")
        else:
            db.create_table(self.table_name, data=rows, mode="create")
        self._buffer = []

    def search(self, vector: Sequence[float], *, top_k: int = 5,
               model: str | None = None) -> list[tuple[EmbeddingRecord, float]]:
        """Cosine search. Returns (record, score) where score is cosine
        similarity in [-1, 1] — higher is better."""
        if not Path(self.path).exists():
            raise SinkError(
                f"LanceDB database not found at {self.path}. "
                f"Run `embedify embed <source> --sink lancedb --output {self.path}` first."
            )
        db = self._connect()
        if self.table_name not in self._table_names(db):
            raise SinkError(f"No table '{self.table_name}' in LanceDB at {self.path}")
        info = self.table_info()
        if info and info["dim"] and len(vector) != info["dim"]:
            raise SinkError(
                f"Query vector dim {len(vector)} does not match stored dim {info['dim']} "
                f"(model={info['model']}). Query with the same model that indexed this table."
            )
        query = db.open_table(self.table_name).search(list(vector))
        metric = getattr(query, "metric", None)
        if callable(metric):
            query = metric("cosine")
        elif hasattr(query, "distance_type"):
            query = query.distance_type("cosine")
        rows = query.limit(top_k).to_list()
        results: list[tuple[EmbeddingRecord, float]] = []
        for row in rows:
            record = _row_to_record(row)
            if model and record.model != model:
                continue
            # LanceDB cosine distance = 1 - cosine similarity
            score = 1.0 - float(row.get("_distance", 0.0))
            results.append((record, score))
        results.sort(key=lambda item: item[1], reverse=True)
        return results


def create_sink(config: SinkConfig) -> BaseSink:
    if config.type == "memory":
        return MemorySink()
    if config.type == "file":
        if not config.path:
            raise SinkError("file sink requires sink.path")
        return FileSink(config.path)
    if config.type == "lancedb":
        return LanceDBSink(config.path or "./embedify.lancedb", config.table)
    if config.type == "pgvector":
        raise NotImplementedError("pgvector sink lands in Phase 4; use lancedb/file for now")
    raise SinkError(f"Unknown sink type '{config.type}'")
