"""Tabular + time-series parsing (Phase 2).

Stdlib-first: CSV/TSV/JSON/JSONL/SQLite need no extras. Parquet needs
``embedify[parquet]`` (pyarrow), Excel needs ``embedify[excel]`` (openpyxl).
Every parser yields row dicts; loaders turn rows into Units and detect
time-series shape (timestamp + numeric columns) automatically.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import Any

from embedify.errors import LoaderError

TIMESTAMP_NAMES = {"time", "timestamp", "date", "datetime", "ts", "t", "epoch", "instant"}


@dataclass
class TableData:
    columns: list[str]
    rows: list[dict[str, Any]]
    table: str = "table"
    extra: dict[str, Any] = field(default_factory=dict)


def _clean(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def iter_tables(path: Path) -> Iterator[TableData]:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh, delimiter=delim)
            rows = [dict(r) for r in reader]
            yield TableData(columns=list(reader.fieldnames or []), rows=rows, table=path.stem)
    elif suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        cols = list(rows[0]) if rows else []
        yield TableData(columns=cols, rows=rows, table=path.stem)
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
            for key, val in data.items():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    yield TableData(columns=list(val[0]), rows=val, table=str(key))
            return
        rows = data if isinstance(data, list) else [data]
        rows = [r for r in rows if isinstance(r, dict)]
        yield TableData(columns=list(rows[0]) if rows else [], rows=rows, table=path.stem)
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise LoaderError("Parquet needs pyarrow: pip install 'embedify[parquet]'") from exc
        table = pq.read_table(path)
        rows = table.to_pylist()
        yield TableData(columns=table.column_names,
                        rows=[{k: _clean(v) for k, v in r.items()} for r in rows],
                        table=path.stem)
    elif suffix in (".xlsx", ".xlsm"):
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise LoaderError("Excel needs openpyxl: pip install 'embedify[excel]'") from exc
        wb = load_workbook(path, read_only=True, data_only=True)
        for ws in wb.worksheets:
            it = ws.iter_rows(values_only=True)
            try:
                header = [str(c) for c in next(it)]
            except StopIteration:
                continue
            rows = [dict(zip(header, row, strict=False)) for row in it]
            yield TableData(columns=header, rows=rows, table=ws.title)
        wb.close()
    elif suffix in (".db", ".sqlite", ".sqlite3"):
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            names = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            for name in names:
                cur = conn.execute(f'SELECT * FROM "{name}"')  # noqa: S608 - local file, own DB
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]
                yield TableData(columns=cols, rows=rows, table=name)
        finally:
            conn.close()
    else:
        raise LoaderError(f"No tabular parser for '{suffix}' ({path})")


def parse_timestamp(value: Any, *, assume_tz: tzinfo | None = UTC) -> datetime | None:
    """Parse a timestamp, ALWAYS returning a timezone-aware datetime.

    Naive inputs (ISO strings without offset, naive ``datetime`` objects)
    are interpreted in ``assume_tz`` (UTC by default — pass another
    tzinfo to override, or ``None`` to leave naive inputs naive).
    Epoch numbers (s/ms) and aware inputs are absolute and returned in UTC
    terms unchanged. Naive datetimes never escape the loader layer.
    """
    def attach(dt: datetime) -> datetime:
        if dt.tzinfo is not None:
            return dt
        return dt.replace(tzinfo=assume_tz) if assume_tz is not None else dt

    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return attach(value)
    if isinstance(value, (int, float)):
        try:
            ts = float(value)
            if ts > 1e12:  # milliseconds
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=UTC)
        except (ValueError, OSError, OverflowError):
            return None
    text = str(value).strip()
    try:
        return attach(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S",
                "%m/%d/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return attach(datetime.strptime(text, fmt))
        except ValueError:
            continue
    return None


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def iter_row_stream(path: Path) -> Iterator[tuple[str, list[str], dict[str, Any]]]:
    """Stream (table, columns, row) lazily — never loads the whole file.

    CSV/TSV/JSONL/SQLite/Parquet/Excel iterate row-by-row (Parquet via
    ``iter_batches``, SQLite via cursor, Excel via read-only rows).
    JSON arrays are inherently whole-document and are the one exception.
    """
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh, delimiter=delim)
            cols = list(reader.fieldnames or [])
            for row in reader:
                yield path.stem, cols, dict(row)
    elif suffix == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    yield path.stem, list(row), row
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise LoaderError("Parquet needs pyarrow: pip install 'embedify[parquet]'") from exc
        pf = pq.ParquetFile(path)
        cols = pf.schema_arrow.names
        for batch in pf.iter_batches(batch_size=512):
            for row in batch.to_pylist():
                yield path.stem, cols, {k: _clean(v) for k, v in row.items()}
    elif suffix in (".xlsx", ".xlsm"):
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise LoaderError("Excel needs openpyxl: pip install 'embedify[excel]'") from exc
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            for ws in wb.worksheets:
                it = ws.iter_rows(values_only=True)
                try:
                    header = [str(c) for c in next(it)]
                except StopIteration:
                    continue
                for row in it:
                    yield ws.title, header, dict(zip(header, row, strict=False))
        finally:
            wb.close()
    elif suffix in (".db", ".sqlite", ".sqlite3"):
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            names = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            for name in names:
                cur = conn.execute(f'SELECT * FROM "{name}"')  # noqa: S608 - user's own local DB
                cols = [d[0] for d in cur.description]
                while True:
                    rows = cur.fetchmany(512)
                    if not rows:
                        break
                    for row in rows:
                        yield name, cols, dict(zip(cols, row, strict=False))
        finally:
            conn.close()
    elif suffix == ".json":
        for table in iter_tables(path):  # whole-document format; small by nature here
            for row in table.rows:
                yield table.table, table.columns, row
    else:
        raise LoaderError(f"No tabular parser for '{suffix}' ({path})")


def detect_time_column(columns: list[str], rows: list[dict[str, Any]]) -> str | None:
    for col in columns:
        named = col.lower() in TIMESTAMP_NAMES
        if named and rows and parse_timestamp(rows[0].get(col)) is not None:
            return col
    # Fallback: any column whose values mostly parse as timestamps.
    # Purely numeric columns only count when values look like real epochs
    # (>= 2001 in seconds, or ms equivalents) — small ints are just data.
    for col in columns:
        if col.lower() == "id":
            continue
        sample = rows[:10]

        def looks_temporal(v: Any) -> bool:
            if isinstance(v, bool):
                return False
            if isinstance(v, (int, float)):
                return abs(float(v)) >= 1e9 and parse_timestamp(v) is not None
            return parse_timestamp(v) is not None

        if sample and sum(looks_temporal(r.get(col)) for r in sample) >= max(1, len(sample) * 0.8):
            return col
    return None


def numeric_columns(columns: list[str], rows: list[dict[str, Any]],
                    exclude: str | None) -> list[str]:
    out = []
    for col in columns:
        if col == exclude:
            continue
        vals = [as_float(r.get(col)) for r in rows[:50]]
        present = [v for v in vals if v is not None]
        if present and len(present) >= max(1, len(vals) * 0.5):
            out.append(col)
    return out


def serialize_row(row: dict[str, Any], columns: list[str], *,
                  include: list[str] | None = None, exclude: list[str] | None = None,
                  max_value_length: int = 200) -> tuple[str, bool]:
    """Serialize a row as ``col: value | col: value``.

    Returns (text, truncated). Values longer than ``max_value_length``
    are truncated with an ellipsis; column include/exclude filters apply.
    """
    cols = [c for c in columns if (include is None or c in include)
            and not (exclude and c in exclude)]
    parts: list[str] = []
    truncated = False
    for c in cols:
        if row.get(c) is None:
            continue
        value = str(row.get(c))
        if len(value) > max_value_length:
            value = value[:max_value_length] + "…"
            truncated = True
        parts.append(f"{c}: {value}")
    return " | ".join(parts), truncated


# -- Prometheus exposition format ------------------------------------------
_PROM_RE = re.compile(
    r"^(?P<metric>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+)(?:\s+(?P<ts>\d+))?\s*$"
)


def parse_prometheus(text: str) -> Iterator[tuple[str, dict[str, str], float, datetime | None]]:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_RE.match(line)
        if not m:
            continue
        labels: dict[str, str] = {}
        if m.group("labels"):
            for part in re.findall(r'(\w+)="([^"]*)"', m.group("labels")):
                labels[part[0]] = part[1]
        ts = None
        if m.group("ts"):
            ts = parse_timestamp(int(m.group("ts")))
        yield m.group("metric"), labels, float(m.group("value")), ts
