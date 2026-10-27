"""Loaders: one per format, emitting a stream of Units.

Phase 1: text/markdown/html/log/email, PDF (via pypdf), code, unknown fallback.
Later phases register tabular/timeseries/image/audio/video loaders here.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol

from embedify.detector import Detection, Detector
from embedify.errors import LoaderError
from embedify.types import Locator, Modality, Unit


class BaseLoader(Protocol):
    def supports(self, detection: Detection) -> bool: ...
    def load(self, path: Path, detection: Detection) -> Iterator[Unit]: ...


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:  # type: ignore[type-arg]
        if tag in ("script", "style"):
            self._skip += 1
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


class TextLoader:
    def supports(self, detection: Detection) -> bool:
        return detection.modality == Modality.TEXT

    def load(self, path: Path, detection: Detection) -> Iterator[Unit]:
        text = _read_text(path)
        if path.suffix.lower() in (".html", ".htm"):
            parser = _TextExtractor()
            parser.feed(text)
            text = re.sub(r"\n{3,}", "\n\n", "".join(parser.parts)).strip()
        if text.strip():
            yield Unit(content=text, modality=Modality.TEXT, source_uri=str(path),
                       metadata={"format": path.suffix.lower().lstrip(".") or "txt"})


class CodeLoader:
    def supports(self, detection: Detection) -> bool:
        return detection.modality == Modality.CODE

    def load(self, path: Path, detection: Detection) -> Iterator[Unit]:
        text = _read_text(path)
        if text.strip():
            yield Unit(content=text, modality=Modality.CODE, source_uri=str(path),
                       metadata={"language": detection.language or "unknown",
                                 "format": path.suffix.lower().lstrip(".")})


class PDFLoader:
    def supports(self, detection: Detection) -> bool:
        return detection.modality == Modality.DOCUMENT and (
            detection.source_uri.lower().endswith(".pdf") or detection.detail == "magic-bytes"
        )

    def load(self, path: Path, detection: Detection) -> Iterator[Unit]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise LoaderError(
                "PDF support needs pypdf: pip install 'embedify[pdf]'"
            ) from exc
        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise LoaderError(f"Could not read PDF {path}: {exc}") from exc
        if reader.is_encrypted:
            raise LoaderError(f"Encrypted PDF not supported in Phase 1: {path}")
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                yield Unit(content=text, modality=Modality.DOCUMENT, source_uri=str(path),
                           locator=Locator(page=i),
                           metadata={"format": "pdf", "page": i})
            # Scanned/empty pages: OCR is Phase 3; metadata-only unit + warning flag.
            else:
                yield Unit(content=f"[pdf page {i}: no extractable text]",
                           modality=Modality.DOCUMENT, source_uri=str(path),
                           locator=Locator(page=i),
                           metadata={"format": "pdf", "page": i, "ocr_needed": True,
                                     "metadata_only": True})


class TabularLoader:
    """CSV/TSV/JSON/JSONL/Parquet/Excel/SQLite -> row units.

    Rows from a table with a timestamp column + numeric columns become
    TIMESERIES point units (windowed later by TimeseriesChunker); all
    other rows become TABULAR row units (batched/aggregated later).
    """

    def supports(self, detection: Detection) -> bool:
        return detection.modality in (Modality.TABULAR, Modality.TIMESERIES) and (
            Path(detection.source_uri).suffix.lower() != ".prom"
        )

    def load(self, path: Path, detection: Detection) -> Iterator[Unit]:
        """Streams: peeks at the first 100 rows of each table to classify
        (tabular vs time-series), then yields rows lazily as parsed."""
        from embedify.tabular import detect_time_column, iter_row_stream, numeric_columns

        stream = iter(iter_row_stream(path))
        pending: tuple[str, list[str], dict[str, object]] | None = next(stream, None)
        while pending is not None:
            table_name, columns, _first_row = pending
            buffered: list[tuple[str, list[str], dict[str, object]]] = []
            while pending is not None and pending[0] == table_name and len(buffered) < 100:
                buffered.append(pending)
                pending = next(stream, None)
            sample_rows: list[dict[str, object]] = [row for _, _, row in buffered]
            time_col = detect_time_column(columns, sample_rows)
            numeric = numeric_columns(columns, sample_rows, time_col)
            is_series = time_col is not None and bool(numeric)
            id_col = next(
                (c for c in columns if c.lower() in ("id", "row_id", "key")), None
            )
            # Emit buffered rows, then keep streaming this table lazily.
            emitted = 0
            for _t, _c, row in buffered:
                yield self._row_unit(path, table_name, columns, row, emitted,
                                     id_col, time_col, numeric, is_series)
                emitted += 1
            while pending is not None and pending[0] == table_name:
                _t, _c, row = pending
                yield self._row_unit(path, table_name, columns, row, emitted,
                                     id_col, time_col, numeric, is_series)
                emitted += 1
                pending = next(stream, None)

    @staticmethod
    def _row_unit(path: Path, table_name: str, columns: list[str],
                  row: dict[str, object], idx: int, id_col: str | None,
                  time_col: str | None, numeric: list[str], is_series: bool) -> Unit:
        from embedify.tabular import as_float, parse_timestamp, serialize_row

        row_id = str(row.get(id_col)) if id_col and row.get(id_col) is not None else str(idx)
        if is_series:
            ts = parse_timestamp(row.get(time_col)) if time_col else None
            values: dict[str, float] = {}
            for c in numeric:
                fv = as_float(row.get(c))
                if fv is not None:
                    values[c] = fv
            return Unit(
                content=serialize_row(row, columns)[0],
                modality=Modality.TIMESERIES,
                source_uri=str(path),
                locator=Locator(row_id=row_id, time_start=ts, time_end=ts),
                metadata={
                    "format": path.suffix.lower().lstrip("."),
                    "table": table_name, "row": {k: str(v) for k, v in row.items()},
                    "timestamp": ts.isoformat() if ts else None,
                    "channels": list(values), "values": values,
                    "series": table_name,
                },
                numeric_values=[[values[c] for c in values]] if values else None,
            )
        return Unit(
            content=serialize_row(row, columns)[0],
            modality=Modality.TABULAR,
            source_uri=str(path),
            locator=Locator(row_id=row_id),
            metadata={
                "format": path.suffix.lower().lstrip("."),
                "table": table_name, "columns": columns,
                "row": {k: (str(v) if v is not None else "") for k, v in row.items()},
                "row_id": row_id,
            },
        )


class PrometheusLoader:
    """Prometheus exposition text -> one time-series point per sample."""

    def supports(self, detection: Detection) -> bool:
        return Path(detection.source_uri).suffix.lower() == ".prom"

    def load(self, path: Path, detection: Detection) -> Iterator[Unit]:
        from embedify.tabular import parse_prometheus

        for metric, labels, value, ts in parse_prometheus(_read_text(path)):
            series = metric + ("{" + ",".join(f"{k}={v}" for k, v in sorted(labels.items())) + "}"
                               if labels else "")
            yield Unit(
                content=f"{series}={value}",
                modality=Modality.TIMESERIES,
                source_uri=str(path),
                locator=Locator(time_start=ts, time_end=ts),
                metadata={"format": "prom", "series": series, "metric": metric,
                          "labels": labels, "channels": [metric],
                          "values": {metric: value},
                          "timestamp": ts.isoformat() if ts else None},
                numeric_values=[[value]],
            )


class FallbackLoader:
    """Unknown/binary: metadata-only unit, never a crash."""

    def supports(self, detection: Detection) -> bool:
        return True

    def load(self, path: Path, detection: Detection) -> Iterator[Unit]:
        size = path.stat().st_size if path.exists() else 0
        yield Unit(
            content=f"[binary/unknown file: {path.name}, {size} bytes]",
            modality=Modality.UNKNOWN,
            source_uri=str(path),
            metadata={"format": "unknown", "metadata_only": True, "size_bytes": size,
                      "warning": "Unknown/binary format; metadata-only embedding"},
        )


LOADERS: list[BaseLoader] = [
    PrometheusLoader(), TabularLoader(), TextLoader(), CodeLoader(), PDFLoader(), FallbackLoader(),
]


def load_path(path: str | Path, detector: Detector | None = None) -> Iterator[Unit]:
    p = Path(path)
    det = (detector or Detector()).detect(p)
    for loader in LOADERS:
        if loader.supports(det):
            yield from loader.load(p, det)
            return
    yield from FallbackLoader().load(p, det)  # pragma: no cover
