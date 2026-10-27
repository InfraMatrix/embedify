"""Chunkers: turn loader units into embeddable units with exact locators."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol

from embedify.config import ChunkerConfig
from embedify.types import Locator, Modality, Unit

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def approx_token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _line_of(text: str, char_offset: int) -> int:
    return text.count("\n", 0, char_offset) + 1


class BaseChunker(Protocol):
    def chunk(self, units: Iterable[Unit]) -> Iterator[Unit]: ...


class TokenTextChunker:
    """Token-aware splitting on paragraph/sentence boundaries.

    Preserves char offsets and line ranges so every chunk can be cited.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk(self, units: Iterable[Unit]) -> Iterator[Unit]:
        for unit in units:
            yield from self._chunk_unit(unit)

    def _chunk_unit(self, unit: Unit) -> Iterator[Unit]:
        text = unit.content
        max_tokens, overlap = self.config.max_tokens, self.config.overlap
        if approx_token_count(text) <= max_tokens:
            yield unit
            return
        # Split into sentences with offsets
        pieces: list[tuple[int, str]] = []
        pos = 0
        for part in _SENTENCE_RE.split(text):
            if not part:
                continue
            idx = text.find(part, pos)
            pieces.append((idx, part))
            pos = idx + len(part)
        buffer: list[tuple[int, str]] = []
        buf_tokens = 0

        def flush() -> Unit | None:
            if not buffer:
                return None
            start = buffer[0][0]
            content = " ".join(p for _, p in buffer).strip()
            end = start + len(content)
            return Unit(
                content=content,
                modality=unit.modality,
                source_uri=unit.source_uri,
                locator=Locator(
                    page=unit.locator.page,
                    char_start=start,
                    char_end=end,
                    line_start=_line_of(text, start),
                    line_end=_line_of(text, max(start, end - 1)),
                ),
                metadata=dict(unit.metadata),
            )

        for offset, piece in pieces:
            t = approx_token_count(piece)
            if t > max_tokens:  # hard-split a single over-long piece
                if (u := flush()) is not None:
                    yield u
                buffer, buf_tokens = [], 0
                yield from self._hard_split(unit, piece, offset)
                continue
            if buf_tokens + t > max_tokens and buffer:
                if (u := flush()) is not None:
                    yield u
                # overlap: keep trailing pieces up to `overlap` tokens
                kept: list[tuple[int, str]] = []
                kept_tokens = 0
                for item in reversed(buffer):
                    kept_tokens += approx_token_count(item[1])
                    if kept_tokens > overlap:
                        break
                    kept.append(item)
                buffer = list(reversed(kept))
                buf_tokens = sum(approx_token_count(p) for _, p in buffer)
            buffer.append((offset, piece))
            buf_tokens += t
        if (u := flush()) is not None:
            yield u

    def _hard_split(self, unit: Unit, piece: str, base_offset: int) -> Iterator[Unit]:
        step = max(1, self.config.max_tokens * 4)  # ~4 chars/token
        overlap_chars = self.config.overlap * 4
        i = 0
        text = unit.content
        while i < len(piece):
            content = piece[i : i + step]
            start = base_offset + i
            yield Unit(
                content=content,
                modality=unit.modality,
                source_uri=unit.source_uri,
                locator=Locator(
                    page=unit.locator.page,
                    char_start=start,
                    char_end=start + len(content),
                    line_start=_line_of(text, start),
                    line_end=_line_of(text, start + max(0, len(content) - 1)),
                ),
                metadata=dict(unit.metadata),
            )
            i += max(1, step - overlap_chars)


class PythonASTChunker:
    """AST chunking for Python: one unit per top-level function/class.

    Falls back to token chunking for non-Python or unparseable code.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()
        self._fallback = TokenTextChunker(config)

    def chunk(self, units: Iterable[Unit]) -> Iterator[Unit]:
        for unit in units:
            if unit.metadata.get("language") != "python":
                yield from self._fallback.chunk([unit])
                continue
            try:
                tree = ast.parse(unit.content)
            except SyntaxError:
                yield from self._fallback.chunk([unit])
                continue
            lines = unit.content.splitlines(keepends=True)
            offsets: list[int] = []
            pos = 0
            for line in lines:
                offsets.append(pos)
                pos += len(line)
            emitted = False
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                emitted = True
                start_line = (node.decorator_list[0].lineno if node.decorator_list else node.lineno)
                end_line = node.end_lineno or node.lineno
                segment = "".join(lines[start_line - 1 : end_line])
                chunk = Unit(
                    content=segment.rstrip("\n"),
                    modality=unit.modality,
                    source_uri=unit.source_uri,
                    locator=Locator(
                        line_start=start_line,
                        line_end=end_line,
                        char_start=offsets[start_line - 1] if start_line - 1 < len(offsets) else 0,
                        char_end=offsets[start_line - 1] + len(segment)
                        if start_line - 1 < len(offsets) else len(segment),
                    ),
                    metadata={**unit.metadata, "symbol": node.name,
                              "symbol_type": type(node).__name__,
                              "symbol_name": node.name,
                              "symbol_kind": ("class" if isinstance(node, ast.ClassDef)
                                              else "function"),
                              "signature": segment.splitlines()[0].strip()
                              if segment.splitlines() else node.name,
                              "parent": None},
                )
                # Oversized symbols still get token-split (locators stay line-accurate
                # relative to the segment; char offsets are segment-relative fallback).
                if approx_token_count(chunk.content) > self.config.max_tokens:
                    yield from self._fallback.chunk([chunk])
                else:
                    yield chunk
            if not emitted:
                yield from self._fallback.chunk([unit])


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


class MarkdownChunker:
    """Markdown-aware chunking: split on headings first, never inside fences.

    - Heading path (e.g. ``Config > Endpoint``) is stored in
      ``metadata["heading_path"]`` and prepended to the chunk text, so a
      retrieved chunk carries its section context.
    - Fenced code blocks are atomic: a fence is never split across chunks
      (an oversized fence becomes one oversized chunk, flagged in metadata).
    - Oversized sections are then packed block-wise up to ``max_tokens``.
    Non-markdown units fall back to :class:`TokenTextChunker`.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()
        self._fallback = TokenTextChunker(config)

    def chunk(self, units: Iterable[Unit]) -> Iterator[Unit]:
        for unit in units:
            fmt = str(unit.metadata.get("format", "")).lower()
            if fmt not in ("md", "markdown"):
                yield from self._fallback.chunk([unit])
                continue
            yield from self._chunk_markdown(unit)

    def _chunk_markdown(self, unit: Unit) -> Iterator[Unit]:
        lines = unit.content.splitlines()
        # Split into (heading_path, start_line_1based, body_lines) sections,
        # tracking fences so '#' inside a fence is not a heading.
        sections: list[tuple[str, int, list[str]]] = []
        stack: list[tuple[int, str]] = []
        heading_path = ""
        start = 1
        body: list[str] = []
        in_fence = False

        def flush(end_line: int) -> None:
            if any(line.strip() for line in body):
                sections.append((heading_path, start, list(body)))

        for lineno, line in enumerate(lines, start=1):
            if _FENCE_RE.match(line):
                in_fence = not in_fence
                body.append(line)
                continue
            m = _HEADING_RE.match(line) if not in_fence else None
            if m:
                flush(lineno - 1)
                level = len(m.group(1))
                title = m.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                heading_path = " > ".join(t for _, t in stack)
                start = lineno + 1  # first body line
                body = []
            else:
                body.append(line)
        flush(len(lines))

        for path, sec_start, sec_lines in sections:
            yield from self._pack_section(unit, path, sec_start, sec_lines)

    def _pack_section(self, unit: Unit, heading_path: str, sec_start: int,
                      sec_lines: list[str]) -> Iterator[Unit]:
        # Group into atomic blocks: fenced blocks, or paragraph line-runs.
        blocks: list[tuple[int, int, str]] = []  # (start_line, end_line, text)
        block_start: int | None = None
        buf: list[str] = []
        in_fence = False

        def close_block(end_line: int) -> None:
            nonlocal block_start, buf
            if buf and any(line.strip() for line in buf):
                blocks.append((block_start or sec_start, end_line, "\n".join(buf).strip("\n")))
            buf = []
            block_start = None

        for i, line in enumerate(sec_lines):
            lineno = sec_start + i
            if _FENCE_RE.match(line):
                if not in_fence:
                    close_block(lineno - 1)
                    block_start = lineno
                    buf = [line]
                    in_fence = True
                else:
                    buf.append(line)
                    in_fence = False
                    close_block(lineno)
                continue
            if in_fence:
                buf.append(line)
                continue
            if not line.strip():
                close_block(lineno - 1)
                continue
            if block_start is None:
                block_start = lineno
            buf.append(line)
        close_block(sec_start + len(sec_lines) - 1)

        prefix = f"{heading_path}\n\n" if heading_path else ""
        current: list[tuple[int, int, str]] = []
        current_tokens = 0

        def emit() -> Unit | None:
            if not current:
                return None
            text = "\n\n".join(b[2] for b in current)
            first, last = current[0], current[-1]
            oversized = (
                len(current) == 1
                and approx_token_count(current[0][2]) > self.config.max_tokens
            )
            return Unit(
                content=f"{prefix}{text}",
                modality=unit.modality,
                source_uri=unit.source_uri,
                locator=Locator(line_start=first[0], line_end=last[1]),
                metadata={**unit.metadata,
                          **({"heading_path": heading_path} if heading_path else {}),
                          **({"oversized_fence_or_block": True} if oversized else {})},
            )

        for block in blocks:
            t = approx_token_count(block[2])
            if current and current_tokens + t > self.config.max_tokens:
                if (u := emit()) is not None:
                    yield u
                current, current_tokens = [], 0
            # A single oversized block is emitted alone, never split.
            current.append(block)
            current_tokens += t
        if (u := emit()) is not None:
            yield u


class TabularChunker:
    """Row batching / column / table granularity over row units (Phase 2).

    ``strategy="serialize"`` (default) embeds serialized text;
    ``strategy="native"`` keeps per-row units untouched so a native
    tabular embedder (``embed_units`` hook, e.g. stats) consumes
    the structured ``metadata["row"]`` payload instead of text.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk(self, units: Iterable[Unit]) -> Iterator[Unit]:
        rows = list(units)
        if not rows:
            return
        if self.config.strategy == "native":
            yield from rows
            return
        gran = self.config.granularity
        if gran == "column":
            yield from self._columns(rows)
        elif gran == "table":
            yield self._table(rows)
        else:
            yield from self._row_batches(rows)

    def _serialize(self, unit: Unit) -> tuple[str, bool]:
        from embedify.tabular import serialize_row

        row = unit.metadata.get("row")
        columns: list[str] = unit.metadata.get("columns") or []
        if not isinstance(row, dict) or not columns:
            return unit.content, False
        return serialize_row(row, columns, include=self.config.include_columns,
                             exclude=self.config.exclude_columns,
                             max_value_length=self.config.max_value_length)

    def _ser_meta(self, truncated_any: bool) -> dict[str, object]:
        return {"serialization": {
            "include_columns": self.config.include_columns,
            "exclude_columns": self.config.exclude_columns,
            "max_value_length": self.config.max_value_length,
            "truncated": truncated_any,
        }}

    def _row_batches(self, rows: list[Unit]) -> Iterator[Unit]:
        size = max(1, self.config.rows_per_chunk)
        for i in range(0, len(rows), size):
            batch = rows[i : i + size]
            first, last = batch[0], batch[-1]
            serialized = [self._serialize(u) for u in batch]
            yield Unit(
                content="\n".join(text for text, _ in serialized),
                modality=Modality.TABULAR,
                source_uri=first.source_uri,
                locator=Locator(row_id=f"{first.locator.row_id}:{last.locator.row_id}"),
                metadata={**first.metadata, "granularity": "row",
                          "row_count": len(batch),
                          "row_ids": [u.locator.row_id for u in batch],
                          **self._ser_meta(any(t for _, t in serialized))},
                numeric_values=[u.numeric_values[0] for u in batch if u.numeric_values] or None,
            )

    def _columns(self, rows: list[Unit]) -> Iterator[Unit]:
        columns: list[str] = rows[0].metadata.get("columns") or []
        if self.config.include_columns is not None:
            columns = [c for c in columns if c in self.config.include_columns]
        if self.config.exclude_columns:
            columns = [c for c in columns if c not in self.config.exclude_columns]
        for col in columns:
            values = [u.metadata.get("row", {}).get(col, "") for u in rows]
            present = [v for v in values if v != ""]
            sample = ", ".join(present[:20])
            yield Unit(
                content=(f"column {col}: {sample}"
                         + (f" … ({len(present)} values)" if present else "")),
                modality=Modality.TABULAR,
                source_uri=rows[0].source_uri,
                locator=Locator(row_id=f"column:{col}"),
                metadata={**rows[0].metadata, "granularity": "column", "column": col,
                          "row_count": len(rows), "non_null": len(present),
                          "distinct": len(set(present))},
            )

    def _table(self, rows: list[Unit]) -> Unit:
        columns: list[str] = rows[0].metadata.get("columns") or []
        sample = "\n".join(self._serialize(u)[0] for u in rows[:5])
        return Unit(
            content=(f"table {rows[0].metadata.get('table', 'table')}: "
                     f"{len(rows)} rows, columns: {', '.join(columns)}\n{sample}"),
            modality=Modality.TABULAR,
            source_uri=rows[0].source_uri,
            locator=Locator(row_id="table"),
            metadata={**rows[0].metadata, "granularity": "table",
                      "row_count": len(rows), "columns": columns},
        )


class TimeseriesChunker:
    """Sliding-window segmentation (Phase 2).

    Groups point units by series, sorts by timestamp, emits windows of
    ``window_size`` points every ``stride`` points. Windows carry the
    (optionally normalized) value matrix in ``numeric_values`` for the
    stats-foundation embedders, plus a text serialization for text models.
    Irregular sampling is detected and flagged, never silently resampled.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk(self, units: Iterable[Unit]) -> Iterator[Unit]:
        groups: dict[str, list[Unit]] = {}
        for u in units:
            groups.setdefault(str(u.metadata.get("series", u.source_uri)), []).append(u)
        for series, points in groups.items():
            yield from self._windows(series, points)

    def _windows(self, series: str, points: list[Unit]) -> Iterator[Unit]:
        def ts_of(u: Unit) -> float:
            return u.locator.time_start.timestamp() if u.locator.time_start else 0.0

        points = sorted(points, key=ts_of)
        channels: list[str] = points[0].metadata.get("channels") or []
        matrix: list[list[float]] = []
        for p in points:
            vals = p.metadata.get("values") or {}
            matrix.append([float(vals.get(c, 0.0)) for c in channels])
        # Irregular-sampling detection from timestamp deltas
        n = len(matrix)
        stamps = [ts_of(p) for p in points]
        deltas = [b - a for a, b in zip(stamps, stamps[1:], strict=False) if b > a]
        irregular = False
        median_delta: float | None = None
        if deltas:
            median_delta = sorted(deltas)[len(deltas) // 2]
            irregular = any(abs(d - median_delta) > 0.5 * median_delta for d in deltas)

        def channel_params(rows: list[list[float]]) -> dict[str, dict[str, float]]:
            """Per-channel normalization params for a set of rows."""
            params: dict[str, dict[str, float]] = {}
            for j, c in enumerate(channels):
                col = [r[j] for r in rows]
                mean = sum(col) / len(col) if col else 0.0
                var = sum((x - mean) ** 2 for x in col) / len(col) if col else 0.0
                params[c] = {"mean": mean, "std": var ** 0.5 or 1.0,
                             "min": min(col) if col else 0.0,
                             "max": max(col) if col else 0.0}
            return params

        series_params = channel_params(matrix)
        scope = self.config.normalization_scope

        def normalized(row: list[float], params: dict[str, dict[str, float]]) -> list[float]:
            if self.config.normalization == "zscore":
                return [(row[j] - params[c]["mean"]) / params[c]["std"]
                        for j, c in enumerate(channels)]
            if self.config.normalization == "minmax":
                return [(row[j] - params[c]["min"]) / ((params[c]["max"] - params[c]["min"]) or 1.0)
                        for j, c in enumerate(channels)]
            return list(row)

        size, stride = max(1, self.config.window_size), max(1, self.config.stride)
        starts = list(range(0, max(1, n - size + 1), stride)) or [0]
        if n > size and (starts[-1] + size) < n:
            starts.append(n - size)
        for start in starts:
            window = matrix[start : start + size]
            if not window:
                continue
            first_p, last_p = points[start], points[min(start + size - 1, n - 1)]
            stats: dict[str, dict[str, float]] = {}
            for j, c in enumerate(channels):
                col = [r[j] for r in window]
                stats[c] = {"mean": sum(col) / len(col), "min": min(col), "max": max(col)}
            summary = "; ".join(
                f"{c} mean={s['mean']:.4g} min={s['min']:.4g} max={s['max']:.4g}"
                for c, s in stats.items()
            )
            params = channel_params(window) if scope == "window" else series_params
            yield Unit(
                content=(f"series {series} window [{start}:{start + len(window)}] "
                         f"channels={','.join(channels)} {summary}"),
                modality=Modality.TIMESERIES,
                source_uri=points[0].source_uri,
                locator=Locator(time_start=first_p.locator.time_start,
                                time_end=last_p.locator.time_end),
                metadata={**points[0].metadata, "series": series, "channels": channels,
                          "window_start_index": start, "window_points": len(window),
                          "window_size": size, "stride": stride,
                          "normalization": self.config.normalization,
                          "normalization_scope": scope,
                          "normalization_params": params,
                          "irregular_sampling": irregular,
                          "median_interval_seconds": median_delta, "stats": stats},
                numeric_values=[normalized(r, params) for r in window],
            )


# -- Tree-sitter code chunking (extensible per language) -------------------
# Units: functions, methods, classes/interfaces/structs/etc, module header.
# Embedded text carries a context header: "<path> > <parent> > <signature>".
# Oversized symbols are split with the header + signature repeated per piece.
# Needs embedify[code]; without it, Python falls back to the AST chunker and
# other languages to token chunking.

_TS_PACKAGES: dict[str, tuple[str, str]] = {
    # language -> (pip package module, language() attr)
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
}

# node type -> symbol kind, per language (containers double as `parent`)
_TS_NODES: dict[str, dict[str, str]] = {
    "python": {"function_definition": "function", "class_definition": "class"},
    "javascript": {"function_declaration": "function", "method_definition": "method",
                   "class_declaration": "class"},
    "typescript": {"function_declaration": "function", "method_definition": "method",
                   "class_declaration": "class", "interface_declaration": "interface"},
    "tsx": {"function_declaration": "function", "method_definition": "method",
            "class_declaration": "class", "interface_declaration": "interface"},
    "go": {"function_declaration": "function", "method_declaration": "method",
           "type_declaration": "type"},
    "rust": {"function_item": "function", "impl_item": "impl", "struct_item": "struct",
             "enum_item": "enum", "trait_item": "trait"},
    "java": {"method_declaration": "method", "constructor_declaration": "method",
             "class_declaration": "class", "interface_declaration": "interface",
             "enum_declaration": "enum"},
    "c": {"function_definition": "function", "struct_specifier": "struct"},
    "cpp": {"function_definition": "function", "class_specifier": "class",
            "struct_specifier": "struct", "namespace_definition": "namespace"},
}
_TS_CONTAINERS = {"class", "interface", "struct", "enum", "trait", "impl", "namespace", "type"}


def _ts_parser(language: str) -> Any | None:
    """Build a tree-sitter Parser for `language`, or None if unavailable."""
    try:
        import importlib

        from tree_sitter import Language, Parser
    except ImportError:
        return None
    spec = _TS_PACKAGES.get(language)
    if spec is None:
        return None
    try:
        module = importlib.import_module(spec[0])
        capsule = getattr(module, spec[1])()
        lang = Language(capsule)
        try:
            return Parser(lang)
        except TypeError:  # older tree-sitter API
            parser = Parser()
            parser.language = lang
            return parser
    except Exception:
        return None


class TreeSitterChunker:
    """Tree-sitter chunker: one unit per symbol + a module-header unit."""

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()
        self._fallback_text = TokenTextChunker(config)
        self._fallback_python = PythonASTChunker(config)

    def chunk(self, units: Iterable[Unit]) -> Iterator[Unit]:
        for unit in units:
            yield from self._chunk_unit(unit)

    def _chunk_unit(self, unit: Unit) -> Iterator[Unit]:
        language = str(unit.metadata.get("language", "")).lower()
        parser = _ts_parser(language) if language in _TS_NODES else None
        if parser is None:
            if language == "python":
                yield from self._fallback_python.chunk([unit])
            else:
                yield from self._fallback_text.chunk([unit])
            return
        source = unit.content.encode("utf-8")
        tree = parser.parse(source)
        lines = unit.content.splitlines()
        path_label = str(unit.metadata.get("path") or Path(unit.source_uri).name)
        emitted: list[tuple[Any, str, str | None]] = []  # node, kind, parent

        def node_name(node: Any) -> str | None:
            named = node.child_by_field_name("name")
            if named is not None:
                return source[named.start_byte:named.end_byte].decode("utf-8", "replace")
            decl = node.child_by_field_name("declarator")
            while decl is not None and decl.child_by_field_name("declarator") is not None:
                decl = decl.child_by_field_name("declarator")
            if decl is not None:
                named = decl.child_by_field_name("name")
                if named is not None:
                    return source[named.start_byte:named.end_byte].decode("utf-8", "replace")
                if decl.child_count == 0:  # C/C++: the declarator IS the identifier
                    return source[decl.start_byte:decl.end_byte].decode("utf-8", "replace")
            return None

        def walk(node: Any, parent: str | None) -> None:
            kind = _TS_NODES[language].get(node.type)
            if kind == "function" and parent is not None and language == "python":
                kind = "method"  # function_definition nested in a class
            name = node_name(node) if kind else None
            if kind and (name or kind in ("function", "method")):
                emitted.append((node, kind, parent))
                if kind in _TS_CONTAINERS and name:
                    parent = name
            for child in node.children:
                walk(child, parent)

        walk(tree.root_node, None)

        if not emitted:
            yield from self._fallback_text.chunk([unit])
            return

        # Module header: everything before the first top-level symbol.
        first = min((n for n, _, _ in emitted), key=lambda n: n.start_point[0], default=None)
        if first is not None and first.start_point[0] > 0:
            header_lines = lines[: first.start_point[0]]
            if any(line.strip() for line in header_lines):
                header_text = "\n".join(header_lines).strip("\n")
                yield self._make_unit(unit, path_label, header_text, 1,
                                      max(1, first.start_point[0]), None, "module_header",
                                      None, header_lines[0].strip() if header_lines else "")

        for node, kind, parent in emitted:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            body = "\n".join(lines[node.start_point[0]: node.end_point[0] + 1])
            name = node_name(node)
            signature = self._signature(body, language)
            yield from self._emit_symbol(unit, path_label, body, start_line, end_line,
                                         name, kind, parent, signature)

    @staticmethod
    def _signature(body: str, language: str) -> str:
        head = body.split("{", 1)[0]
        if language == "python":
            head = head.split("\n", 1)[0] if "\n" in head else head.split(":", 1)[0] + ":"
        sig = " ".join(head.split())[:200]
        return sig or body.splitlines()[0].strip()[:200] if body.splitlines() else ""

    def _context_header(self, path_label: str, parent: str | None, signature: str) -> str:
        parts = [path_label] + ([parent] if parent else []) + ([signature] if signature else [])
        return " > ".join(parts)

    def _make_unit(self, unit: Unit, path_label: str, body: str, start: int, end: int,
                   name: str | None, kind: str, parent: str | None, signature: str) -> Unit:
        header = self._context_header(path_label, parent, signature)
        return Unit(
            content=f"{header}\n\n{body}" if header else body,
            modality=Modality.CODE,
            source_uri=unit.source_uri,
            locator=Locator(line_start=start, line_end=end),
            metadata={**unit.metadata, "symbol": name, "symbol_name": name,
                      "symbol_kind": kind, "signature": signature, "parent": parent,
                      "context_header": header},
        )

    def _emit_symbol(self, unit: Unit, path_label: str, body: str, start: int, end: int,
                     name: str | None, kind: str, parent: str | None,
                     signature: str) -> Iterator[Unit]:
        if approx_token_count(body) <= self.config.max_tokens:
            yield self._make_unit(unit, path_label, body, start, end, name, kind, parent, signature)
            return
        # Oversized: split line-wise; header + signature repeated on every piece.
        max_tokens = max(16, self.config.max_tokens)
        piece: list[str] = []
        piece_start = start
        tokens = 0

        def flush(piece_lines: list[str], s: int) -> Unit:
            return self._make_unit(unit, path_label, "\n".join(piece_lines), s,
                                   s + len(piece_lines) - 1, name, kind, parent, signature)

        for offset, line in enumerate(body.splitlines()):
            t = approx_token_count(line) + 1
            if piece and tokens + t > max_tokens:
                yield flush(piece, piece_start)
                piece, tokens, piece_start = [], 0, start + offset
            piece.append(line)
            tokens += t
        if piece:
            yield flush(piece, piece_start)


def register_code_language(language: str, package: str, attr: str,
                           nodes: dict[str, str]) -> None:
    """Register an additional tree-sitter language (extensibility hook)."""
    _TS_PACKAGES[language] = (package, attr)
    _TS_NODES[language] = nodes


def get_chunker(modality: Modality, config: ChunkerConfig) -> BaseChunker:
    strategy = config.strategy
    if modality == Modality.CODE and strategy == "ast":
        return PythonASTChunker(config)
    if modality == Modality.CODE and strategy in (None, "treesitter", "tree-sitter"):
        # Default for code; falls back to Python AST / token chunking when
        # tree-sitter or the language grammar is not installed.
        return TreeSitterChunker(config)
    if modality == Modality.CODE:
        return TokenTextChunker(config)
    if modality == Modality.TEXT and strategy in (None, "markdown"):
        return MarkdownChunker(config)
    if modality == Modality.TABULAR:
        return TabularChunker(config)
    if modality == Modality.TIMESERIES:
        return TimeseriesChunker(config)
    return TokenTextChunker(config)
