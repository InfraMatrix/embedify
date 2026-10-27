"""Pipeline: detect -> load -> chunk -> embed -> sink. Streaming, per-file errors."""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

from embedify.chunkers import approx_token_count, get_chunker
from embedify.config import ArchiveConfig, EmbedifyConfig
from embedify.detector import Detector
from embedify.embedders import BaseEmbedder, create_embedder
from embedify.errors import EmbedderError
from embedify.loaders import LOADERS
from embedify.sinks import BaseSink, LanceDBSink, MemorySink
from embedify.types import (
    EmbeddingRecord,
    FileError,
    InputType,
    JobResult,
    JobStats,
    Modality,
    SecretFinding,
    Unit,
)

log = logging.getLogger("embedify")


def _note_skip(stats: JobStats, reason: str) -> None:
    stats.files_skipped += 1
    stats.skipped_by_reason[reason] = stats.skipped_by_reason.get(reason, 0) + 1


def normalize_vector(vector: list[float]) -> list[float]:
    """L2-normalize; zero vectors pass through. All stored vectors are
    unit-norm so LanceDB cosine scores are comparable."""
    import math

    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 0 else list(vector)

SUPPORTED_PHASE1 = {
    Modality.TEXT, Modality.CODE, Modality.DOCUMENT, Modality.UNKNOWN,
    Modality.TABULAR, Modality.TIMESERIES,
}
_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", ".ruff_cache"}


def _expand_archives(files: list[Path], errors: list[FileError],
                     stats: JobStats,
                     limits: ArchiveConfig) -> tuple[list[Path], dict[str, str],
                                                     tempfile.TemporaryDirectory[str] | None]:
    """Extract archive sources to a temp dir; process exactly like a directory.

    Archives found in directory walks — and archives inside archives — are
    expanded recursively (up to ``limits.max_nesting_depth``), chaining
    source_uris: ``outer.zip!inner.zip!deep/file.py``. Extraction limits
    (files / member bytes / total bytes) come from ``limits`` too.
    Returns (expanded files, {extracted_path: display_uri} map, temp-dir
    handle kept alive for the run). Unsafe/corrupt archives are errors.
    """
    import itertools

    from embedify.archives import archive_uri, extract_archive, is_archive
    from embedify.repo import iter_repo_files

    if not any(is_archive(f) for f in files):
        return files, {}, None
    tmp_dir = tempfile.TemporaryDirectory(prefix="embedify-archive-")
    tmp: tempfile.TemporaryDirectory[str] | None = tmp_dir
    expanded: list[Path] = []
    mapping: dict[str, str] = {}
    counter = itertools.count()

    def expand(candidates: list[tuple[Path, str]], depth: int) -> None:
        for path, display_uri in candidates:
            if not is_archive(path):
                expanded.append(path)
                mapping[str(path)] = display_uri
                continue
            if depth > limits.max_nesting_depth:
                stats.files_failed += 1
                errors.append(FileError(
                    source_uri=display_uri,
                    error=f"Archive nesting exceeds {limits.max_nesting_depth} levels"))
                continue
            root = Path(tmp_dir.name) / str(next(counter))
            try:
                extract_archive(path, root, max_files=limits.max_files,
                                max_member_bytes=limits.max_member_bytes,
                                max_total_bytes=limits.max_total_bytes)
            except Exception as exc:  # per-file isolation, like other loaders
                stats.files_failed += 1
                errors.append(FileError(source_uri=display_uri,
                                        error=f"{type(exc).__name__}: {exc}"))
                log.warning("embedify: failed archive %s: %s", display_uri, exc)
                continue
            root_resolved = root.resolve()
            inner = [
                (extracted, archive_uri(display_uri,
                                        extracted.relative_to(root_resolved).as_posix()))
                for extracted in iter_repo_files(root)
            ]
            expand(inner, depth + 1)

    expand([(f, str(f)) for f in files], 0)
    return expanded, mapping, tmp


def _iter_files(source: Path) -> Iterator[Path]:
    if source.is_file():
        yield source
    elif source.is_dir():
        # Repo-aware walk: .gitignore, vendored/generated/minified/lock/binary skips
        from embedify.repo import iter_repo_files

        yield from iter_repo_files(source)
    else:
        raise FileNotFoundError(f"Source not found: {source}")


def _resolve_source(source: object, tmp: Path | None = None) -> tuple[list[Path], str | None]:
    """Return (file paths, raw_text_if_inline)."""
    if isinstance(source, Path):
        return list(_iter_files(source)), None
    if isinstance(source, str):
        p = Path(source)
        if p.exists():
            return list(_iter_files(p)), None
        if source.startswith(("http://", "https://")):
            raise NotImplementedError(
                "URL sources land in Phase 2; download the file first, then embed the path"
            )
        # Treat as inline raw text
        return [], source
    if isinstance(source, (list, tuple)):
        files: list[Path] = []
        for item in source:
            sub, _ = _resolve_source(item)
            files.extend(sub)
        return files, None
    raise TypeError(f"Unsupported source type: {type(source).__name__}")


def _structured_rows(source: object) -> list[dict[str, object]] | None:
    """DataFrame-like objects and list-of-dict sources -> row dicts."""
    if isinstance(source, list) and source and all(isinstance(r, dict) for r in source):
        return [dict(r) for r in source if isinstance(r, dict)]
    to_dict = getattr(source, "to_dict", None)
    columns = getattr(source, "columns", None)
    if callable(to_dict) and columns is not None:
        try:
            records = to_dict("records")
        except Exception:
            return None
        if isinstance(records, list) and all(isinstance(r, dict) for r in records):
            return records
    return None


class Pipeline:
    def __init__(self, config: EmbedifyConfig, *, detector: Detector | None = None,
                 embedder: BaseEmbedder | None = None) -> None:
        self.config = config
        self.detector = detector or Detector()
        self._embedder_override = embedder
        self._embedders: dict[str, BaseEmbedder] = {}

    def _embedder(self, modality: Modality) -> BaseEmbedder:
        if self._embedder_override is not None:
            return self._embedder_override
        key = str(modality)
        if key not in self._embedders:
            self._embedders[key] = create_embedder(self.config.embedder_for(key))
        return self._embedders[key]

    def run(self, source: object, sink: BaseSink, *, incremental: bool = False) -> JobResult:
        started = time.perf_counter()
        stats = JobStats()
        errors: list[FileError] = []
        findings: list[SecretFinding] = []
        records: list[EmbeddingRecord] = []
        structured_rows = _structured_rows(source)
        files: list[Path]
        inline_text: str | None
        if structured_rows is not None:
            files, inline_text = [], None
        else:
            files, inline_text = _resolve_source(source)

        units_with_modality: list[tuple[Unit, Modality]] = []
        if inline_text is not None:
            from embedify.types import Locator

            inline_unit = Unit(content=inline_text, modality=Modality.TEXT,
                               source_uri="<inline>", locator=Locator())
            chunker = get_chunker(Modality.TEXT, self.config.chunker_for("text"))
            for unit in chunker.chunk([inline_unit]):
                units_with_modality.append((unit, Modality.TEXT))
            stats.files_processed += 1

        # In-memory structured sources: list[dict] rows or a DataFrame-like object
        if structured_rows is not None:
            from embedify.tabular import serialize_row

            columns = list(structured_rows[0]) if structured_rows else []
            row_units = [
                Unit(
                    content=serialize_row(row, columns)[0], modality=Modality.TABULAR,
                    source_uri="<dataframe>",
                    metadata={"format": "dataframe", "table": "dataframe",
                              "columns": columns,
                              "row": {k: str(v) for k, v in row.items()}, "row_id": str(idx)},
                )
                for idx, row in enumerate(structured_rows)
            ]
            chunker = get_chunker(Modality.TABULAR, self.config.chunker_for("tabular"))
            for unit in chunker.chunk(row_units):
                units_with_modality.append((unit, Modality.TABULAR))
            stats.files_processed += 1
            files = []

        # Archive sources (.tar/.tar.gz/.tgz/.zip) are extracted safely and
        # then processed exactly like a directory; source_uri keeps the
        # "archive!inner/path" identity via archive_map.
        files, archive_map, archive_tmp = _expand_archives(
            files, errors, stats, self.config.archive)
        stats.files_scanned = len(files) + (
            1 if inline_text is not None or structured_rows is not None else 0)

        for path in files:
            display_uri = archive_map.get(str(path), str(path))
            try:
                from embedify.secrets import is_secret_filename, scan_text

                if is_secret_filename(path):
                    findings.append(SecretFinding(source_uri=display_uri, kind="secret-file"))
                    _note_skip(stats, "secret-file")
                    stats.secrets_found += 1
                    continue
                det = self.detector.detect(path)
                if det.modality not in SUPPORTED_PHASE1:
                    raise NotImplementedError(
                        f"Modality '{det.modality}' for {path} is deferred "
                        "(media modalities are not in this phase)"
                    )
                loader = next(loader for loader in LOADERS if loader.supports(det))
                raw_units = list(loader.load(path, det))
                for raw in raw_units:
                    raw.source_uri = display_uri
                # Secret scan BEFORE embedding: a file with any finding is
                # excluded wholesale; findings never contain the secret value.
                file_findings: list[SecretFinding] = []
                for raw in raw_units:
                    file_findings.extend(scan_text(raw.content, source_uri=display_uri))
                if file_findings:
                    findings.extend(file_findings)
                    _note_skip(stats, "secret-content")
                    stats.secrets_found += len(file_findings)
                    log.warning("embedify: excluded %s (%d secret finding(s))",
                                path, len(file_findings))
                    continue
                # Git provenance for code records (repo/branch/commit/path).
                if det.modality == Modality.CODE:
                    if str(path) in archive_map:
                        # Inside an archive there is no git repo; the path is
                        # the archive-inner path.
                        provenance: dict[str, object] = {
                            "repo": None, "branch": None, "commit": None,
                            "path": display_uri.split("!", 1)[1],
                        }
                    else:
                        from embedify.repo import git_metadata

                        provenance = dict(git_metadata(path))
                    for raw in raw_units:
                        raw.metadata.update(provenance)
                        raw.metadata.setdefault("language", det.language or "unknown")
                # A loader may reclassify units (CSV -> time-series points),
                # so chunk per unit-modality, not per file detection.
                by_unit_modality: dict[Modality, list[Unit]] = {}
                for raw in raw_units:
                    by_unit_modality.setdefault(raw.modality, []).append(raw)
                for unit_modality, group in by_unit_modality.items():
                    chunker = get_chunker(unit_modality,
                                          self.config.chunker_for(str(unit_modality)))
                    for unit in chunker.chunk(group):
                        units_with_modality.append((unit, unit_modality))
                stats.files_processed += 1
            except Exception as exc:  # per-file isolation: never crash the job
                stats.files_failed += 1
                errors.append(FileError(source_uri=display_uri,
                                        error=f"{type(exc).__name__}: {exc}"))
                log.warning("embedify: failed %s: %s", path, exc)

        # Incremental re-embedding: unchanged files are skipped, changed
        # files replace their stored records, deleted files are removed.
        # Only sinks exposing stored_source_hashes()/delete_source() support it.
        if incremental and hasattr(sink, "stored_source_hashes"):
            stored: dict[str, set[str]] = sink.stored_source_hashes()
            current_hashes: dict[str, set[str]] = {}
            for unit, _m in units_with_modality:
                current_hashes.setdefault(unit.source_uri, set()).add(unit.content_hash)
            current_uris = {str(p) for p in files}
            kept: list[tuple[Unit, Modality]] = []
            replaced: set[str] = set()
            for unit, modality in units_with_modality:
                uri = unit.source_uri
                if uri in current_uris and stored.get(uri) == current_hashes.get(uri):
                    continue  # unchanged: stored records already current
                if uri in current_uris and uri in stored and uri not in replaced:
                    stats.records_deleted += sink.delete_source(uri)  # type: ignore[attr-defined]
                    replaced.add(uri)
                kept.append((unit, modality))
            skipped_uris = current_uris & {u for u, h in stored.items()
                                           if h == current_hashes.get(u)}
            for _uri in skipped_uris:
                _note_skip(stats, "unchanged")
            # Deleted files: stored under an embedded directory but gone now.
            roots = [Path(s) for s in [source] if isinstance(s, (str, Path))
                     and Path(str(s)).is_dir()]
            for uri in list(stored):
                if uri not in current_uris and any(
                        uri.startswith(str(r) + "/") or uri.startswith(str(r.resolve()) + "/")
                        for r in roots):
                    stats.records_deleted += sink.delete_source(uri)  # type: ignore[attr-defined]
            units_with_modality = kept

        # Embed grouped by modality (each modality may route to a different model)
        by_modality: dict[Modality, list[Unit]] = {}
        for unit, modality in units_with_modality:
            by_modality.setdefault(modality, []).append(unit)

        for modality, units in by_modality.items():
            try:
                embedder = self._embedder(modality)
                if modality not in embedder.modalities:
                    raise EmbedderError(
                        f"Embedder '{embedder.model_id}' does not declare modality "
                        f"'{modality}'. Configure a per-modality embedder for it "
                        f"in embedify.yaml (embedders.{modality})."
                    )
                batch_size = max(1, embedder.max_batch)
                for i in range(0, len(units), batch_size):
                    batch = units[i : i + batch_size]
                    payloads = [u.payload_for(embedder.payload_types) for u in batch]
                    embed_fn = getattr(embedder, "embed", None)
                    if callable(embed_fn):
                        vectors = embed_fn(payloads, input_type=InputType.DOCUMENT)
                    else:  # legacy custom embedders implementing only embed_batch
                        vectors = embedder.embed_batch(
                            [u.content for u in batch], input_type=InputType.DOCUMENT
                        )
                    if len(vectors) != len(batch):
                        raise ValueError("Embedder returned wrong number of vectors")
                    batch_records: list[EmbeddingRecord] = []
                    for unit, vector in zip(batch, vectors, strict=True):
                        vector = normalize_vector([float(x) for x in vector])
                        rec = EmbeddingRecord(
                            id=uuid.uuid5(uuid.NAMESPACE_URL,
                                          f"{unit.source_uri}:{unit.content_hash}").hex,
                            vector=vector,
                            model=embedder.model_id,
                            model_version=embedder.model_version,
                            dim=len(vector),
                            modality=unit.modality,
                            source_uri=unit.source_uri,
                            locator=unit.locator,
                            content_hash=unit.content_hash,
                            text_preview=unit.text_preview,
                            metadata=dict(unit.metadata),
                        )
                        batch_records.append(rec)
                        stats.units_embedded += 1
                        stats.tokens_estimated += approx_token_count(unit.content)
                    stats.units_processed += len(batch)
                    sink.write(batch_records)
            except Exception as exc:  # modality-level isolation, like per-file
                stats.files_failed += 1
                errors.append(FileError(
                    source_uri=f"<modality:{modality}>",
                    error=f"{type(exc).__name__}: {exc}",
                ))
                log.warning("embedify: modality %s failed: %s", modality, exc)
        sink.finalize()
        if isinstance(sink, MemorySink):
            records = list(sink.records)
        if archive_tmp is not None:
            archive_tmp.cleanup()
        stats.elapsed_seconds = time.perf_counter() - started
        return JobResult(records=records, stats=stats, errors=errors, findings=findings)


def embedder_for_stored_model(config: EmbedifyConfig, sink: LanceDBSink) -> BaseEmbedder:
    """Pick the query embedder from the collection's *stored* model.

    Never silently falls back to the configured default text embedder:
    querying a stats/CLIP/bge collection with a different model would
    produce incomparable (or wrong-dim) vectors.
    """
    from copy import copy

    from embedify.config import EmbedderConfig

    info = sink.table_info()
    if not info or not info.get("model"):
        return create_embedder(config.embedder_for("text"))
    stored = str(info["model"])
    if stored == "hash":
        return create_embedder(EmbedderConfig(backend="hash", model="hash", dim=info["dim"]))
    if stored in ("stats-features", "stats"):
        return create_embedder(
            EmbedderConfig(backend="stats", model=stored, dim=info["dim"])
        )
    cfg = config.embedder_for("text")
    if cfg.model == stored:
        return create_embedder(cfg)
    if cfg.backend == "endpoint":
        overridden = copy(cfg)
        overridden.model = stored
        return create_embedder(overridden)
    # Local (or shared-space CLIP-style) model id stored: load exactly that model.
    return create_embedder(
        EmbedderConfig(backend=cfg.backend if cfg.backend in ("local", "shared") else "local",
                       model=stored, dim=info["dim"])
    )


def query_lancedb(
    query_text: str, *, config: EmbedifyConfig, db_path: str, table: str = "chunks",
    top_k: int = 5, embedder: BaseEmbedder | None = None,
) -> list[tuple[EmbeddingRecord, float]]:
    """Embed a query with input_type='query' and search a LanceDB sink."""
    sink = LanceDBSink(db_path, table)
    emb = embedder or embedder_for_stored_model(config, sink)
    vector = normalize_vector(
        [float(x) for x in emb.embed_batch([query_text], input_type=InputType.QUERY)[0]]
    )
    return sink.search(vector, top_k=top_k)


def query_by_example_lancedb(
    source: object, *, config: EmbedifyConfig, db_path: str, table: str = "chunks",
    top_k: int = 5, time_start: object = None, time_end: object = None,
    embedder: BaseEmbedder | None = None,
) -> list[tuple[EmbeddingRecord, float]]:
    """Query-by-example for time series (and any re-embeddable source).

    Embeds ``source`` in-memory (e.g. a window cut from a CSV, optionally
    restricted to records overlapping ``time_start``/``time_end``), takes
    the matching window's vector, and returns the most similar stored
    windows. The example itself is excluded from results.
    """
    from datetime import datetime as _dt

    sink = LanceDBSink(db_path, table)
    emb = embedder or embedder_for_stored_model(config, sink)
    result = Pipeline(config, embedder=emb).run(source, MemorySink())
    if result.errors and not result.records:
        raise EmbedderError(f"Could not embed example source: {result.errors[0].error}")

    def _parse(v: object) -> _dt | None:
        if v is None:
            return None
        parsed = v if isinstance(v, _dt) else _dt.fromisoformat(str(v))
        # Stored locators are tz-aware (UTC); treat naive input as UTC so
        # range comparisons never mix naive/aware datetimes.
        if parsed.tzinfo is None:
            from datetime import UTC

            parsed = parsed.replace(tzinfo=UTC)
        return parsed

    start, end = _parse(time_start), _parse(time_end)
    candidates = result.records
    if start is not None or end is not None:
        def overlaps(r: EmbeddingRecord) -> bool:
            lo, hi = r.locator.time_start, r.locator.time_end
            if lo is None:
                return False
            return (end is None or lo <= end) and (start is None or (hi or lo) >= start)

        candidates = [r for r in candidates if overlaps(r)]
    if not candidates:
        raise EmbedderError(
            "No embedded window from the example source overlaps the requested time range."
        )
    example = candidates[0]
    hits = sink.search(example.vector, top_k=top_k + 1)
    return [(r, s) for r, s in hits if r.content_hash != example.content_hash][:top_k]
