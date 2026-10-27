"""Public Python API: ``embedify.embed(...)`` and ``embedify.query(...)``."""

from __future__ import annotations

from pathlib import Path

from embedify.config import EmbedifyConfig
from embedify.detector import Detector
from embedify.embedders import BaseEmbedder
from embedify.pipeline import Pipeline, query_by_example_lancedb, query_lancedb
from embedify.sinks import BaseSink, create_sink
from embedify.types import EmbeddingRecord, JobResult


def embed(source: object, *, config: EmbedifyConfig | str | Path | None = None,
          sink: BaseSink | None = None, embedder: BaseEmbedder | None = None,
          detector: Detector | None = None, incremental: bool = False) -> JobResult:
    """Embed a file, directory, list of paths, or inline text string.

    Indexing always passes ``input_type="document"`` to embedders.
    Returns a JobResult; ``result.records`` is populated for the memory sink.
    For file/LanceDB sinks, records are on disk — stats/errors are returned.

    ``incremental=True`` (sinks supporting it, e.g. LanceDB/memory):
    unchanged files keep their stored records, changed files replace
    theirs, and files deleted under an embedded directory are removed.
    Secret-bearing files are always excluded and reported in
    ``result.findings`` (kind + line only, never the secret value).
    """
    cfg = _coerce_config(config)
    resolved_sink = sink or create_sink(cfg.sink)
    return Pipeline(cfg, detector=detector, embedder=embedder).run(
        source, resolved_sink, incremental=incremental)


def query(text: str, *, db: str | Path, table: str = "chunks", top_k: int = 5,
          config: EmbedifyConfig | str | Path | None = None,
          embedder: BaseEmbedder | None = None) -> list[tuple[EmbeddingRecord, float]]:
    """Search a LanceDB store built by :func:`embed`.

    Passes ``input_type="query"`` so BGE/E5 models get their query prefix;
    vectors are only comparable within the same model+dim that indexed them.
    """
    cfg = _coerce_config(config)
    return query_lancedb(text, config=cfg, db_path=str(db), table=table,
                         top_k=top_k, embedder=embedder)


def query_by_example(source: object, *, db: str | Path, table: str = "chunks", top_k: int = 5,
                     start: object = None, end: object = None,
                     config: EmbedifyConfig | str | Path | None = None,
                     embedder: BaseEmbedder | None = None) -> list[tuple[EmbeddingRecord, float]]:
    """Find stored windows similar to an example window cut from ``source``.

    ``source`` is anything :func:`embed` accepts (typically a time-series
    file); ``start``/``end`` optionally restrict to windows overlapping
    that time range (ISO strings or datetimes). The embedder is resolved
    from the target collection's stored model.
    """
    cfg = _coerce_config(config)
    return query_by_example_lancedb(source, config=cfg, db_path=str(db), table=table,
                                    top_k=top_k, time_start=start, time_end=end,
                                    embedder=embedder)


async def aembed(*args: object, **kwargs: object) -> JobResult:
    raise NotImplementedError("Async API lands in Phase 4 with the FastAPI server; use embed()")


def _coerce_config(config: EmbedifyConfig | str | Path | None) -> EmbedifyConfig:
    if config is None:
        return EmbedifyConfig.load()
    if isinstance(config, EmbedifyConfig):
        return config
    return EmbedifyConfig.from_yaml(config)
