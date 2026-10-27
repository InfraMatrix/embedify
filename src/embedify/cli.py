"""CLI: ``embedify embed``, ``embedify query``, ``embedify inspect``, ``embedify models``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from embedify.api import embed as _embed
from embedify.api import query as _query
from embedify.config import EmbedderConfig, EmbedifyConfig, SinkConfig
from embedify.detector import Detector
from embedify.sinks import create_sink

app = typer.Typer(help="Embed any data for RAG and agentic systems.", no_args_is_help=True)


def _load_config(path: str | None) -> EmbedifyConfig:
    return EmbedifyConfig.load(Path(path) if path else None)


@app.command()
def embed(
    source: Annotated[str, typer.Argument(help="File, directory, or inline text")],
    config: Annotated[str | None, typer.Option("--config", "-c")] = None,
    sink_type: Annotated[str | None, typer.Option("--sink", help="memory|file|lancedb")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Sink path")] = None,
    table: Annotated[str, typer.Option("--table")] = "chunks",
    backend: Annotated[str | None, typer.Option("--backend", help="local|endpoint|hash")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    incremental: Annotated[bool, typer.Option(
        "--incremental", help="Skip unchanged, replace changed, remove deleted files")] = False,
) -> None:
    """Embed a source (indexing: input_type=document)."""
    cfg = _load_config(config)
    if backend or model or base_url:
        for key in list(cfg.embedders) or ["text"]:
            ecfg = cfg.embedders.setdefault(key, EmbedderConfig())
            if backend:
                ecfg.backend = backend
            if model:
                ecfg.model = model
            if base_url:
                ecfg.base_url = base_url
        if not cfg.embedders:
            cfg.embedders["text"] = EmbedderConfig(backend=backend or "local",
                                                   model=model or EmbedderConfig().model,
                                                   base_url=base_url)
    if sink_type:
        cfg.sink = SinkConfig(type=sink_type, path=output or cfg.sink.path, table=table)
    elif output:
        cfg.sink.path = output
    elif cfg.sink.type == "memory":
        # UX default: bare `embedify embed` persists so `embedify query`
        # works in a later run. Memory is opt-in via --sink memory.
        cfg.sink = SinkConfig(type="lancedb", path="./embedify.lancedb", table=table)
    result = _embed(source, config=cfg, sink=create_sink(cfg.sink), incremental=incremental)
    payload: dict[str, Any] = {
        "units_embedded": result.stats.units_embedded,
        "files_processed": result.stats.files_processed,
        "files_failed": result.stats.files_failed,
        "files_skipped": result.stats.files_skipped,
        "records_deleted": result.stats.records_deleted,
        "elapsed_seconds": round(result.stats.elapsed_seconds, 3),
        "stats": {
            "files_scanned": result.stats.files_scanned,
            "files_processed": result.stats.files_processed,
            "files_failed": result.stats.files_failed,
            "files_skipped": result.stats.files_skipped,
            "skipped_by_reason": result.stats.skipped_by_reason,
            "secrets_found": result.stats.secrets_found,
            "units_processed": result.stats.units_processed,
            "units_embedded": result.stats.units_embedded,
            "records_deleted": result.stats.records_deleted,
            "elapsed_seconds": round(result.stats.elapsed_seconds, 3),
            "units_per_second": round(result.stats.units_per_second, 2),
        },
        "errors": [e.to_dict() for e in result.errors],
        # Secret findings: kind + line only, never the secret value.
        "secret_findings": [f.to_dict() for f in result.findings],
    }
    if cfg.sink.type == "memory":
        payload["records"] = [r.to_dict() for r in result.records]
    else:
        payload["sink"] = {"type": cfg.sink.type, "path": cfg.sink.path,
                           "table": getattr(cfg.sink, "table", None)}
        typer.echo(
            f"Embedded {result.stats.units_embedded} units -> "
            f"{cfg.sink.type}:{cfg.sink.path} (table={cfg.sink.table})",
            err=True,
        )
    typer.echo(json.dumps(payload, indent=2))


@app.command()
def query(
    text: Annotated[str, typer.Argument(help="Query text")],
    db: Annotated[str, typer.Option("--db", help="LanceDB path")] = "./embedify.lancedb",
    table: Annotated[str, typer.Option("--table")] = "chunks",
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 5,
    config: Annotated[str | None, typer.Option("--config", "-c")] = None,
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
) -> None:
    """Search a LanceDB store (search: input_type=query)."""
    cfg = _load_config(config)
    if backend or model or base_url:
        ecfg = cfg.embedders.setdefault("text", EmbedderConfig())
        if backend:
            ecfg.backend = backend
        if model:
            ecfg.model = model
        if base_url:
            ecfg.base_url = base_url
    from embedify.errors import SinkError

    try:
        results = _query(text, db=db, table=table, top_k=top_k, config=cfg)
    except SinkError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps([
        {"score": round(score, 4), "source_uri": r.source_uri, "locator": r.locator.to_dict(),
         "model": r.model, "text_preview": r.text_preview}
        for r, score in results
    ], indent=2))


@app.command(name="query-example")
def query_example(
    source: Annotated[str, typer.Argument(help="Example source file (e.g. a series CSV)")],
    db: Annotated[str, typer.Option("--db", help="LanceDB path")] = "./embedify.lancedb",
    table: Annotated[str, typer.Option("--table")] = "chunks",
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 5,
    start: Annotated[str | None, typer.Option(
        "--start", help="ISO start of example window")] = None,
    end: Annotated[str | None, typer.Option("--end", help="ISO end of example window")] = None,
    config: Annotated[str | None, typer.Option("--config", "-c")] = None,
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
) -> None:
    """Find stored windows similar to a window cut from SOURCE."""
    from embedify.api import query_by_example as _qbe
    from embedify.errors import EmbedderError, SinkError

    cfg = _load_config(config)
    if backend or model or base_url:
        ecfg = cfg.embedders.setdefault("text", EmbedderConfig())
        if backend:
            ecfg.backend = backend
        if model:
            ecfg.model = model
        if base_url:
            ecfg.base_url = base_url
    try:
        results = _qbe(source, db=db, table=table, top_k=top_k, start=start, end=end, config=cfg)
    except (SinkError, EmbedderError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps([
        {"score": round(score, 4), "source_uri": r.source_uri, "locator": r.locator.to_dict(),
         "model": r.model, "text_preview": r.text_preview}
        for r, score in results
    ], indent=2))


@app.command()
def inspect(path: Annotated[str, typer.Argument(help="File to detect")]) -> None:
    """Detect modality/format without embedding."""
    det = Detector().detect(Path(path))
    typer.echo(json.dumps({"modality": str(det.modality), "mime_type": det.mime_type,
                           "language": det.language, "detail": det.detail,
                           "source_uri": det.source_uri}, indent=2))


@app.command()
def models() -> None:
    """List known embedder backends/models and their input_type behaviour."""
    from embedify.config import DEFAULT_LOCAL_MODEL, FAST_LOCAL_MODEL, MULTILINGUAL_MODEL
    from embedify.embedders import KNOWN_PREFIXES

    typer.echo(json.dumps({
        "default_local": DEFAULT_LOCAL_MODEL,
        "fastest_local": FAST_LOCAL_MODEL,
        "multilingual": MULTILINGUAL_MODEL,
        "backends": ["local (sentence-transformers)", "endpoint (OpenAI-compatible /v1/embeddings)",
                     "hash (tests/offline)"],
        "input_type_prefixes": {k: {"query": v[0], "document": v[1]}
                                for k, v in KNOWN_PREFIXES.items()},
        "note": "Vectors are only comparable within the same model+model_version+dim.",
    }, indent=2))


if __name__ == "__main__":
    app()
