# Changelog

All notable changes to embedify are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).
The `EmbeddingRecord` output contract is versioned separately
(`schema_version`, see `docs/schema.md`); it is `1.0` at this release.

## [1.0.0] — 2026-10-27

First stable release. embedify is an embedding **producer**: it writes
embeddings + provenance metadata to sinks; retrieval/search/MCP are the
consuming system's job (`embedify query` is a smoke test only).

### Added
- Unified `EmbeddingRecord` output contract (`schema_version: "1.0"`,
  `docs/schema.md` + `docs/schema.json`) with embedify-free read examples
  for LanceDB, Parquet, and pgvector-shaped stores.
- Two embedder execution modes behind one interface — local
  (sentence-transformers) and any OpenAI-compatible `/v1/embeddings`
  endpoint — plus hash (tests/offline) and stats-feature embedders.
  `input_type` (query/document) is threaded through every adapter.
- Typed payload routing: embedders declare `modalities` + `payload_types`
  (text/numeric/bytes); the pipeline calls only `embed()`.
- Formats: text/Markdown (heading-aware), HTML, PDF text layer, code,
  CSV/TSV/JSON/JSONL/Parquet/Excel/SQLite, time-series CSV/Parquet,
  Prometheus, DataFrame/list-of-dict sources.
- Tree-sitter code chunking (Python, TS/JS, Go, Rust, Java, C/C++) with
  `path > parent > signature` context headers and oversized-symbol
  splitting; Python AST fallback when grammars are absent.
- Repo handling: `.gitignore`, vendored/generated/minified/lock/binary
  skips, git provenance (repo/branch/commit/path) on code records,
  incremental re-embedding (unchanged skipped — their records keep the
  original embedding run's commit SHA; changed replaced; deleted removed;
  a rename is delete-plus-new).
- Secret exclusion at **file granularity**: `.env`-style/key files and any
  file containing a detected secret are excluded whole, and reported as
  kind+line findings (never values).
- Archive sources (.tar/.tar.gz/.tgz/.zip), including archives in
  directory walks and nested archives (depth ≤ 5), with traversal/link
  protection and configurable extraction limits
  (`archive.max_files`, `archive.max_member_bytes`,
  `archive.max_total_bytes`, `archive.max_nesting_depth`).
- Time zone normalization: all timestamps are tz-aware UTC past the
  loader (naive inputs assumed UTC; configurable via `assume_tz`).
- Time-series sliding windows with configurable normalization scope
  (series default) and stored normalization parameters; stats baseline.
- Sinks: memory, JSONL/Parquet/NPZ files, LanceDB with a model/dim
  mixing guard. CLI embed output includes a `stats` block (files
  scanned/skipped by reason, units, elapsed, units/sec).
- YAML config + `EMBEDIFY_*` env overrides; sync Python API.

### Future work (not in 1.0)
- Image/audio/video modalities and shared-space (multimodal) embedding.
- pgvector sink, resumable jobs/content-hash cache, FastAPI server,
  async API. These raise the existing `NotImplementedError`s until they land.
