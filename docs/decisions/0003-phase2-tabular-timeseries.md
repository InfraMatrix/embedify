# ADR 0003: Phase 2 tabular & time-series design

Date: 2026-10-27 · Status: accepted · Phase 2

## Decision
- Parsers are stdlib-first (`csv`, `json`, `sqlite3`); Parquet/Excel are
  optional extras (`pyarrow`/`openpyxl`). Parsers yield row dicts.
- A table with a parseable timestamp column + numeric columns is
  reclassified at load time from `tabular` to `timeseries` point units;
  chunking/embedding route on the *unit* modality, not the file detection.
- Tabular granularity is a chunker concern: `row` (batched, default),
  `column`, or `table` units. Default strategy is serialize-to-text;
  `strategy: native` leaves per-row units untouched for embedders that
  implement the optional `embed_units(units)` hook instead of `embed_batch(texts)`.
- Time series: sliding windows (`window_size`/`stride`) over per-series
  sorted points, global per-channel normalization (`zscore` default),
  irregular sampling detected from timestamp deltas and flagged in
  metadata — never silently resampled.
- `backend: stats` is the lightweight baseline embedder: per-channel
  mean/std/min/max/first/last/slope features from `Unit.numeric_values`,
  fixed-dim, deterministic, no downloads. Foundation-model encoders use
  the same window units via their text serialization.

## Consequences
All stored vectors remain L2-normalized; LanceDB search uses the cosine
metric and reports `score = 1 - distance` (higher = better).
Remote SQL databases (Postgres etc.) arrive with the Phase 4 pgvector work;
Phase 2 SQL support is local SQLite files.
