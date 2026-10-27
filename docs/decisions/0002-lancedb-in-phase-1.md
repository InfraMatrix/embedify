# ADR 0002: Basic LanceDB sink in Phase 1

Date: 2026-10-27 · Status: accepted · Phase 1 (amends original phasing)

## Context
In-memory sinks cannot demonstrate the core loop `embed` (run 1) → `query`
(run 2), which is the entire point of a RAG ingestion tool.

## Decision
Ship a basic LanceDB sink in Phase 1 (append-or-create + vector search).
pgvector, content-hash caching, and resumable jobs stay in Phase 4 behind the
same `BaseSink` interface.

## Consequences
`embedify[lancedb]` is an optional extra; the sink raises a clear install
error when missing. LanceDB tables store `locator`/`metadata` as JSON strings
alongside flat columns so records round-trip losslessly.
