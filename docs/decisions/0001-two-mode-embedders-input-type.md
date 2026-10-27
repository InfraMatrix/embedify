# ADR 0001: Two embedder execution modes, one interface, `input_type` from day one

Date: 2026-10-27 · Status: accepted · Phase 1

## Context
Users want to either (a) launch a model locally, or (b) send embedding
workloads to an endpoint they run/pay for and receive vectors back for RAG.
Asymmetric models (BGE, E5, Cohere, Voyage) treat queries and documents
differently; retrofitting that distinction would break the pipeline API.

## Decision
- `BaseEmbedder.embed_batch(texts, input_type="document"|"query")` is the only contract.
- Backends: `local` (sentence-transformers, default `BAAI/bge-small-en-v1.5`),
  `endpoint` (any OpenAI-compatible `/v1/embeddings`: OpenAI, Ollama, vLLM, TEI,
  LM Studio), `hash` (deterministic, tests/offline only).
- Adapters map `input_type` → model-specific behaviour: BGE query instruction
  prefix, E5 `query:`/`passage:` prefixes, explicit `query_prefix`/`document_prefix`
  config overrides. The OpenAI-compatible API has no such parameter, so the
  endpoint adapter applies the same prefix mapping locally. Cohere/Voyage native
  `input_type` parameters become plugins later with no pipeline change.
- Pipeline passes `document` when indexing; `embedify query` passes `query`.

## Consequences
Vectors are only comparable within the same `model + model_version + dim`.
Every `EmbeddingRecord` stamps all three so RAG indexes can filter/enforce it.
