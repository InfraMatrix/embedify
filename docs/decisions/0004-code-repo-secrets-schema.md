# 0004 — Code chunking, repo handling, secrets, output contract

Status: accepted (2026-10-27). Scope clarification from the maintainer:
embedify **only produces embeddings + metadata into sinks**. Retrieval,
search and MCP are consuming systems' jobs; `embedify query` remains a
smoke test only. Media (former Phase 3) is deferred.

## Decisions

1. **Tree-sitter code chunking** behind the `embedify[code]` extra
   (grammar packages per language: Python, TS/JS, Go, Rust, Java, C/C++;
   extensible via `register_code_language`). Units are function / method /
   class-like / module-header. Embedded text carries the context header
   `"<path> > <parent> > <signature>"`; oversized symbols split line-wise
   with header + signature repeated. Without the extra, Python falls back
   to the existing AST chunker and other languages to token chunking —
   never a hard failure.

2. **Code records carry git provenance** (`repo`, `branch`, `commit`,
   `path` repo-relative) via `git` subprocess calls, plus `language`,
   `symbol_name`, `symbol_kind`, `signature`, `parent`. Line range is in
   the record `locator`. Provenance is best-effort: nulls outside a repo.

3. **Repo traversal** respects `.gitignore` (built-in matcher), and skips
   vendored/generated/minified/lock/binary files in directory walks.
   Explicitly named single files are always processed (except secrets).

4. **Secrets are excluded before embedding**: `.env`-style / key /
   credential files skipped wholesale; all other content is scanned with
   high-precision patterns. Findings are reported as
   `SecretFinding{source_uri, kind, line}` — never the secret value — in
   `JobResult.findings` and CLI output. The containing file is excluded.

5. **Incremental re-embedding** is opt-in (`incremental=True` /
   `--incremental`): chunk-hash sets per `source_uri` are compared with
   the sink's stored sets; unchanged files are skipped, changed files'
   records are deleted then replaced, and files deleted under an embedded
   directory are removed. Sinks opt in by implementing
   `stored_source_hashes()` / `delete_source()` (LanceDB, memory).

6. **Output contract is versioned**: every record carries
   `schema_version` (currently `"1.0"`), documented in `docs/schema.md`
   with a JSON Schema at `docs/schema.json`, including examples of reading
   LanceDB / Parquet / pgvector stores without importing embedify.
