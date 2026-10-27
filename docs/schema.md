# EmbeddingRecord output contract — schema version 1.0

embedify is a **producer**: it writes `EmbeddingRecord`s to a sink and stops
there. Retrieval, search, reranking, and MCP servers are the consuming
system's job. This document is the contract those consumers code against;
it is stable within a `schema_version` (machine-readable form:
[`schema.json`](schema.json), JSON Schema draft 2020-12).

## Fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Contract version, currently `"1.0"` |
| `id` | string | `uuid5(source_uri + ":" + content_hash)` — stable for same source+content |
| `vector` | float[] | **L2-normalized** embedding; `len(vector) == dim` |
| `model` | string | Model id that produced the vector |
| `model_version` | string | Model revision/version reported by the adapter |
| `dim` | int | Vector dimensionality |
| `modality` | string | `text` \| `code` \| `document` \| `tabular` \| `timeseries` \| `image` \| `audio` \| `video` \| `unknown` |
| `source_uri` | string | File path / URI the unit came from |
| `locator` | object | Exact source span (see below) |
| `content_hash` | string | SHA-256 (hex) of the exact unit content that was embedded |
| `text_preview` | string | First 500 chars of the embedded text |
| `metadata` | object | Modality-specific metadata (see below) |
| `created_at` | string | ISO-8601 timestamp (UTC) |

**Comparability rule:** vectors are only comparable within the same
`model` + `model_version` + `dim`. embedify's LanceDB sink refuses to mix
models in one table; consumers must apply the same rule to their own stores.

**Locator:** only one family is populated per record — `page` (documents),
`line_start`/`line_end` + `char_start`/`char_end` (text/code), `row_id`
(tabular), `time_start`/`time_end` (time series), `frame_ts` (video).
Line/char numbers are 1-based lines / 0-based chars.

**Metadata, code records** (in addition to the generic keys):
`repo`, `branch`, `commit` (git provenance; null outside a repo), `path`
(repo-relative path), `language`, `symbol_name`, `symbol_kind`
(`function|method|class|interface|struct|enum|trait|impl|namespace|type|module_header`),
`signature`, `parent` (enclosing class/impl, or null), `context_header`
(the `"<path> > <parent> > <signature>"` line prepended to the embedded text).

**Sink encodings:** in LanceDB and Parquet, `locator` and `metadata` are
stored as JSON **strings** (all other fields are native columns); in JSONL
they are nested objects. `created_at` may come back as a timestamp type
from columnar stores — parse it accordingly.

## Reading records WITHOUT importing embedify

### LanceDB (Python)

```python
import json
import lancedb

table = lancedb.connect("./embedify.lancedb").open_table("chunks")
rows = table.to_arrow().to_pylist()          # every record, as dicts
for row in rows:
    locator = json.loads(row["locator"])     # JSON string -> dict
    metadata = json.loads(row["metadata"])
    vector = row["vector"]                   # list[float], unit-norm

# nearest neighbours with the raw LanceDB API (cosine: embedify stores
# unit-norm vectors; _distance returned by LanceDB is 1 - similarity):
hits = table.search(my_query_vector).metric("cosine").limit(5).to_list()
score = 1.0 - hits[0]["_distance"]
```

### Parquet (pyarrow / pandas)

```python
import json
import pyarrow.parquet as pq

table = pq.read_table("out.parquet")
rows = table.to_pylist()
locator = json.loads(rows[0]["locator"])
df = table.to_pandas()                       # vectors arrive as numpy arrays
```

### pgvector

embedify's pgvector sink lands in a later phase; consumers loading
records themselves (from JSONL/Parquet export) use this shape:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE chunks (
    id            text PRIMARY KEY,
    vector        vector(384),          -- must match the records' `dim`
    model         text NOT NULL,
    model_version text NOT NULL,
    dim           int  NOT NULL,
    modality      text NOT NULL,
    source_uri    text NOT NULL,
    locator       jsonb NOT NULL,
    content_hash  text NOT NULL,
    text_preview  text NOT NULL,
    metadata      jsonb NOT NULL DEFAULT '{}',
    created_at    timestamptz NOT NULL
);
CREATE INDEX ON chunks USING hnsw (vector vector_cosine_ops);

-- nearest neighbours (cosine distance `<=>`; similarity = 1 - distance):
SELECT id, source_uri, text_preview, 1 - (vector <=> $1::vector) AS score
FROM chunks
WHERE model = $2 AND dim = $3           -- respect the comparability rule
ORDER BY vector <=> $1::vector
LIMIT 5;
```

Insert with any Postgres driver (psycopg, asyncpg, …) directly from the
JSONL/Parquet export — no embedify import is involved at read time.

## Versioning

- Additive changes (new optional metadata keys, new modalities) stay
  within `1.x` and bump the minor component.
- Removing/renaming fields or changing semantics bumps the major
  component; records written by older majors remain readable via the
  field table of their own version of this document (git tags).
