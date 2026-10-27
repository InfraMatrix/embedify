import pytest

pytest.importorskip("lancedb")
pytest.importorskip("pyarrow")

import embedify
from embedify.sinks import LanceDBSink


def test_lancedb_embed_then_query_separate_runs(tmp_path, fixtures_dir, hash_config, hash_embedder):
    db = tmp_path / "store.lancedb"
    sink = LanceDBSink(db, "chunks")
    result = embedify.embed(fixtures_dir / "sample.txt", config=hash_config,
                            embedder=hash_embedder, sink=sink)
    assert result.stats.units_embedded >= 1

    # Separate run simulation: fresh sink object, query path only
    hits = embedify.query("refunds purchase", db=db, config=hash_config,
                          embedder=hash_embedder)
    assert hits
    record, score = hits[0]
    assert record.source_uri.endswith("sample.txt")
    # score is cosine similarity: higher is better, bounded in [-1, 1]
    assert -1.0 <= score <= 1.0
    # vectors are stored unit-norm
    import math
    stored = embedify.query("embeddings vectors", db=db, config=hash_config,
                            embedder=hash_embedder)[0][0]
    assert abs(math.sqrt(sum(x * x for x in stored.vector)) - 1.0) < 1e-5
