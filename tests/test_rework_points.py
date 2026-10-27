"""Points 4-7: query-by-example, model/dim guard, serialization, streaming."""

import pytest

import embedify
from embedify.config import ChunkerConfig, EmbedderConfig, EmbedifyConfig, SinkConfig
from embedify.errors import SinkError
from embedify.sinks import LanceDBSink
from embedify.tabular import iter_row_stream, serialize_row


def _stats_config(db: str, table: str = "chunks") -> EmbedifyConfig:
    return EmbedifyConfig(
        embedders={"timeseries": EmbedderConfig(backend="stats", model="stats-features", dim=32)},
        chunkers={"timeseries": ChunkerConfig(window_size=4, stride=2)},
        sink=SinkConfig(type="lancedb", path=db, table=table),
    )


# Point 4 -----------------------------------------------------------------
def test_query_by_example_returns_similar_windows_excluding_example(tmp_path, fixtures_dir):
    pytest.importorskip("lancedb")
    db = str(tmp_path / "ts.lancedb")
    cfg = _stats_config(db)
    result = embedify.embed(fixtures_dir / "series.csv", config=cfg,
                            sink=LanceDBSink(db, "chunks"))
    assert result.records == []  # non-memory sink: records live in LanceDB
    assert not result.errors
    hits = embedify.query_by_example(
        fixtures_dir / "series.csv", db=db, table="chunks", top_k=2,
        start="2024-01-01T00:00:00", end="2024-01-01T00:03:00", config=cfg)
    assert hits, "expected similar windows"
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    # The example window (00:00-00:03) itself must not be returned.
    for record, _ in hits:
        assert not (record.locator.time_start and record.locator.time_end
                    and record.locator.time_start.isoformat().startswith("2024-01-01T00:00")
                    and record.locator.time_end.isoformat().startswith("2024-01-01T00:03"))


# Point 5 -----------------------------------------------------------------
def test_lancedb_guard_refuses_mixing_models_but_allows_separate_tables(tmp_path, fixtures_dir):
    pytest.importorskip("lancedb")
    db = str(tmp_path / "guard.lancedb")
    text_cfg = EmbedifyConfig(
        embedders={"text": EmbedderConfig(backend="hash", model="hash", dim=32)},
        sink=SinkConfig(type="lancedb", path=db, table="chunks"))
    embedify.embed(fixtures_dir / "sample.txt", config=text_cfg, sink=LanceDBSink(db, "chunks"))

    stats_cfg = _stats_config(db)
    with pytest.raises(SinkError, match="Refusing to mix models"):
        embedify.embed(fixtures_dir / "series.csv", config=stats_cfg,
                       sink=LanceDBSink(db, "chunks"))
    # A separate table is the sanctioned separation and works.
    ok = embedify.embed(fixtures_dir / "series.csv", config=stats_cfg,
                        sink=LanceDBSink(db, "stats-features"))
    assert not ok.errors
    assert embedify.query_by_example(fixtures_dir / "series.csv", db=db,
                                     table="stats-features", top_k=1, config=stats_cfg)


def test_lancedb_search_rejects_wrong_dim_vector(tmp_path, fixtures_dir, hash_config):
    pytest.importorskip("lancedb")
    db = str(tmp_path / "dim.lancedb")
    embedify.embed(fixtures_dir / "sample.txt", config=hash_config, sink=LanceDBSink(db, "chunks"))
    with pytest.raises(SinkError, match="does not match stored dim"):
        LanceDBSink(db, "chunks").search([0.1, 0.2], top_k=1)


# Point 6 -----------------------------------------------------------------
def test_serialize_row_format_include_exclude_and_truncation():
    row = {"id": 1, "name": "alice", "note": "x" * 500, "secret": "hide"}
    text, truncated = serialize_row(row, list(row), max_value_length=10)
    assert text.startswith("id: 1 | name: alice")
    assert "note: " + "x" * 10 + "…" in text
    assert truncated is True
    text2, truncated2 = serialize_row(row, list(row), include=["name"])
    assert text2 == "name: alice" and truncated2 is False
    text3, _ = serialize_row(row, list(row), exclude=["secret", "note"], max_value_length=600)
    assert "secret" not in text3 and "note" not in text3


def test_tabular_chunker_applies_serialization_config(fixtures_dir):
    from embedify.chunkers import TabularChunker
    from embedify.detector import Detector
    from embedify.loaders import TabularLoader

    units = list(TabularLoader().load(fixtures_dir / "table.csv",
                                      Detector().detect(fixtures_dir / "table.csv")))
    chunks = list(TabularChunker(ChunkerConfig(
        include_columns=["name", "role"], max_value_length=3)).chunk(units))
    assert chunks
    assert "name: ali…" in chunks[0].content and "score:" not in chunks[0].content
    ser = chunks[0].metadata["serialization"]
    assert ser["include_columns"] == ["name", "role"] and ser["max_value_length"] == 3
    assert ser["truncated"] is True


# Point 7 -----------------------------------------------------------------
def test_iter_row_stream_is_lazy_and_yields_rows(fixtures_dir):
    stream = iter_row_stream(fixtures_dir / "table.csv")
    assert iter(stream) is stream  # a generator: nothing read until iterated
    rows = list(stream)
    assert len(rows) == 4
    table, columns, first = rows[0]
    assert table == "table" and columns == ["id", "name", "role", "score"]
    assert first["name"] == "alice"


def test_iter_row_stream_parquet_uses_batched_reader(fixtures_dir, tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}), path)
    rows = list(iter_row_stream(path))
    assert [r[2]["a"] for r in rows] == [1, 2, 3]
