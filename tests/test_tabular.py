import sqlite3

import pytest

import embedify
from embedify.config import ChunkerConfig, EmbedderConfig, EmbedifyConfig, SinkConfig


def _cfg(**chunker_kw):
    return EmbedifyConfig(
        embedders={"text": EmbedderConfig(backend="hash", model="hash", dim=32)},
        chunkers={
            "tabular": ChunkerConfig(**chunker_kw),
            "timeseries": ChunkerConfig(**chunker_kw),
        },
        sink=SinkConfig(type="memory"),
    )


def test_tabular_csv_row_granularity(fixtures_dir, hash_embedder):
    cfg = _cfg(granularity="row", rows_per_chunk=2)
    result = embedify.embed(fixtures_dir / "table.csv", config=cfg, embedder=hash_embedder)
    assert result.stats.files_failed == 0
    assert len(result.records) == 2
    assert result.records[0].metadata["row_ids"] == ["1", "2"]
    assert result.records[0].locator.row_id == "1:2"
    assert "alice" in result.records[0].text_preview


def test_tabular_column_and_table_granularity(fixtures_dir, hash_embedder):
    cols = embedify.embed(fixtures_dir / "table.csv", config=_cfg(granularity="column"),
                          embedder=hash_embedder)
    assert {r.metadata["column"] for r in cols.records} == {"id", "name", "role", "score"}
    table = embedify.embed(fixtures_dir / "table.csv", config=_cfg(granularity="table"),
                           embedder=hash_embedder)
    assert len(table.records) == 1
    assert table.records[0].metadata["row_count"] == 4


def test_tabular_jsonl_and_dataframe_source(fixtures_dir, hash_embedder):
    result = embedify.embed(fixtures_dir / "rows.jsonl", config=_cfg(rows_per_chunk=10),
                            embedder=hash_embedder)
    assert len(result.records) == 1
    assert "bergen" in result.records[0].text_preview
    inline = embedify.embed([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], config=_cfg(),
                            embedder=hash_embedder)
    assert inline.records and inline.records[0].source_uri == "<dataframe>"


def test_tabular_parquet_roundtrip(tmp_path, hash_embedder):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa
    src = tmp_path / "t.parquet"
    pq.write_table(pa.table({"id": [1, 2], "v": ["a", "b"]}), src)
    result = embedify.embed(src, config=_cfg(), embedder=hash_embedder)
    assert len(result.records) == 1


def test_tabular_sqlite(tmp_path, hash_embedder):
    db = tmp_path / "data.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO users VALUES (?,?)", [(1, "ada"), (2, "grace")])
    conn.commit()
    conn.close()
    result = embedify.embed(db, config=_cfg(), embedder=hash_embedder)
    assert result.records and "ada" in result.records[0].text_preview
    assert result.records[0].metadata["table"] == "users"


def test_tabular_excel_if_available(tmp_path, hash_embedder):
    pytest.importorskip("openpyxl")
    from openpyxl import Workbook
    src = tmp_path / "book.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["id", "city"])
    ws.append([1, "oslo"])
    wb.save(src)
    result = embedify.embed(src, config=_cfg(), embedder=hash_embedder)
    assert result.records and "oslo" in result.records[0].text_preview
