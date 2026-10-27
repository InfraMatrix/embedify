"""Point 5: versioned EmbeddingRecord output contract."""

import json
from pathlib import Path

import embedify
from embedify.types import SCHEMA_VERSION

DOCS = Path(__file__).parent.parent / "docs"


def test_record_carries_schema_version(fixtures_dir, hash_config):
    result = embedify.embed(fixtures_dir / "sample.txt", config=hash_config)
    rec = result.records[0].to_dict()
    assert rec["schema_version"] == SCHEMA_VERSION == "1.0"
    assert len(rec["vector"]) == rec["dim"]


def test_json_schema_matches_record_shape(fixtures_dir, hash_config):
    schema = json.loads((DOCS / "schema.json").read_text())
    result = embedify.embed(fixtures_dir / "sample.txt", config=hash_config)
    rec = result.records[0].to_dict()
    assert set(rec) == set(schema["properties"])
    assert set(schema["required"]) == set(rec)
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert "locator" in rec and "metadata" in rec


def test_schema_doc_covers_all_sinks():
    doc = (DOCS / "schema.md").read_text()
    for needle in ("LanceDB", "pgvector", "Parquet", "WITHOUT importing embedify",
                   "schema_version", "comparable"):
        assert needle in doc
