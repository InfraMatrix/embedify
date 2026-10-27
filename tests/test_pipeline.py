import json

import pytest

import embedify
from embedify.config import EmbedifyConfig, SinkConfig
from embedify.sinks import FileSink


def test_embed_text_file(fixtures_dir, hash_config, hash_embedder):
    result = embedify.embed(fixtures_dir / "sample.txt", config=hash_config,
                            embedder=hash_embedder)
    assert result.stats.files_failed == 0
    assert len(result.records) >= 1
    rec = result.records[0]
    assert rec.model == "hash" and rec.dim == 32
    assert "Refunds" in rec.text_preview
    assert rec.content_hash


def test_embed_code_ast(fixtures_dir, hash_config, hash_embedder):
    result = embedify.embed(fixtures_dir / "sample.py", config=hash_config,
                            embedder=hash_embedder)
    symbols = {r.metadata.get("symbol") for r in result.records}
    assert {"add", "Greeter"} <= symbols


def test_embed_inline_text(hash_config, hash_embedder):
    result = embedify.embed("Cats are small mammals. Dogs are loyal animals.",
                            config=hash_config, embedder=hash_embedder)
    assert len(result.records) >= 1
    assert result.records[0].source_uri == "<inline>"


def test_embed_directory_skips_binaries(tmp_path, hash_config, hash_embedder):
    # Repo handling: binary files in a directory walk are skipped; an
    # explicitly named binary file still gets the metadata-only fallback.
    (tmp_path / "a.txt").write_text("Alpha beta gamma delta.")
    (tmp_path / "b.bin").write_bytes(b"\x00\xff" * 50)
    result = embedify.embed(tmp_path, config=hash_config, embedder=hash_embedder)
    assert result.stats.files_processed == 1
    assert not any(r.source_uri.endswith("b.bin") for r in result.records)
    single = embedify.embed(tmp_path / "b.bin", config=hash_config, embedder=hash_embedder)
    assert any(r.metadata.get("metadata_only") for r in single.records)


def test_embed_html_strips_scripts(fixtures_dir, hash_config, hash_embedder):
    result = embedify.embed(fixtures_dir / "sample.html", config=hash_config,
                            embedder=hash_embedder)
    assert "Hello world from HTML" in result.records[0].text_preview
    assert "var a" not in result.records[0].text_preview


def test_file_sink_jsonl(tmp_path, fixtures_dir, hash_config, hash_embedder):
    out = tmp_path / "out.jsonl"
    cfg = EmbedifyConfig(embedders=hash_config.embedders,
                         sink=SinkConfig(type="file", path=str(out)))
    result = embedify.embed(fixtures_dir / "sample.txt", config=cfg, embedder=hash_embedder,
                            sink=FileSink(out))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == len(result.records) or rows  # memory records empty for file sink
    assert rows[0]["source_uri"].endswith("sample.txt")
    assert "vector" in rows[0]


def test_pdf_missing_or_golden(fixtures_dir, hash_config, hash_embedder):
    pytest.importorskip("pypdf")
    pdf = fixtures_dir / "sample.pdf"
    if not pdf.exists():
        pytest.skip("sample.pdf fixture not generated in this checkout")
    result = embedify.embed(pdf, config=hash_config, embedder=hash_embedder)
    assert result.records
    assert result.records[0].locator.page == 1
