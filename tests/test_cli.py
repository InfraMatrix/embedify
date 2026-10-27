from typer.testing import CliRunner

from embedify.cli import app

runner = CliRunner()


def test_cli_inspect(fixtures_dir):
    result = runner.invoke(app, ["inspect", str(fixtures_dir / "sample.txt")])
    assert result.exit_code == 0
    assert "text" in result.stdout


def test_cli_models():
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "bge-small-en-v1.5" in result.stdout


def test_cli_embed_hash_memory(tmp_path):
    (tmp_path / "doc.txt").write_text("The quick brown fox jumps over the lazy dog.")
    result = runner.invoke(app, [
        "embed", str(tmp_path / "doc.txt"), "--backend", "hash", "--sink", "memory",
    ])
    assert result.exit_code == 0, result.stdout
    assert "units_embedded" in result.stdout


def test_cli_embed_output_includes_stats_block(tmp_path):
    import json

    (tmp_path / "doc.txt").write_text("stats please, one document")
    (tmp_path / ".env").write_text("API_KEY=whatever123456\n")
    result = runner.invoke(app, [
        "embed", str(tmp_path), "--backend", "hash", "--sink", "memory",
    ])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    stats = payload["stats"]
    assert stats["files_scanned"] >= 2
    assert stats["units_embedded"] >= 1
    assert stats["elapsed_seconds"] >= 0
    assert stats["units_per_second"] > 0
    assert stats["skipped_by_reason"].get("secret-file") == 1
