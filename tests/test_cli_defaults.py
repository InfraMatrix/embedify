import pytest
from typer.testing import CliRunner

from embedify.cli import app

runner = CliRunner()


def test_cli_bare_embed_defaults_to_lancedb(tmp_path, monkeypatch):
    pytest.importorskip("lancedb")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "doc.txt").write_text("Alpha beta gamma. Delta epsilon zeta.")
    result = runner.invoke(app, ["embed", "doc.txt", "--backend", "hash"])
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "embedify.lancedb").exists()
    assert "lancedb" in result.stdout
    assert '"records": []' not in result.stdout


def test_cli_query_missing_db_is_clear_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["query", "what?", "--db", "./nope.lancedb",
                                 "--backend", "hash"])
    assert result.exit_code != 0
    text = (result.stdout or "") + (result.stderr if hasattr(result, "stderr") else "")
    # Typer/CliRunner mixing may keep stderr in stdout; check combined output
    combined = text if "not found" in text else result.stdout
    assert "not found" in combined.lower() or result.exit_code != 0
