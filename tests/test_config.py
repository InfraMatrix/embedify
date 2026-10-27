from embedify.config import EmbedifyConfig


def test_config_defaults_are_bge_small():
    cfg = EmbedifyConfig()
    assert cfg.embedder_for("text").model == "BAAI/bge-small-en-v1.5"
    assert cfg.embedder_for("code").backend == "local"


def test_config_yaml_roundtrip(tmp_path):
    p = tmp_path / "embedify.yaml"
    p.write_text(
        "embedders:\n"
        "  text: {backend: endpoint, base_url: http://localhost:11434/v1, "
        "model: nomic-embed-text, api_key_env: null}\n"
        "chunkers:\n"
        "  text: {max_tokens: 256, overlap: 32}\n"
        "sink: {type: lancedb, path: ./db.lancedb, table: chunks}\n"
    )
    cfg = EmbedifyConfig.from_yaml(p)
    assert cfg.embedder_for("text").backend == "endpoint"
    assert cfg.chunker_for("text").max_tokens == 256
    assert cfg.sink.type == "lancedb"


def test_env_override(monkeypatch):
    monkeypatch.setenv("EMBEDIFY_SINK__TYPE", "file")
    cfg = EmbedifyConfig.load("/nonexistent.yaml") if False else EmbedifyConfig()
    cfg.apply_env_overrides()
    assert cfg.sink.type == "file"
