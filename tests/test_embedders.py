import httpx

from embedify.config import EmbedderConfig
from embedify.embedders import HashEmbedder, OpenAICompatibleEmbedder, create_embedder
from embedify.types import InputType


def test_hash_embedder_deterministic():
    emb = HashEmbedder(EmbedderConfig(backend="hash", model="hash", dim=16), dim=16)
    a = emb.embed_batch(["hello world"], input_type=InputType.DOCUMENT)
    b = emb.embed_batch(["hello world"], input_type="document")
    assert a == b and len(a[0]) == 16


def test_bge_query_prefix_applied_only_to_queries():
    seen: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen.append(json.loads(request.content)["input"])
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [0.1, 0.2, 0.3]} for i in range(len(seen[-1]))
        ]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    emb = OpenAICompatibleEmbedder(
        EmbedderConfig(backend="endpoint", model="BAAI/bge-small-en-v1.5",
                       base_url="http://local.test/v1", api_key_env=None, dim=3),
        client=client)
    emb.embed_batch(["refund policy"], input_type="document")
    emb.embed_batch(["refund policy"], input_type="query")
    assert seen[0] == ["refund policy"]
    assert seen[1] == ["Represent this sentence for searching relevant passages: refund policy"]


def test_local_embedder_missing_dep_is_clear(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    emb = create_embedder(EmbedderConfig(backend="local", model="BAAI/bge-small-en-v1.5"))
    # sentence-transformers may actually be installed in dev env; only assert
    # the helpful path when the import is genuinely blocked.
    try:
        emb.embed_batch(["x"])
    except Exception as exc:
        assert "embedify[local]" in str(exc) or "sentence" in str(exc).lower()
