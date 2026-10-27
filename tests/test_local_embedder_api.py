import sys
import types as _types

from embedify.config import EmbedderConfig
from embedify.embedders import LocalEmbedder


class _FakeModelNewAPI:
    def get_embedding_dimension(self):  # new name only; old name must not be called
        return 3

    def get_sentence_embedding_dimension(self):
        raise AssertionError("old API should not be used when new one exists")

    def encode(self, texts, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_local_embedder_prefers_get_embedding_dimension(monkeypatch):
    fake = _types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = lambda model: _FakeModelNewAPI()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    emb = LocalEmbedder(EmbedderConfig(backend="local", model="fake-model"))
    vecs = emb.embed_batch(["hello"])
    assert emb.dim == 3
    assert vecs == [[0.1, 0.2, 0.3]]
