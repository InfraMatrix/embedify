"""Embedders: local models and hosted endpoints behind one interface.

The interface accepts an ``input_type`` hint (``"query"`` | ``"document"``)
from day one. Adapters map it to whatever the model needs:

- BGE models: query instruction prefix on queries only (local + compatible endpoints)
- explicit ``query_prefix`` / ``document_prefix`` config overrides everything
- Cohere/Voyage native ``input_type`` params: future plugins (Phase 3+), the
  hint is already threaded through, so no pipeline change will be needed.

Two execution modes, one output shape:

- ``local``    — sentence-transformers model loaded in-process
- ``endpoint`` — any OpenAI-compatible ``/v1/embeddings`` server
  (OpenAI, Ollama, vLLM, TEI, LM Studio all expose this)
- ``hash``     — deterministic hash embedder for tests/offline smoke runs
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from embedify.config import EmbedderConfig
from embedify.errors import ConfigError, EmbedderError
from embedify.types import InputPayload, InputType, Modality, NumericInput, TextInput

# Known asymmetric models: (query_prefix, document_prefix)
KNOWN_PREFIXES: dict[str, tuple[str, str]] = {
    "bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "bge-large-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "multilingual-e5-small": ("query: ", "passage: "),
    "multilingual-e5-base": ("query: ", "passage: "),
    "multilingual-e5-large": ("query: ", "passage: "),
    "e5-small-v2": ("query: ", "passage: "),
    "e5-base-v2": ("query: ", "passage: "),
    "e5-large-v2": ("query: ", "passage: "),
}


def resolve_prefixes(config: EmbedderConfig) -> tuple[str, str]:
    if config.query_prefix is not None or config.document_prefix is not None:
        return (config.query_prefix or "", config.document_prefix or "")
    model_leaf = config.model.rstrip("/").split("/")[-1].lower()
    return KNOWN_PREFIXES.get(model_leaf, ("", ""))


def apply_input_type(texts: Sequence[str], config: EmbedderConfig,
                     input_type: InputType | str) -> list[str]:
    q_prefix, d_prefix = resolve_prefixes(config)
    prefix = q_prefix if InputType(input_type) == InputType.QUERY else d_prefix
    return [f"{prefix}{t}" for t in texts] if prefix else list(texts)


class BaseEmbedder(Protocol):
    """The embedder contract (reworked before Phase 3).

    Embedders *declare* what they accept — ``modalities`` and
    ``payload_types`` (``"text"`` | ``"numeric"`` | ``"bytes"``) — and the
    pipeline routes each unit to them as a typed payload via
    ``Unit.payload_for(embedder.payload_types)``. ``embed_batch`` remains
    as the text-only convenience used by direct callers/tests.
    """

    modalities: set[Modality]
    payload_types: set[str]
    dim: int
    max_input: int
    max_batch: int
    model_id: str
    model_version: str

    def embed(self, payloads: Sequence[InputPayload], *,
              input_type: InputType | str = InputType.DOCUMENT) -> list[list[float]]: ...

    def embed_batch(self, texts: Sequence[str], *,
                    input_type: InputType | str = InputType.DOCUMENT) -> list[list[float]]: ...


def _require_text(payloads: Sequence[InputPayload], who: str) -> list[str]:
    texts: list[str] = []
    for p in payloads:
        if not isinstance(p, TextInput):
            raise EmbedderError(
                f"{who} accepts text payloads only, got '{p.kind}'. "
                "Route this modality to an embedder that accepts it."
            )
        texts.append(p.text)
    return texts


class HashEmbedder:
    """Deterministic bag-of-hashed-tokens embedder. Tests/offline only."""

    def __init__(self, config: EmbedderConfig | None = None, dim: int = 64) -> None:
        cfg = config or EmbedderConfig(backend="hash", model="hash")
        self.config = cfg
        self.dim = cfg.dim or dim
        self.modalities = set(Modality)
        self.payload_types = {"text", "numeric"}
        self.max_input = 100_000
        self.max_batch = cfg.batch_size
        self.model_id = "hash"
        self.model_version = "1"

    def embed(self, payloads: Sequence[InputPayload], *,
              input_type: InputType | str = InputType.DOCUMENT) -> list[list[float]]:
        texts: list[str] = []
        for p in payloads:
            if isinstance(p, TextInput):
                texts.append(p.text)
            elif isinstance(p, NumericInput):
                texts.append(" ".join(f"{v:.4g}" for row in p.values for v in row))
            else:
                raise EmbedderError("hash embedder does not accept bytes payloads")
        return self.embed_batch(texts, input_type=input_type)

    def embed_batch(self, texts: Sequence[str], *,
                    input_type: InputType | str = InputType.DOCUMENT) -> list[list[float]]:
        out: list[list[float]] = []
        for text in apply_input_type(texts, self.config, input_type):
            vec = [0.0] * self.dim
            for token in text.lower().split():
                idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class LocalEmbedder:
    """sentence-transformers model loaded in-process (``embedify[local]``)."""

    def __init__(self, config: EmbedderConfig) -> None:
        self.config = config
        self.modalities = {
            Modality.TEXT, Modality.CODE, Modality.DOCUMENT,
            Modality.TABULAR, Modality.TIMESERIES,  # via text serialization
        }
        self.payload_types = {"text"}
        self.max_batch = config.batch_size
        self.max_input = 8192
        self.model_id = config.model
        self.model_version = config.model_version
        self._model: Any = None
        self._dim: int | None = config.dim

    def embed(self, payloads: Sequence[InputPayload], *,
              input_type: InputType | str = InputType.DOCUMENT) -> list[list[float]]:
        return self.embed_batch(_require_text(payloads, "local embedder"), input_type=input_type)

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbedderError(
                    "Local models need sentence-transformers: pip install 'embedify[local]' "
                    "— or use backend: endpoint in config"
                ) from exc
            self._model = SentenceTransformer(self.config.model)
            # sentence-transformers >=3.x renamed the method; the old name
            # emits a FutureWarning. Prefer the new name, fall back for old.
            dim_fn = (
                getattr(self._model, "get_embedding_dimension", None)
                or getattr(self._model, "get_sentence_embedding_dimension", None)
            )
            if callable(dim_fn):
                try:
                    self._dim = int(dim_fn())
                except Exception:
                    self._dim = None
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        assert self._dim is not None
        return self._dim

    @dim.setter
    def dim(self, value: int) -> None:
        self._dim = value

    def embed_batch(self, texts: Sequence[str], *,
                    input_type: InputType | str = InputType.DOCUMENT) -> list[list[float]]:
        model = self._load()
        prepared = apply_input_type(texts, self.config, input_type)
        vectors = model.encode(prepared, batch_size=self.max_batch,
                               normalize_embeddings=True, show_progress_bar=False)
        result = [[float(x) for x in v] for v in vectors]
        if self._dim is None and result:
            self._dim = len(result[0])
        return result


class OpenAICompatibleEmbedder:
    """Any OpenAI-compatible ``/embeddings`` endpoint.

    Works with OpenAI, Ollama (``/v1``), vLLM, TEI, LM Studio.
    The standard API has no ``input_type`` param, so the hint is mapped to
    instruction prefixes (BGE/E5) — exactly as the local adapter does.
    """

    def __init__(self, config: EmbedderConfig, *,
                 client: httpx.Client | None = None) -> None:
        if not config.base_url:
            raise ConfigError("endpoint embedder requires base_url")
        self.config = config
        self.modalities = {
            Modality.TEXT, Modality.CODE, Modality.DOCUMENT,
            Modality.TABULAR, Modality.TIMESERIES,  # via text serialization
        }
        self.payload_types = {"text"}
        self.max_batch = config.batch_size
        self.max_input = 8192
        self.model_id = config.model
        self.model_version = config.model_version
        self._dim: int | None = config.dim
        self._client = client

    def embed(self, payloads: Sequence[InputPayload], *,
              input_type: InputType | str = InputType.DOCUMENT) -> list[list[float]]:
        return self.embed_batch(
            _require_text(payloads, "endpoint embedder"), input_type=input_type
        )

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Discover dimension with a 1-token probe rather than guessing.
            self._dim = len(self.embed_batch(["dimension probe"])[0])
        assert self._dim is not None
        return self._dim

    @dim.setter
    def dim(self, value: int) -> None:
        self._dim = value

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            key = os.environ.get(self.config.api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    def embed_batch(self, texts: Sequence[str], *,
                    input_type: InputType | str = InputType.DOCUMENT) -> list[list[float]]:
        prepared = apply_input_type(texts, self.config, input_type)
        client = self._client or httpx.Client(timeout=120.0)
        close = self._client is None
        base_url = self.config.base_url
        if not base_url:
            raise ConfigError("endpoint embedder requires base_url")
        url = f"{base_url.rstrip('/')}/embeddings"
        try:
            vectors: list[list[float]] = []
            for i in range(0, len(prepared), self.max_batch):
                batch = prepared[i : i + self.max_batch]
                try:
                    resp = client.post(url, headers=self._headers(),
                                       json={"model": self.config.model, "input": batch})
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    raise EmbedderError(f"Endpoint request failed ({url}): {exc}") from exc
                data = resp.json().get("data")
                if not isinstance(data, list) or len(data) != len(batch):
                    raise EmbedderError(
                        f"Unexpected embeddings response from {url}: {resp.text[:500]}"
                    )
                ordered = sorted(data, key=lambda d: d.get("index", 0))
                vectors.extend([float(x) for x in d["embedding"]] for d in ordered)
                if self._dim is None and vectors:
                    self._dim = len(vectors[0])
            return vectors
        finally:
            if close:
                client.close()


class StatsFeatureEmbedder:
    """Lightweight statistical-feature baseline (Phase 2).

    Consumes ``Unit.numeric_values`` (time-series windows / tabular rows)
    via the optional ``embed_units`` hook and produces fixed-dim feature
    vectors: per-channel mean/std/min/max/first/last/slope, flattened,
    truncated or zero-padded to ``dim``. No model download, no API key —
    a sanity baseline to compare foundation-model encoders against.
    """

    _FEATURES = 7

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        cfg = config or EmbedderConfig(backend="stats", model="stats-features")
        self.config = cfg
        self.dim = cfg.dim or 64
        self.modalities = {Modality.TIMESERIES, Modality.TABULAR}
        self.payload_types = {"numeric", "text"}
        self.max_input = 100_000
        self.max_batch = cfg.batch_size
        self.model_id = cfg.model
        self.model_version = cfg.model_version

    def embed(self, payloads: Sequence[InputPayload], *,
              input_type: InputType | str = InputType.DOCUMENT) -> list[list[float]]:
        matrices: list[list[list[float]] | None] = []
        texts: list[str] = []
        for p in payloads:
            if isinstance(p, NumericInput):
                matrices.append(p.values)
                texts.append("")
            elif isinstance(p, TextInput):
                matrices.append(None)
                texts.append(p.text)
            else:
                raise EmbedderError("stats embedder does not accept bytes payloads")
        out = self._embed_matrices(matrices)
        fallback = HashEmbedder(self.config, dim=self.dim)
        for i, matrix in enumerate(matrices):
            if matrix is None:
                out[i] = fallback.embed_batch([texts[i]], input_type=input_type)[0]
        return out

    def embed_units(self, units: Sequence[Any], *,
                    input_type: InputType | str = InputType.DOCUMENT) -> list[list[float]]:
        """Back-compat shim: pre-rework hook, now via typed payloads."""
        return self.embed(
            [u.payload_for(self.payload_types) for u in units], input_type=input_type
        )

    def _embed_matrices(
        self, matrices: Sequence[list[list[float]] | None]
    ) -> list[list[float]]:
        out: list[list[float]] = []
        for matrix in matrices:
            features: list[float] = []
            if matrix:
                n_channels = max(len(r) for r in matrix)
                for j in range(n_channels):
                    col = [float(r[j]) for r in matrix if j < len(r)]
                    if not col:
                        features.extend([0.0] * self._FEATURES)
                        continue
                    mean = sum(col) / len(col)
                    var = sum((x - mean) ** 2 for x in col) / len(col)
                    slope = (col[-1] - col[0]) / len(col) if len(col) > 1 else 0.0
                    features.extend([mean, var ** 0.5, min(col), max(col),
                                     col[0], col[-1], slope])
            vec = (features + [0.0] * self.dim)[: self.dim]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    def embed_batch(self, texts: Sequence[str], *,
                    input_type: InputType | str = InputType.DOCUMENT) -> list[list[float]]:
        # Text-only fallback: deterministic hash so accidental routing is visible in tests
        return HashEmbedder(self.config, dim=self.dim).embed_batch(texts, input_type=input_type)


def create_embedder(config: EmbedderConfig, *,
                    client: httpx.Client | None = None) -> BaseEmbedder:
    if config.backend == "local":
        return LocalEmbedder(config)
    if config.backend == "endpoint":
        return OpenAICompatibleEmbedder(config, client=client)
    if config.backend == "hash":
        return HashEmbedder(config)
    if config.backend == "stats":
        return StatsFeatureEmbedder(config)
    raise ConfigError(
        f"Unknown embedder backend '{config.backend}'. "
        "Native Cohere/Voyage adapters are planned as plugins; "
        "use 'local' or OpenAI-compatible 'endpoint' in Phase 1."
    )
