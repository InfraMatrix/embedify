from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedify.config import EmbedderConfig, EmbedifyConfig, SinkConfig
from embedify.embedders import HashEmbedder


@pytest.fixture
def hash_embedder() -> HashEmbedder:
    return HashEmbedder(EmbedderConfig(backend="hash", model="hash", dim=32), dim=32)


@pytest.fixture
def hash_config() -> EmbedifyConfig:
    return EmbedifyConfig(
        embedders={"text": EmbedderConfig(backend="hash", model="hash", dim=32)},
        sink=SinkConfig(type="memory"),
    )


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
