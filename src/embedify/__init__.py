"""embedify — embed any data for RAG and agentic systems.

Quickstart::

    import embedify
    result = embedify.embed("docs/")                      # document input_type
    hits = embedify.query("how do refunds work?", db="./embedify.lancedb")  # query input_type
"""

from embedify.api import aembed, embed, query, query_by_example
from embedify.config import EmbedifyConfig
from embedify.detector import Detector, detect
from embedify.embedders import BaseEmbedder, HashEmbedder, StatsFeatureEmbedder, create_embedder
from embedify.sinks import LanceDBSink, MemorySink
from embedify.types import (
    SCHEMA_VERSION,
    BytesInput,
    EmbeddingRecord,
    InputPayload,
    InputType,
    JobResult,
    Locator,
    Modality,
    NumericInput,
    SecretFinding,
    TextInput,
    Unit,
)

__version__ = "1.0.0"

__all__ = [
    "SCHEMA_VERSION",
    "BaseEmbedder",
    "BytesInput",
    "Detector",
    "EmbeddingRecord",
    "EmbedifyConfig",
    "HashEmbedder",
    "InputPayload",
    "InputType",
    "JobResult",
    "LanceDBSink",
    "Locator",
    "MemorySink",
    "Modality",
    "NumericInput",
    "SecretFinding",
    "StatsFeatureEmbedder",
    "TextInput",
    "Unit",
    "__version__",
    "aembed",
    "create_embedder",
    "detect",
    "embed",
    "query",
    "query_by_example",
]
