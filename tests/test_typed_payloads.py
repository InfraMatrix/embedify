"""Point 1: typed payloads, declared modalities, pipeline routing."""

import embedify
from embedify.config import EmbedderConfig
from embedify.embedders import HashEmbedder, LocalEmbedder, StatsFeatureEmbedder
from embedify.types import BytesInput, Modality, NumericInput, TextInput, Unit


def test_embedders_declare_modalities_and_payload_types():
    assert Modality.TEXT in LocalEmbedder(EmbedderConfig()).modalities
    assert LocalEmbedder(EmbedderConfig()).payload_types == {"text"}
    assert StatsFeatureEmbedder().payload_types == {"numeric", "text"}
    assert StatsFeatureEmbedder().modalities == {Modality.TABULAR, Modality.TIMESERIES}
    assert "bytes" in HashEmbedder().payload_types or "text" in HashEmbedder().payload_types


def test_unit_payload_for_routes_by_acceptance():
    unit = Unit(content="series summary", modality=Modality.TIMESERIES, source_uri="x",
                numeric_values=[[1.0, 2.0]], metadata={"channels": ["a", "b"]})
    numeric = unit.payload_for({"numeric"})
    assert isinstance(numeric, NumericInput) and numeric.channels == ("a", "b")
    text = unit.payload_for({"text"})
    assert isinstance(text, TextInput) and text.text == "series summary"
    img = Unit(content="[image]", modality=Modality.IMAGE, source_uri="x.png",
               content_bytes=b"\x89PNG", mime_type="image/png")
    assert isinstance(img.payload_for({"bytes"}), BytesInput)
    assert isinstance(img.payload_for({"text"}), TextInput)


def test_pipeline_rejects_undeclared_modality_with_clear_error(fixtures_dir, hash_config):
    # A stats embedder declares only tabular/timeseries; embedding text must
    # be reported as a modality error, not silently embedded.
    stats = StatsFeatureEmbedder(EmbedderConfig(backend="stats", model="stats-features", dim=16))
    result = embedify.embed(fixtures_dir / "sample.txt", config=hash_config, embedder=stats)
    assert result.records == []
    assert result.errors and "does not declare modality" in result.errors[0].error


def test_pipeline_uses_only_embed_not_embed_batch(fixtures_dir, hash_config):
    # The pipeline must route through embed() (typed payloads); embed_batch
    # is a direct-caller convenience and must never be called by the pipeline.
    from embedify.types import InputType, Modality

    class OnlyEmbed:
        modalities = {Modality.TEXT}
        payload_types = {"text"}
        dim = 4
        max_input = 100_000
        max_batch = 32
        model_id = "only-embed"
        model_version = "1"

        def embed(self, payloads, *, input_type=InputType.DOCUMENT):
            return [[1.0, 0.0, 0.0, 0.0] for _ in payloads]

        def embed_batch(self, texts, *, input_type=InputType.DOCUMENT):
            raise AssertionError("pipeline must not call embed_batch")

    result = embedify.embed(fixtures_dir / "sample.txt", config=hash_config,
                            embedder=OnlyEmbed())
    assert result.records and not result.errors
