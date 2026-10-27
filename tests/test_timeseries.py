import embedify
from embedify.config import ChunkerConfig, EmbedderConfig, EmbedifyConfig, SinkConfig
from embedify.embedders import StatsFeatureEmbedder, create_embedder
from embedify.types import Modality


def _cfg(**kw):
    return EmbedifyConfig(
        embedders={"text": EmbedderConfig(backend="hash", model="hash", dim=32)},
        chunkers={"timeseries": ChunkerConfig(**kw), "tabular": ChunkerConfig(**kw)},
        sink=SinkConfig(type="memory"),
    )


def test_timeseries_windows_from_csv(fixtures_dir, hash_embedder):
    cfg = _cfg(window_size=4, stride=2, normalization="zscore")
    result = embedify.embed(fixtures_dir / "series.csv", config=cfg, embedder=hash_embedder)
    assert result.stats.files_failed == 0
    assert len(result.records) >= 3
    rec = result.records[0]
    assert rec.modality == Modality.TIMESERIES
    assert rec.locator.time_start is not None and rec.locator.time_end > rec.locator.time_start
    assert rec.metadata["channels"] == ["sensor_a", "sensor_b"]
    # gap at 00:04 -> irregular sampling flagged, not silently resampled
    assert rec.metadata["irregular_sampling"] is True


def test_timeseries_stats_embedder_windows(fixtures_dir):
    cfg = _cfg(window_size=4, stride=4)
    stats_emb = create_embedder(EmbedderConfig(backend="stats", model="stats-features", dim=32))
    result = embedify.embed(fixtures_dir / "series.csv", config=cfg, embedder=stats_emb)
    assert result.records
    assert all(r.dim == 32 for r in result.records)
    assert all(r.model == "stats-features" for r in result.records)


def test_timeseries_prometheus(fixtures_dir, hash_embedder):
    cfg = _cfg(window_size=2, stride=1)
    result = embedify.embed(fixtures_dir / "metrics.prom", config=cfg, embedder=hash_embedder)
    assert result.records
    assert result.records[0].modality == Modality.TIMESERIES
    assert result.records[0].metadata["metric"] == "http_requests_total"


def test_stats_embedder_deterministic():
    from embedify.types import Locator, Unit
    emb = StatsFeatureEmbedder(EmbedderConfig(backend="stats", model="s", dim=16))
    unit = Unit(content="w", modality=Modality.TIMESERIES, source_uri="x",
                locator=Locator(), numeric_values=[[1.0, 2.0], [3.0, 4.0]])
    a = emb.embed_units([unit])
    b = emb.embed_units([unit])
    assert a == b and len(a[0]) == 16
