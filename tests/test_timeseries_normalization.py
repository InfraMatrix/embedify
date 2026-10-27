"""Points 2-3: normalization params stored; series scope != constant features."""

from embedify.chunkers import TimeseriesChunker
from embedify.config import ChunkerConfig, EmbedderConfig
from embedify.detector import Detector
from embedify.embedders import StatsFeatureEmbedder
from embedify.loaders import TabularLoader


def _windows(fixtures_dir, **kw):
    units = list(TabularLoader().load(fixtures_dir / "series.csv",
                                      Detector().detect(fixtures_dir / "series.csv")))
    return list(TimeseriesChunker(ChunkerConfig(**kw)).chunk(units))


def test_normalization_params_and_window_config_stored(fixtures_dir):
    windows = _windows(fixtures_dir, window_size=4, stride=2, normalization="zscore")
    w = windows[0]
    assert w.metadata["normalization"] == "zscore"
    assert w.metadata["normalization_scope"] == "series"
    assert w.metadata["window_size"] == 4 and w.metadata["stride"] == 2
    params = w.metadata["normalization_params"]
    assert set(params) == {"sensor_a", "sensor_b"}
    assert params["sensor_a"]["mean"] > 0 and params["sensor_a"]["std"] > 0


def test_series_scope_stats_features_vary_across_windows(fixtures_dir):
    # Series-level z-score (default): window feature means must NOT be the
    # constant 0/1 that per-window normalization would produce.
    windows = _windows(fixtures_dir, window_size=4, stride=4,
                       normalization="zscore", normalization_scope="series")
    assert len(windows) >= 2
    emb = StatsFeatureEmbedder(EmbedderConfig(backend="stats", model="s", dim=32))
    vecs = emb.embed([w.payload_for({"numeric"}) for w in windows])
    first_features = [v[0] for v in vecs]
    assert len(set(round(f, 6) for f in first_features)) > 1


def test_window_scope_is_configurable_and_differs(fixtures_dir):
    series = _windows(fixtures_dir, window_size=4, stride=4, normalization_scope="series")
    windowed = _windows(fixtures_dir, window_size=4, stride=4, normalization_scope="window")
    assert windowed[0].metadata["normalization_scope"] == "window"
    assert series[0].numeric_values != windowed[0].numeric_values
    # per-window z-score: each window's first channel has ~zero mean
    col = [r[0] for r in windowed[0].numeric_values]
    assert abs(sum(col) / len(col)) < 1e-9
