"""Timestamps are always tz-aware past the loader (naive = assumed UTC)."""

from datetime import UTC, datetime, timedelta, timezone

from embedify.chunkers import TimeseriesChunker
from embedify.config import ChunkerConfig
from embedify.detector import Detector
from embedify.loaders import TabularLoader
from embedify.tabular import parse_timestamp
from embedify.types import Locator, Modality, Unit

EPOCH_2024 = 1704067200  # 2024-01-01T00:00:00Z


def test_parse_timestamp_all_input_kinds_are_aware():
    cases = [
        "2024-01-01T00:00:00",                    # tz-naive ISO string
        "2024-01-01T00:00:00+00:00",              # tz-aware ISO string
        "2024-01-01T00:00:00Z",                   # Zulu string
        EPOCH_2024,                               # epoch seconds
        EPOCH_2024 * 1000,                        # epoch milliseconds
        datetime(2024, 1, 1, 0, 0, 0),            # naive datetime object
        datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),  # aware datetime object
    ]
    parsed = [parse_timestamp(c) for c in cases]
    assert all(p is not None and p.tzinfo is not None for p in parsed)
    assert all(p == datetime(2024, 1, 1, tzinfo=UTC) for p in parsed)
    # Mixed aware datetimes compare without TypeError.
    assert sorted(parsed) == parsed


def test_parse_timestamp_assume_tz_is_configurable():
    plus2 = timezone(timedelta(hours=2))
    naive = parse_timestamp("2024-01-01T00:00:00", assume_tz=plus2)
    assert naive is not None and naive.utcoffset() == timedelta(hours=2)
    assert naive == datetime(2023, 12, 31, 22, 0, tzinfo=UTC)
    # Aware inputs are absolute: assume_tz must not shift them.
    aware = parse_timestamp("2024-01-01T00:00:00+05:00", assume_tz=plus2)
    assert aware is not None and aware.utcoffset() == timedelta(hours=5)


def test_mixed_format_series_windows_without_naive_aware_errors(tmp_path):
    path = tmp_path / "mixed.json"
    path.write_text(
        '[{"timestamp": "2024-01-01T00:00:00", "v": 1.0},'
        ' {"timestamp": "2024-01-01T00:01:00+00:00", "v": 2.0},'
        f' {{"timestamp": {EPOCH_2024 + 120}, "v": 3.0}},'
        ' {"timestamp": "2024-01-01 00:03:00", "v": 4.0}]'
    )
    points = list(TabularLoader().load(path, Detector().detect(path)))
    assert points and all(p.locator.time_start is not None for p in points)
    assert all(p.locator.time_start.tzinfo is not None for p in points)
    windows = list(TimeseriesChunker(
        ChunkerConfig(window_size=4, stride=4)).chunk(points))
    assert len(windows) == 1
    w = windows[0]
    assert w.locator.time_start == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    assert w.locator.time_end == datetime(2024, 1, 1, 0, 3, tzinfo=UTC)


def test_mixed_series_including_datetime_objects_sorts_and_windows():
    raw = ["2024-01-01T00:02:00", "2024-01-01T00:00:00+00:00",
           EPOCH_2024 + 60, datetime(2024, 1, 1, 0, 3)]
    points = [
        Unit(content=f"v={i}", modality=Modality.TIMESERIES, source_uri="mixed",
             locator=Locator(time_start=parse_timestamp(v), time_end=parse_timestamp(v)),
             metadata={"series": "s", "channels": ["v"], "values": {"v": float(i)}})
        for i, v in enumerate(raw)
    ]
    windows = list(TimeseriesChunker(
        ChunkerConfig(window_size=4, stride=4)).chunk(points))
    assert len(windows) == 1
    assert windows[0].locator.time_start == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    assert windows[0].locator.time_end == datetime(2024, 1, 1, 0, 3, tzinfo=UTC)
