"""DB okuma ve pairs adimi testleri."""

from pathlib import Path

import polars as pl

from tammuz.db.reader import ChangeReader
from tammuz.pipeline.mask import run_mask
from tammuz.pipeline.pairs import build_pairs


def test_reader_reads_events_and_snapshots(test_settings):
    with ChangeReader(test_settings.paths.raw_db) as reader:
        events = list(reader.all_changes())
        assert len(events) == 3
        first = events[0]
        assert first["tile_col"] == 37500
        assert first["from_release_id"]
        snap = reader.snapshot(first["from_release_id"], first["tile_col"], first["tile_row"])
        assert snap and snap[:8] == b"\x89PNG\r\n\x1a\n"


def test_reader_missing_snapshot_returns_none(test_settings):
    with ChangeReader(test_settings.paths.raw_db) as reader:
        assert reader.snapshot("yok", 1, 2) is None


def test_build_pairs_writes_images_and_parquet(test_settings):
    out = build_pairs(test_settings)
    assert out.exists()
    df = pl.read_parquet(out)
    assert df.height == 3
    assert "before_image" in df.columns
    assert "after_image" in df.columns
    before = Path(test_settings.paths.images_dir) / Path(df["before_image"][0]).name
    assert before.exists()


def test_mask_runs_on_pairs(test_settings):
    build_pairs(test_settings)
    changes = pl.read_parquet(test_settings.paths.pairs_parquet)
    changes.write_parquet(test_settings.paths.changes_parquet)
    run_mask(test_settings)
    df = pl.read_parquet(test_settings.paths.changes_parquet)
    assert "geometry_mask" in df.columns
