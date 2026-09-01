"""Pipeline ve sorgu katmani testleri."""

import json
from pathlib import Path

import polars as pl

from tammuz.pipeline.enrich import run_enrich
from tammuz.pipeline.export import run_export
from tammuz.pipeline.filter import run_filter
from tammuz.pipeline.mask import run_mask
from tammuz.pipeline.pairs import build_pairs
from tammuz.serve.queries import ChangesStore


def _seed_changes(settings) -> None:
    build_pairs(settings)
    pairs = pl.read_parquet(settings.paths.pairs_parquet)
    pairs = pairs.with_columns(pl.Series("dino_similarity", [0.70, 0.95, 0.80]))
    pairs.write_parquet(settings.paths.pairs_parquet)
    run_filter(settings)


def test_filter_keeps_only_below_threshold(test_settings):
    _seed_changes(test_settings)
    changes = pl.read_parquet(test_settings.paths.changes_parquet)
    assert changes.height == 2
    sims = changes["dino_similarity"].to_list()
    assert all(s < 0.92 for s in sims)


def test_enrich_adds_geometry_episodes_hotspots(test_settings):
    _seed_changes(test_settings)
    run_mask(test_settings)
    run_enrich(test_settings)
    changes = pl.read_parquet(test_settings.paths.changes_parquet)
    for col in ("geometry", "centroid_lon", "centroid_lat", "episode_id", "importance", "hotspot_id"):
        assert col in changes.columns
    geom = json.loads(changes["geometry"][0])
    assert geom["type"] == "Polygon"


def test_export_writes_all_formats(test_settings):
    _seed_changes(test_settings)
    run_mask(test_settings)
    run_enrich(test_settings)
    run_export(test_settings)

    exports = Path(test_settings.paths.exports_dir)
    assert (exports / "degisimler.geojson").exists()
    assert (exports / "degisimler.ndjson").exists()
    assert (exports / "degisimler.geoparquet").exists()
    assert (exports / "summary.json").exists()

    fc = json.loads((exports / "degisimler.geojson").read_text(encoding="utf-8"))
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 2


def test_store_count_and_features(test_settings):
    _seed_changes(test_settings)
    run_enrich(test_settings)
    store = ChangesStore(test_settings)
    total = store.count(0, 100000, 0, 100000)
    assert total == 2
    fc = store.features(0, 100000, 0, 100000, limit=10)
    assert len(fc["features"]) == 2
    ov = store.overview()
    assert ov["total"] == 2
    assert "change_types" in ov

    ev = store.event(fc["features"][0]["properties"]["event_id"])
    assert ev is not None
