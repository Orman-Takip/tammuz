"""Veri ihrac adimi.

`changes.parquet`'i herkesin kullanabilecegi formatlara cevirir:

- GeoJSON: harita araclari, kepler.gl, geojson.io
- NDJSON: satir satir islenebilir, buyuk veri dostu
- GeoParquet: veri bilimi icin (DuckDB, pyogrio), WKB geometri + geo metadata
- summary.json: genel istatistik ozeti
"""

from __future__ import annotations

import json
import struct
import tarfile
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console

from tammuz.config import Settings

console = Console()

EXPORT_COLUMNS = [
    "event_id",
    "tile_col",
    "tile_row",
    "centroid_lon",
    "centroid_lat",
    "from_release_id",
    "from_date",
    "to_release_id",
    "to_date",
    "detected_at",
    "ssim_score",
    "change_ratio",
    "largest_run",
    "dino_similarity",
    "land_cover_from",
    "land_cover_to",
    "change_type_tr",
    "change_type_en",
    "importance",
    "episode_id",
    "episode_index",
    "episode_count",
    "hotspot_id",
    "before_image",
    "after_image",
    "mask_image",
    "ai_is_real",
    "ai_change_type",
    "ai_cause_tr",
    "ai_confidence",
]


def _geometry_geojson(row) -> dict:
    if row.get("geometry_mask"):
        return json.loads(row["geometry_mask"])
    return json.loads(row["geometry"])


def _wkb_polygon(ring: list[list[float]]) -> bytes:
    data = struct.pack("<B I", 1, 3)
    data += struct.pack("<I", 1)
    data += struct.pack("<I", len(ring))
    for lon, lat in ring:
        data += struct.pack("<dd", float(lon), float(lat))
    return data


def _geometry_wkb(geom: dict) -> bytes:
    if geom["type"] == "Polygon":
        return _wkb_polygon(geom["coordinates"][0])
    polys = b""
    for ring in geom["coordinates"]:
        polys += _wkb_polygon(ring[0])
    data = struct.pack("<B I", 1, 6)
    data += struct.pack("<I", len(geom["coordinates"]))
    data += polys
    return data


def _features(changes: pl.DataFrame) -> list[dict]:
    features = []
    for row in changes.iter_rows(named=True):
        props = {c: row.get(c) for c in EXPORT_COLUMNS if c in row and c not in ("geometry", "geometry_mask")}
        for key in ("ai_is_real", "ai_change_type", "ai_cause_tr", "ai_confidence"):
            if key not in props:
                props[key] = None
        features.append({"type": "Feature", "geometry": _geometry_geojson(row), "properties": props})
    return features


def _write_geojson(path: Path, features: list[dict]) -> None:
    fc = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")


def _write_ndjson(path: Path, features: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for feat in features:
            fh.write(json.dumps(feat, ensure_ascii=False) + "\n")


def _write_geoparquet(path: Path, changes: pl.DataFrame) -> None:
    keep_cols = [c for c in changes.columns if c not in ("geometry", "geometry_mask")]
    table = changes.select(keep_cols).to_arrow()
    wkb = [_geometry_wkb(_geometry_geojson(row)) for row in changes.iter_rows(named=True)]
    table = table.append_column("geometry", pa.array(wkb, type=pa.binary()))

    geo_meta = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Polygon", "MultiPolygon"],
                "crs": {"id": {"authority": "EPSG", "code": 4326}},
            }
        },
    }
    metadata = table.schema.metadata or {}
    metadata[b"geo"] = json.dumps(geo_meta).encode()
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path, compression="zstd")


def _summary(changes: pl.DataFrame) -> dict:
    from collections import Counter

    def safe_min_max(col):
        if col not in changes.columns:
            return "", ""
        vals = [v for v in changes[col].to_list() if v]
        if not vals:
            return "", ""
        return str(min(vals)), str(max(vals))

    types = Counter(changes["change_type_tr"].to_list()) if "change_type_tr" in changes.columns else Counter()
    years = Counter()
    for v in changes["to_date"].to_list():
        if v:
            years[str(v)[:4]] += 1

    return {
        "total_changes": int(changes.height),
        "distinct_tiles": int(changes.select(pl.col("tile_col"), pl.col("tile_row")).unique().height),
        "date_range": {"from": safe_min_max("from_date")[0], "to": safe_min_max("to_date")[1]},
        "change_types": dict(types.most_common()),
        "by_year": dict(sorted(years.items())),
        "labeled": int(((changes["land_cover_from"] != "") & (changes["land_cover_to"] != "")).sum())
        if "land_cover_from" in changes.columns
        else 0,
        "with_mask": int((changes["geometry_mask"] != "").sum()) if "geometry_mask" in changes.columns else 0,
        "episodes": int(changes["episode_id"].n_unique()) if "episode_id" in changes.columns else 0,
        "hotspots": int((changes["hotspot_id"] > 0).sum()) if "hotspot_id" in changes.columns else 0,
    }


def run_export(settings: Settings, release: bool = False) -> Path:
    changes_path = Path(settings.paths.changes_parquet)
    if not changes_path.exists():
        raise FileNotFoundError(f"once `tammuz enrich` calistirin: {changes_path}")

    changes = pl.read_parquet(changes_path)
    exports_dir = Path(settings.paths.exports_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)

    features = _features(changes)

    geojson_path = exports_dir / "degisimler.geojson"
    ndjson_path = exports_dir / "degisimler.ndjson"
    geoparquet_path = exports_dir / "degisimler.geoparquet"
    summary_path = exports_dir / "summary.json"

    _write_geojson(geojson_path, features)
    _write_ndjson(ndjson_path, features)
    _write_geoparquet(geoparquet_path, changes)
    summary_path.write_text(json.dumps(_summary(changes), ensure_ascii=False, indent=2), encoding="utf-8")

    console.print(
        f"[green]export:[/green] {len(features)} degisim -> "
        f"{geojson_path.name}, {ndjson_path.name}, {geoparquet_path.name}, summary.json"
    )

    if release:
        _build_release_archive(settings, exports_dir)
    return exports_dir


def _build_release_archive(settings: Settings, exports_dir: Path) -> Path:
    archive = exports_dir.parent / "tammuz-export.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("degisimler.geojson", "degisimler.ndjson", "degisimler.geoparquet", "summary.json"):
            tar.add(exports_dir / name, arcname=name)
        masks = Path(settings.paths.masks_dir)
        if masks.exists():
            tar.add(masks, arcname="masks")
    console.print(f"[green]release:[/green] {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    return archive
