"""Episode birlestirme, onem skoru ve hotspot kumeleme adimi.

- Episode: ayni tile uzerindeki zamanla orten degisimler tek bir sureklilik
  hikayesi olarak birlestirilir (ornegin yayilan bir yapilasmanin her yil
  tekrar yakalanmasi).
- Onem skoru: degisimin buyuklugu, yayginligi, arazi etkisi ve guncelligi
  birlestirilerek 0..1 arasi tek bir deger uretilir.
- Hotspot: konum bazli kumeleme ile "bolgesel degisim noktalari" bulunur.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console
from sklearn.cluster import DBSCAN

from tammuz.config import Settings
from tammuz.utils.geo import bounds_polygon, lon_lat_to_km_scale, tile_bounds

console = Console()


def _to_dates(values) -> np.ndarray:
    out = np.empty(len(values), dtype="datetime64[ns]")
    for i, v in enumerate(values):
        try:
            out[i] = np.datetime64(v)
        except (ValueError, TypeError):
            out[i] = np.datetime64("NaT")
    return out


def _assign_episodes(changes: pl.DataFrame) -> pl.DataFrame:
    df = changes.sort(["tile_col", "tile_row", "to_date"])
    n = df.height
    from_dates = _to_dates(df["from_date"].to_list())
    to_dates = _to_dates(df["to_date"].to_list())
    cols = df["tile_col"].to_numpy()
    rows = df["tile_row"].to_numpy()

    groups: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        groups.setdefault((int(cols[i]), int(rows[i])), []).append(i)

    episode_id = np.zeros(n, dtype=np.int64)
    eid = 0
    for idxs in groups.values():
        for pos, i in enumerate(idxs):
            if pos > 0 and not np.isnat(from_dates[i]) and from_dates[i] <= to_dates[idxs[pos - 1]]:
                episode_id[i] = episode_id[idxs[pos - 1]]
            else:
                eid += 1
                episode_id[i] = eid

    episode_count = np.zeros(n, dtype=np.int64)
    episode_index = np.zeros(n, dtype=np.int64)
    uniq, counts = np.unique(episode_id, return_counts=True)
    count_map = dict(zip(uniq, counts, strict=False))
    counter: dict[int, int] = {}
    for i in range(n):
        e = int(episode_id[i])
        episode_count[i] = count_map[e]
        episode_index[i] = counter.get(e, 0)
        counter[e] = counter.get(e, 0) + 1

    return df.with_columns(
        pl.Series("episode_id", episode_id),
        pl.Series("episode_index", episode_index),
        pl.Series("episode_count", episode_count),
    )


def _importance(changes: pl.DataFrame, weights) -> pl.Series:
    dino = np.clip(changes["dino_similarity"].fill_null(1.0).to_numpy(), 0, 1)
    magnitude = 1.0 - dino
    ratio = np.clip(changes["change_ratio"].fill_null(0.0).to_numpy(), 0, 1)
    if "change_type_tr" in changes.columns:
        land_impact = (changes["change_type_tr"] != "").to_numpy().astype(np.float64)
    else:
        land_impact = np.zeros(len(changes), dtype=np.float64)

    dates = _to_dates(changes["to_date"].to_list())
    recency = np.zeros(len(dates), dtype=np.float64)
    valid = ~np.isnat(dates)
    if valid.any():
        dmin = np.min(dates[valid])
        dmax = np.max(dates[valid])
        span = max(float((dmax - dmin) / np.timedelta64(1, "s")), 1e-9)
        recency[valid] = (dates[valid] - dmin).astype("int64") / 1e9 / span

    score = (
        weights.change_ratio * ratio
        + weights.magnitude * magnitude
        + weights.land_impact * land_impact
        + weights.recency * recency
    )
    return pl.Series(np.clip(score, 0, 1))


def _hotspots(changes: pl.DataFrame, zoom: int, eps_km: float, min_samples: int):
    bounds = [tile_bounds(int(c), int(r), zoom) for c, r in zip(changes["tile_col"], changes["tile_row"], strict=False)]
    lons = np.array([(b.west + b.east) / 2 for b in bounds])
    lats = np.array([(b.south + b.north) / 2 for b in bounds])

    scale = lon_lat_to_km_scale(float(np.mean(lats)))
    coords = np.column_stack([lons * scale, lats * 111.32])

    labels = DBSCAN(eps=eps_km, min_samples=min_samples).fit(coords).labels_

    ids = np.where(labels == -1, 0, labels + 1)
    centroids_lon = np.full(len(labels), np.nan)
    centroids_lat = np.full(len(labels), np.nan)
    for cluster in np.unique(labels):
        if cluster == -1:
            continue
        member = labels == cluster
        centroids_lon[member] = float(np.mean(lons[member]))
        centroids_lat[member] = float(np.mean(lats[member]))

    return (
        pl.Series("hotspot_id", ids),
        pl.Series("hotspot_centroid_lon", centroids_lon),
        pl.Series("hotspot_centroid_lat", centroids_lat),
    )


def run_enrich(settings: Settings) -> Path:
    changes_path = Path(settings.paths.changes_parquet)
    if not changes_path.exists():
        raise FileNotFoundError(f"once `tammuz filter` calistirin: {changes_path}")

    changes = pl.read_parquet(changes_path)
    zoom = settings.db.zoom

    geometries: list[str] = []
    centroid_lons: list[float] = []
    centroid_lats: list[float] = []
    for row in changes.iter_rows(named=True):
        col, r = int(row["tile_col"]), int(row["tile_row"])
        geometries.append(json.dumps({"type": "Polygon", "coordinates": [bounds_polygon(col, r, zoom)]}))
        b = tile_bounds(col, r, zoom)
        centroid_lons.append((b.west + b.east) / 2)
        centroid_lats.append((b.south + b.north) / 2)

    changes = changes.with_columns(
        pl.Series("geometry", geometries),
        pl.Series("centroid_lon", centroid_lons),
        pl.Series("centroid_lat", centroid_lats),
    )

    changes = _assign_episodes(changes)
    changes = changes.with_columns(_importance(changes, settings.enrich.importance_weights).alias("importance"))

    h_id, h_lon, h_lat = _hotspots(changes, zoom, settings.enrich.hotspot_eps_km, settings.enrich.hotspot_min_samples)
    changes = changes.with_columns(h_id, h_lon, h_lat)

    if settings.enrich.drop_ai_false and "ai_is_real" in changes.columns:
        before = changes.height
        changes = changes.filter(changes["ai_is_real"].is_null() | changes["ai_is_real"])
        console.print(f"[yellow]ai_is_real=false olan {before - changes.height} kayit elendi[/yellow]")

    changes.write_parquet(changes_path)

    console.print(
        f"[green]enrich:[/green] {changes.height} degisim, "
        f"episode sayisi {changes['episode_id'].n_unique()}, "
        f"hotspot kumesi {int((changes['hotspot_id'] > 0).sum())}"
    )
    return changes_path
