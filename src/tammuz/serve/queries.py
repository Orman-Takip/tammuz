"""Harita uygulamasi icin sorgu katmani."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import polars as pl

from tammuz.config import Settings
from tammuz.utils.geo import tile_bounds


class ChangesStore:
    """changes.parquet'i DuckDB uzerinden hizli filtrelemeli olarak sunar."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = Path(settings.paths.changes_parquet)
        if not self.path.exists():
            raise FileNotFoundError(
                f"islenmis veri yok: {self.path}. Once pipeline'i calistirin veya bir release arsivi indirin."
            )
        self.df = pl.read_parquet(self.path)
        self.conn = duckdb.connect()
        self.conn.execute("CREATE TABLE changes AS SELECT * FROM read_parquet(?)", [str(self.path)])
        self._columns = set(self.df.columns)

    def _rows(self, sql: str, params: list | None = None) -> list[dict]:
        table = self.conn.execute(sql, params or []).to_arrow_table()
        return pl.from_arrow(table).to_dicts()

    def _types_expr(self, types: list[str] | None):
        if types:
            if "change_type_tr" not in self._columns:
                return "0=1"
            return f"change_type_tr IN ({', '.join(repr(t) for t in types)})"
        return "1=1"

    def _date_expr(self, from_date: str | None, to_date: str | None):
        parts = []
        if from_date:
            parts.append(f"to_date >= '{from_date}'")
        if to_date:
            parts.append(f"to_date <= '{to_date}'")
        return " AND ".join(parts) if parts else "1=1"

    def overview(self) -> dict:
        row = self._rows(
            """
            SELECT
              count(*) AS total,
              count(DISTINCT (tile_col, tile_row)) AS tiles,
              min(to_date) AS min_to,
              max(to_date) AS max_to,
              min(from_date) AS min_from
            FROM changes
            """
        )[0]
        types = (
            self._rows("SELECT change_type_tr AS t, count(*) AS c FROM changes GROUP BY 1 ORDER BY 2 DESC")
            if "change_type_tr" in self._columns
            else []
        )
        years = self._rows("SELECT substr(to_date, 1, 4) AS y, count(*) AS c FROM changes GROUP BY 1 ORDER BY 1")
        return {
            "total": int(row["total"]),
            "tiles": int(row["tiles"]),
            "date_range": {"from": row["min_to"], "to": row["max_to"]},
            "min_from": row["min_from"],
            "change_types": [{"type": r["t"], "count": int(r["c"])} for r in types if r["t"]],
            "by_year": [{"year": r["y"], "count": int(r["c"])} for r in years],
        }

    def count(
        self,
        min_col: int,
        max_col: int,
        min_row: int,
        max_row: int,
        types: list[str] | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> int:
        sql = (
            f"SELECT count(*) AS n FROM changes WHERE tile_col BETWEEN {min_col} AND {max_col} "
            f"AND tile_row BETWEEN {min_row} AND {max_row} "
            f"AND {self._types_expr(types)} AND {self._date_expr(from_date, to_date)}"
        )
        return int(self._rows(sql)[0]["n"])

    def features(
        self,
        min_col: int,
        max_col: int,
        min_row: int,
        max_row: int,
        types: list[str] | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        limit: int = 4000,
    ) -> dict:
        sql = (
            f"SELECT * FROM changes WHERE tile_col BETWEEN {min_col} AND {max_col} "
            f"AND tile_row BETWEEN {min_row} AND {max_row} "
            f"AND {self._types_expr(types)} AND {self._date_expr(from_date, to_date)} "
            f"ORDER BY importance DESC LIMIT {int(limit)}"
        )
        return {"type": "FeatureCollection", "features": [self._feature(r) for r in self._rows(sql)]}

    def grid(
        self,
        min_col: int,
        max_col: int,
        min_row: int,
        max_row: int,
        cell: int,
        types: list[str] | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict:
        features = []
        sql = (
            f"SELECT tile_col / {cell} AS gc, tile_row / {cell} AS gr, count(*) AS n "
            f"FROM changes WHERE tile_col BETWEEN {min_col} AND {max_col} "
            f"AND tile_row BETWEEN {min_row} AND {max_row} "
            f"AND {self._types_expr(types)} AND {self._date_expr(from_date, to_date)} "
            f"GROUP BY 1, 2"
        )
        zoom = self.settings.db.zoom
        for row in self._rows(sql):
            col0 = int(row["gc"]) * cell
            row0 = int(row["gr"]) * cell
            col1 = col0 + cell - 1
            row1 = row0 + cell - 1
            b0 = tile_bounds(col0, row0, zoom)
            b1 = tile_bounds(col1, row1, zoom)
            ring = [
                [b0.west, b1.south],
                [b1.east, b1.south],
                [b1.east, b0.north],
                [b0.west, b0.north],
                [b0.west, b1.south],
            ]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {"count": int(row["n"])},
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def event(self, event_id: int) -> dict | None:
        rows = self._rows("SELECT * FROM changes WHERE event_id = ?", [event_id])
        return self._feature(rows[0]) if rows else None

    def events_for_tile(self, col: int, row: int) -> dict:
        rows = self._rows("SELECT * FROM changes WHERE tile_col = ? AND tile_row = ? ORDER BY to_date", [col, row])
        return {"type": "FeatureCollection", "features": [self._feature(r) for r in rows]}

    def _feature(self, row: dict) -> dict:
        col = int(row["tile_col"])
        r = int(row["tile_row"])
        geometry = row.get("geometry_mask")
        if not geometry:
            geometry = row.get("geometry")
        props = {}
        for key in (
            "event_id",
            "tile_col",
            "tile_row",
            "centroid_lon",
            "centroid_lat",
            "from_date",
            "to_date",
            "from_release_id",
            "to_release_id",
            "change_type_tr",
            "change_type_en",
            "land_cover_from",
            "land_cover_to",
            "importance",
            "episode_id",
            "episode_index",
            "episode_count",
            "hotspot_id",
            "dino_similarity",
            "ssim_score",
            "change_ratio",
            "largest_run",
            "before_image",
            "after_image",
            "mask_image",
            "ai_is_real",
            "ai_change_type",
            "ai_cause_tr",
            "ai_confidence",
        ):
            if key in row:
                props[key] = row[key]
        if not geometry:
            geometry = json.dumps({"type": "Polygon", "coordinates": [self._tile_ring(col, r)]}, ensure_ascii=False)
        return {
            "type": "Feature",
            "geometry": json.loads(geometry),
            "properties": props,
        }

    def _tile_ring(self, col: int, row: int):
        b = tile_bounds(col, row, self.settings.db.zoom)
        return [[b.west, b.south], [b.east, b.south], [b.east, b.north], [b.west, b.north], [b.west, b.south]]
