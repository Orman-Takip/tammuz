"""Herakles SQLite dosyasindan degisim kayitlarini ve tile gorsellerini okur.

Herakles'in `change_events` tablosu tespit edilen her degisim icin bir satir
icerir; `tile_snapshots` ise o degisimin dayandigi once/sonra gorsellerini
saklar (Esri Wayback arsivi kalici olmasa bile gorseller elimizde kalir).

Bu modul veritabanini yalnizca okur, hicbir sey degistirmez.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

CHANGE_TABLE = "change_events"
SNAPSHOT_TABLE = "tile_snapshots"

# Yaygin olarak kullanilan kolonlar; eski veritabanlarinda bulunmayabilecek
# alanlar SELECT * ile dinamik olarak karsilanir.
CORE_COLUMNS = (
    "tile_col",
    "tile_row",
    "from_release_id",
    "from_date",
    "to_release_id",
    "to_date",
    "detected_at",
)


class ChangeReader:
    """Bir herakles.db dosyasini salt-okunur acan ve sorgu nesnesi saglayan sinif."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"veritabani bulunamadi: {self.db_path}")
        self._conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row
        self._columns = self._table_columns(CHANGE_TABLE)

    def _table_columns(self, table: str) -> set[str]:
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {row["name"] for row in rows}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ChangeReader:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def change_count(self) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) AS n FROM {CHANGE_TABLE}").fetchone()
        return int(row["n"])

    @property
    def snapshot_count(self) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) AS n FROM {SNAPSHOT_TABLE}").fetchone()
        return int(row["n"])

    def _change_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        for key in ("from_date", "to_date", "detected_at"):
            if key in out and isinstance(out[key], str) and out[key]:
                try:
                    out[key] = datetime.fromisoformat(out[key].replace("Z", "+00:00"))
                except ValueError:
                    pass
        return out

    def all_changes(self) -> Iterator[dict[str, Any]]:
        """Tum degisim kayitlarini, kayit sirasiyla dondurur."""
        select_cols = ", ".join(sorted(self._columns & set(CORE_COLUMNS)))
        extra_cols = [
            c
            for c in (
                "id",
                "ssim_score",
                "change_ratio",
                "land_windows",
                "water_windows",
                "largest_run",
                "ssim_min",
                "ssim_p05",
                "embedding_similarity",
                "embedding_computed",
                "from_land_cover",
                "to_land_cover",
            )
            if c in self._columns
        ]
        cols = ", ".join(dict.fromkeys([*extra_cols, *select_cols.split(", ")]))
        for row in self._conn.execute(f"SELECT {cols} FROM {CHANGE_TABLE} ORDER BY id"):
            yield self._change_row_to_dict(row)

    def snapshot(self, release_id: str, col: int, row: int) -> bytes | None:
        """Bir tile gorselini BLOB olarak dondurur; yoksa None."""
        result = self._conn.execute(
            f"SELECT image FROM {SNAPSHOT_TABLE} WHERE release_id = ? AND tile_col = ? AND tile_row = ?",
            (release_id, col, row),
        ).fetchone()
        return bytes(result["image"]) if result else None

    def snapshot_content_type(self, release_id: str, col: int, row: int) -> str | None:
        if "content_type" not in self._columns:
            return None
        result = self._conn.execute(
            f"SELECT content_type FROM {SNAPSHOT_TABLE} WHERE release_id = ? AND tile_col = ? AND tile_row = ?",
            (release_id, col, row),
        ).fetchone()
        return str(result["content_type"]) if result else None

    def has_snapshots(self, event: dict[str, Any]) -> bool:
        before = self.snapshot(event["from_release_id"], event["tile_col"], event["tile_row"])
        after = self.snapshot(event["to_release_id"], event["tile_col"], event["tile_row"])
        return before is not None and after is not None
