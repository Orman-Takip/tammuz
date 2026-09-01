"""Ortak test kurulumu: sentetik Herakles DB ve yapilandirma ureticileri."""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from tammuz.config import Settings


def make_image(size: int = 64, color: tuple[int, int, int] = (120, 140, 160)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, "PNG")
    return buf.getvalue()


def make_test_db(path: Path, n_events: int = 3) -> Path:
    """change_events + tile_snapshots tablolarini iceren sentetik bir DB uretir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE change_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tile_col INTEGER NOT NULL,
            tile_row INTEGER NOT NULL,
            from_release_id TEXT NOT NULL,
            from_date TEXT NOT NULL,
            to_release_id TEXT NOT NULL,
            to_date TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            ssim_score REAL NOT NULL DEFAULT 0,
            change_ratio REAL NOT NULL DEFAULT 0,
            land_windows INTEGER NOT NULL DEFAULT 0,
            water_windows INTEGER NOT NULL DEFAULT 0,
            largest_run INTEGER NOT NULL DEFAULT 0,
            ssim_min REAL NOT NULL DEFAULT 0,
            ssim_p05 REAL NOT NULL DEFAULT 0,
            embedding_similarity REAL NOT NULL DEFAULT 0,
            embedding_computed INTEGER NOT NULL DEFAULT 0,
            from_land_cover TEXT NOT NULL DEFAULT '',
            to_land_cover TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE tile_snapshots (
            release_id TEXT NOT NULL,
            tile_col INTEGER NOT NULL,
            tile_row INTEGER NOT NULL,
            image BLOB NOT NULL,
            content_type TEXT NOT NULL,
            PRIMARY KEY (release_id, tile_col, tile_row)
        );
        """
    )
    base_col, base_row = 37500, 25100
    for i in range(n_events):
        from_rid = f"2020-{i + 1:02d}-01"
        to_rid = f"2021-{i + 1:02d}-01"
        conn.execute(
            "INSERT INTO change_events (tile_col, tile_row, from_release_id, from_date, to_release_id, to_date,"
            " detected_at, ssim_score, change_ratio, land_windows, largest_run, from_land_cover, to_land_cover)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                base_col + i,
                base_row,
                from_rid,
                from_rid,
                to_rid,
                to_rid,
                to_rid,
                0.6 - i * 0.1,
                0.4,
                100,
                8,
                "AnnualCrop",
                "Residential",
            ),
        )
        conn.execute(
            "INSERT INTO tile_snapshots (release_id, tile_col, tile_row, image, content_type) VALUES (?, ?, ?, ?, ?)",
            (from_rid, base_col + i, base_row, make_image(color=(100 + i, 100, 100)), "image/png"),
        )
        conn.execute(
            "INSERT INTO tile_snapshots (release_id, tile_col, tile_row, image, content_type) VALUES (?, ?, ?, ?, ?)",
            (to_rid, base_col + i, base_row, make_image(color=(40 + i, 200, 60)), "image/png"),
        )
    conn.commit()
    conn.close()
    return path


def make_settings(root: Path, n_events: int = 3) -> Settings:
    """Tum yollari tmp dizinine isaret eden bir Settings uretir."""
    root = Path(root)
    make_test_db(root / "raw" / "herakles.db", n_events=n_events)
    data = {
        "paths": {
            "data_dir": str(root / "data"),
            "raw_db": str(root / "raw" / "herakles.db"),
            "models_dir": str(root / "models"),
            "datasets_dir": str(root / "datasets"),
            "processed_dir": str(root / "processed"),
            "images_dir": str(root / "processed" / "images"),
            "masks_dir": str(root / "processed" / "masks"),
            "lancedb_dir": str(root / "processed" / "lancedb"),
            "exports_dir": str(root / "processed" / "exports"),
            "pairs_parquet": str(root / "processed" / "pairs.parquet"),
            "embeddings_parquet": str(root / "processed" / "embeddings.parquet"),
            "changes_parquet": str(root / "processed" / "changes.parquet"),
        }
    }
    return Settings.model_validate(data)


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)
