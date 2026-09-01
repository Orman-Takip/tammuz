"""Degisim ciftlerini ve gorsellerini cikarma adimi.

Herakles DB'sindeki her `change_events` satiri icin once/sonra gorsellerini
`tile_snapshots`'tan okur, disk'e kaydeder ve `pairs.parquet` uzerine tum
metadata ile birlikte yazar. Bu dosya, sonraki tum adimlarin ham girdisidir.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import polars as pl
from rich.console import Console

from tammuz.config import Settings
from tammuz.db.reader import ChangeReader
from tammuz.utils.images import detect_extension, write_image

console = Console()

PAIR_COLUMNS = [
    "event_id",
    "tile_col",
    "tile_row",
    "from_release_id",
    "from_date",
    "to_release_id",
    "to_date",
    "detected_at",
    "ssim_score",
    "change_ratio",
    "largest_run",
    "land_windows",
    "water_windows",
    "ssim_min",
    "ssim_p05",
    "embedding_similarity",
    "embedding_computed",
    "from_land_cover",
    "to_land_cover",
    "before_image",
    "after_image",
]


def _iso(value) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value else ""


def _safe_release_id(release_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", release_id)


def _write_snapshot(
    images_dir: Path, col: int, row: int, release_id: str, data: bytes | None, written: int
) -> tuple[str, int]:
    if not data:
        return "", written
    ext = detect_extension(data)
    rel = f"{col}_{row}_{_safe_release_id(release_id)}.{ext}"
    path = images_dir / rel
    if not path.exists():
        write_image(path, data)
        written += 1
    return f"images/{rel}", written


def build_pairs(settings: Settings) -> Path:
    images_dir = Path(settings.paths.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    written = 0
    with ChangeReader(settings.paths.raw_db) as reader:
        for ev in reader.all_changes():
            event_id = int(ev.get("id") or len(rows))
            col = int(ev["tile_col"])
            row = int(ev["tile_row"])
            before = reader.snapshot(ev["from_release_id"], col, row)
            after = reader.snapshot(ev["to_release_id"], col, row)
            before_rel, written = _write_snapshot(images_dir, col, row, ev["from_release_id"], before, written)
            after_rel, written = _write_snapshot(images_dir, col, row, ev["to_release_id"], after, written)

            rows.append(
                {
                    "event_id": event_id,
                    "tile_col": col,
                    "tile_row": row,
                    "from_release_id": str(ev["from_release_id"]),
                    "from_date": _iso(ev.get("from_date")),
                    "to_release_id": str(ev["to_release_id"]),
                    "to_date": _iso(ev.get("to_date")),
                    "detected_at": _iso(ev.get("detected_at")),
                    "ssim_score": float(ev.get("ssim_score") or 0),
                    "change_ratio": float(ev.get("change_ratio") or 0),
                    "largest_run": int(ev.get("largest_run") or 0),
                    "land_windows": int(ev.get("land_windows") or 0),
                    "water_windows": int(ev.get("water_windows") or 0),
                    "ssim_min": float(ev.get("ssim_min") or 0),
                    "ssim_p05": float(ev.get("ssim_p05") or 0),
                    "embedding_similarity": float(ev.get("embedding_similarity") or 0),
                    "embedding_computed": int(ev.get("embedding_computed") or 0),
                    "from_land_cover": str(ev.get("from_land_cover") or ""),
                    "to_land_cover": str(ev.get("to_land_cover") or ""),
                    "before_image": before_rel,
                    "after_image": after_rel,
                }
            )

    df = pl.DataFrame(rows).select(PAIR_COLUMNS)
    out = Path(settings.paths.pairs_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)

    console.print(f"[green]pairs:[/green] {df.height} degisim cifti, {written} gorsel diske yazildi")
    return out
