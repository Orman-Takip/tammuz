"""Degisim bolgesi maskeleme ve poligon adimi.

Once/sonra gorselleri arasindaki piksel farkindan kaba bir degisim haritasi
cikarir: goruntuyu esit boyutlu hucrelere boler, her hucrenin ortalama mutlak
farkini esikler ve baglantili bolgeleri bulur. Her bolge icin tile'in gercek
lon/lat sinirlariyla hizali bir dikdortgen poligon uretilir; boylece "tile'in
neresinde degisim var" sorusu tile-level'dan bolge-level'a tasinir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image
from rich.console import Console
from scipy import ndimage

from tammuz.config import Settings
from tammuz.utils.geo import col_to_lon, row_to_lat

console = Console()


def _to_gray_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"))


def _diff_mask(a: np.ndarray, b: np.ndarray, cell: int, threshold: int) -> tuple[np.ndarray, int, int]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]

    cells_y = h // cell
    cells_x = w // cell
    if cells_y == 0 or cells_x == 0:
        return np.zeros((h // cell, w // cell), dtype=bool), 0, 0

    a_crop = a[: cells_y * cell, : cells_x * cell]
    b_crop = b[: cells_y * cell, : cells_x * cell]
    diff = np.abs(a_crop.astype(np.int16) - b_crop.astype(np.int16))
    cell_mean = diff.reshape(cells_y, cell, cells_x, cell).mean(axis=(1, 3))
    return cell_mean > threshold, cells_x, cells_y


def _cell_boxes(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    labeled, n = ndimage.label(mask)
    if n == 0:
        return []
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < min_area:
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return boxes


def _box_to_polygon(box, cells_x, cells_y, w, h, col, row, zoom) -> list[list[float]]:
    x0, y0, x1, y1 = box
    u0, u1 = x0 / cells_x, x1 / cells_x
    v0, v1 = y0 / cells_y, y1 / cells_y
    west = col_to_lon(col + u0, zoom)
    east = col_to_lon(col + u1, zoom)
    north = row_to_lat(row + v0, zoom)
    south = row_to_lat(row + v1, zoom)
    return [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
    ]


def run_mask(settings: Settings) -> Path:
    changes_path = Path(settings.paths.changes_parquet)
    if not changes_path.exists():
        raise FileNotFoundError(f"once `tammuz filter` calistirin: {changes_path}")

    changes = pl.read_parquet(changes_path)
    images_dir = Path(settings.paths.images_dir)
    masks_dir = Path(settings.paths.masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    zoom = settings.db.zoom
    cell = settings.mask.cell_size
    threshold = settings.mask.diff_threshold
    min_area = settings.mask.min_area

    geometries: list[str] = []
    mask_images: list[str] = []

    for row in changes.iter_rows(named=True):
        before = row.get("before_image") or ""
        after = row.get("after_image") or ""
        geometry = ""
        mask_rel = ""
        if before and after:
            a = _to_gray_uint8(images_dir / Path(before).name)
            b = _to_gray_uint8(images_dir / Path(after).name)
            mask, cells_x, cells_y = _diff_mask(a, b, cell, threshold)
            boxes = _cell_boxes(mask, min_area)
            if boxes:
                h = min(a.shape[0], b.shape[0])
                w = min(a.shape[1], b.shape[1])
                polygons = [
                    _box_to_polygon(box, cells_x, cells_y, w, h, row["tile_col"], row["tile_row"], zoom)
                    for box in boxes
                ]
                geometry = _geo_json(polygons)

                overlay = _mask_overlay(a.shape[0], a.shape[1], cell, mask)
                event_id = int(row["event_id"])
                mask_path = masks_dir / f"{event_id}.png"
                Image.fromarray(overlay).save(mask_path)
                mask_rel = f"masks/{event_id}.png"

        geometries.append(geometry)
        mask_images.append(mask_rel)

    changes = changes.with_columns(
        pl.Series("geometry_mask", geometries),
        pl.Series("mask_image", mask_images),
    )
    changes.write_parquet(changes_path)

    with_mask = changes["geometry_mask"] != ""
    console.print(f"[green]mask:[/green] {changes.height} adaydan {int(with_mask.sum())} bolge poligonu uretildi")
    return changes_path


def _geo_json(polygons: list[list[list[float]]]) -> str:
    import json

    if len(polygons) == 1:
        return json.dumps({"type": "Polygon", "coordinates": [polygons[0]]}, ensure_ascii=False)
    return json.dumps({"type": "MultiPolygon", "coordinates": [[p] for p in polygons]}, ensure_ascii=False)


def _mask_overlay(h: int, w: int, cell: int, mask: np.ndarray) -> np.ndarray:
    mask_up = np.kron(mask.astype(np.uint8), np.ones((cell, cell), dtype=np.uint8))
    mask_up = mask_up[:h, :w]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 2] = 255
    rgba[..., 3] = 150
    rgba[mask_up == 0, 3] = 0
    return rgba
