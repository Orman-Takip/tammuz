"""XYZ slippy-map tile koordinatlari ile lat/lon donusumleri ve geometri yardimcilari.

Herakles ile ayni WebMercator kurali kullanilir (Esri Wayback ile uyumlu):
tile col = x, tile row = y.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TileBounds:
    west: float
    south: float
    east: float
    north: float


def lon_to_col(lon: float, zoom: int) -> int:
    n = 2**zoom
    return int(math.floor((lon + 180.0) / 360.0 * n))


def col_to_lon(col: int, zoom: int) -> float:
    n = 2**zoom
    return col / n * 360.0 - 180.0


def lat_to_row(lat: float, zoom: int) -> int:
    n = 2**zoom
    rad = math.radians(lat)
    merc = math.log(math.tan(rad) + 1.0 / math.cos(rad)) / math.pi
    return int(math.floor((1.0 - merc) / 2.0 * n))


def row_to_lat(row: int, zoom: int) -> float:
    n = 2**zoom
    y = 1.0 - 2.0 * row / n
    return math.degrees(math.atan(math.sinh(math.pi * y)))


def tile_bounds(col: int, row: int, zoom: int) -> TileBounds:
    return TileBounds(
        west=col_to_lon(col, zoom),
        east=col_to_lon(col + 1, zoom),
        north=row_to_lat(row, zoom),
        south=row_to_lat(row + 1, zoom),
    )


def bounds_polygon(col: int, row: int, zoom: int) -> list[list[float]]:
    """Tile sinirlarini GeoJSON Polygon outer ring olarak dondurur ([[lon, lat], ...])."""
    b = tile_bounds(col, row, zoom)
    return [
        [b.west, b.south],
        [b.east, b.south],
        [b.east, b.north],
        [b.west, b.north],
        [b.west, b.south],
    ]


def viewport_col_row_range(
    west: float, south: float, east: float, north: float, zoom: int
) -> tuple[int, int, int, int]:
    """Bir gorunum penceresinin (bbox) kapsadigi tile araligini dondurur."""
    min_col = lon_to_col(west, zoom)
    max_col = lon_to_col(east, zoom)
    min_row = lat_to_row(north, zoom)
    max_row = lat_to_row(south, zoom)
    return min_col, max_col, min_row, max_row


def lon_lat_to_km_scale(lat: float) -> float:
    """Belirli bir enlemde 1 derece boylamin yaklasik km karsiligini verir."""
    return 111.320 * math.cos(math.radians(lat))


def pairwise_haversine_km(a: Sequence[float], b: Sequence[float]) -> float:
    """Iki (lon, lat) noktasi arasindaki km cinsinden mesafe."""
    r_lat = math.radians(a[1])
    d_lat = math.radians(b[1] - a[1])
    d_lon = math.radians(b[0] - a[0])
    h = math.sin(d_lat / 2) ** 2 + math.cos(r_lat) * math.cos(math.radians(b[1])) * math.sin(d_lon / 2) ** 2
    return 2.0 * 6371.0 * math.asin(math.sqrt(h))
