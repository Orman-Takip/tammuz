"""Tile koordinati donusum testleri."""

import math

from tammuz.utils.geo import (
    col_to_lon,
    lat_to_row,
    lon_to_col,
    row_to_lat,
    tile_bounds,
    viewport_col_row_range,
)


def test_roundtrip_col_lon():
    for lon in (-180.0, -30.0, 0.0, 34.0, 179.9):
        col = lon_to_col(lon, 16)
        back = col_to_lon(col, 16)
        assert back <= lon < col_to_lon(col + 1, 16)


def test_roundtrip_row_lat():
    for lat in (-85.0, 0.0, 39.0, 42.0, 85.0):
        row = lat_to_row(lat, 16)
        north = row_to_lat(row, 16)
        south = row_to_lat(row + 1, 16)
        assert south - 1e-9 <= lat <= north + 1e-9
        assert south < north


def test_tile_bounds_order():
    b = tile_bounds(37500, 25100, 16)
    assert b.west < b.east
    assert b.south < b.north


def test_viewport_range_covers_center():
    b = tile_bounds(37500, 25100, 16)
    lon = (b.west + b.east) / 2
    lat = (b.south + b.north) / 2
    min_col, max_col, min_row, max_row = viewport_col_row_range(lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001, 16)
    assert min_col <= 37500 <= max_col
    assert min_row <= 25100 <= max_row


def test_haversine_symmetry():
    from tammuz.utils.geo import pairwise_haversine_km

    a = (34.0, 39.0)
    b = (34.1, 39.0)
    d1 = pairwise_haversine_km(a, b)
    d2 = pairwise_haversine_km(b, a)
    assert math.isclose(d1, d2)
    assert 0 < d1 < 20
