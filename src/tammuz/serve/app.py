"""Harita uygulamasi (FastAPI).

`changes.parquet` + gorseller + LanceDB vektor indeksi uzerinden:
degisim haritasi, tur/zaman filtreleri, tile detayi, benzer degisim arama.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tammuz.config import Settings, load_config
from tammuz.serve.queries import ChangesStore
from tammuz.vector.store import search_similar

STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_config()
    store = ChangesStore(settings)
    processed_dir = Path(settings.paths.processed_dir)

    app = FastAPI(title="Tammuz", version="0.1.0")
    app.state.settings = settings
    app.state.store = store
    app.state.processed_dir = processed_dir

    @app.get("/api/overview")
    def overview() -> dict:
        return store.overview()

    @app.get("/api/count")
    def count(
        min_col: int,
        max_col: int,
        min_row: int,
        max_row: int,
        types: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> JSONResponse:
        return JSONResponse(
            {"count": store.count(min_col, max_col, min_row, max_row, _split(types), from_date, to_date)}
        )

    @app.get("/api/changes")
    def changes(
        min_col: int,
        max_col: int,
        min_row: int,
        max_row: int,
        types: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        limit: int = Query(default=0, le=20000),
    ) -> dict:
        limit = limit or settings.serve.max_features
        return store.features(min_col, max_col, min_row, max_row, _split(types), from_date, to_date, limit=limit)

    @app.get("/api/grid")
    def grid(
        min_col: int,
        max_col: int,
        min_row: int,
        max_row: int,
        cell: int,
        types: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict:
        return store.grid(min_col, max_col, min_row, max_row, cell, _split(types), from_date, to_date)

    @app.get("/api/event/{event_id}")
    def event(event_id: int) -> dict:
        feat = store.event(event_id)
        if not feat:
            raise HTTPException(status_code=404, detail="event bulunamadi")
        return feat

    @app.get("/api/tile/{col}/{row}")
    def tile_events(col: int, row: int) -> dict:
        return store.events_for_tile(col, row)

    @app.get("/api/image")
    def image(event_id: int, side: str = "before") -> FileResponse:
        feat = store.event(event_id)
        if not feat:
            raise HTTPException(status_code=404, detail="event bulunamadi")
        key = "before_image" if side == "before" else "after_image"
        rel = feat["properties"].get(key) or ""
        if not rel:
            raise HTTPException(status_code=404, detail="gorsel yok")
        path = processed_dir / rel
        if not path.exists():
            raise HTTPException(status_code=404, detail="gorsel dosyasi yok")
        return FileResponse(path)

    @app.get("/api/mask")
    def mask(event_id: int) -> FileResponse:
        feat = store.event(event_id)
        if not feat:
            raise HTTPException(status_code=404, detail="event bulunamadi")
        rel = feat["properties"].get("mask_image") or ""
        if not rel:
            raise HTTPException(status_code=404, detail="maske yok")
        path = processed_dir / rel
        if not path.exists():
            raise HTTPException(status_code=404, detail="maske dosyasi yok")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/similar")
    def similar(event_id: int, k: int = 0) -> list[dict]:
        k = k or settings.vector.k_default
        hits = search_similar(settings, event_id, k)
        out = []
        for h in hits:
            feat = store.event(int(h["event_id"]))
            if feat:
                out.append(feat)
        return out

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def _split(types: str | None) -> list[str] | None:
    if not types:
        return None
    return [t for t in types.split(",") if t]
