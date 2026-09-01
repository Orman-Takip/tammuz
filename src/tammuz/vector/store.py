"""LanceDB vektor indeksi.

Her degisimin "sonra" (degisim sonrasi) gorselinin DINOv2 embedding'i vektor
indeksine konur. Boylece kullanicilar bir degisime benzer diger degisimleri
kullanabilir: benzer sonuc durumlari (ornegin hepsi benzer sekilde yapilasmis
araziler) birlikte bulunur.
"""

from __future__ import annotations

from pathlib import Path

import lancedb
import numpy as np
import polars as pl
import pyarrow as pa
from rich.console import Console

from tammuz.config import Settings

console = Console()

METADATA_COLUMNS = [
    "tile_col",
    "tile_row",
    "centroid_lon",
    "centroid_lat",
    "from_date",
    "to_date",
    "change_type_tr",
    "change_type_en",
    "importance",
    "episode_id",
    "has_mask",
]


def _embeddings_map(embeddings_path: Path) -> dict[str, np.ndarray]:
    df = pl.read_parquet(embeddings_path)
    out: dict[str, np.ndarray] = {}
    for row in df.iter_rows(named=True):
        out[str(row["image"])] = np.asarray(row["embedding"], dtype=np.float32)
    return out


def build_index(settings: Settings) -> Path:
    changes_path = Path(settings.paths.changes_parquet)
    if not changes_path.exists():
        raise FileNotFoundError(f"once `tammuz enrich` calistirin: {changes_path}")

    changes = pl.read_parquet(changes_path)
    emb = _embeddings_map(Path(settings.paths.embeddings_parquet))

    records = []
    for row in changes.iter_rows(named=True):
        after_name = Path(row["after_image"] or "").name
        vec = emb.get(after_name)
        if vec is None:
            continue
        records.append(
            {
                "event_id": int(row["event_id"]),
                "vector": vec,
                **{c: row.get(c) for c in METADATA_COLUMNS},
            }
        )

    if not records:
        raise RuntimeError("embeddings.parquet'ta hic eslesen goruntu yok; once `tammuz embed` calistirin.")

    table_path = Path(settings.paths.lancedb_dir)
    table_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(table_path))
    table_name = settings.vector.table
    if table_name in db.table_names():
        db.drop_table(table_name)

    data = pa.table(
        {
            "event_id": pa.array([r["event_id"] for r in records], pa.int64()),
            "vector": pa.array([r["vector"] for r in records], type=pa.list_(pa.float32(), len(records[0]["vector"]))),
            **{c: pa.array([r.get(c) for r in records]) for c in METADATA_COLUMNS},
        }
    )
    table = db.create_table(table_name, data)
    console.print(f"[green]vector:[/green] {table.count_rows()} kayit LanceDB'ye yazildi")
    return table_path


def search_similar(settings: Settings, event_id: int, k: int) -> list[dict]:
    """Bir degisime benzer degisimleri dondurur (once index kurulmus olmali)."""
    import pyarrow.compute as pc

    db = lancedb.connect(str(Path(settings.paths.lancedb_dir)))
    table = db.open_table(settings.vector.table)

    arrow = table.to_arrow()
    rows = arrow.filter(pc.equal(arrow["event_id"], event_id)).to_pylist()
    if not rows:
        return []
    vec = rows[0]["vector"]

    hits = table.search(vec).limit(k).to_list()
    return [h for h in hits if h["event_id"] != event_id][:k]
