"""Gurultu eleme adimi.

DINOv2 benzerligi esiginin altinda kalan ciftler "semantik olarak degismis"
kabul edilir ve degisim adayi olarak secilir. Elesilenler `pairs.parquet`
icerisinde kayitli kalir (tekrarlanabilirlik icin), yalnizca secilenler
`changes.parquet` uzerine yazilir ve sonraki adimlar bu dosyayi kullanir.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from rich.console import Console

from tammuz.config import Settings

console = Console()


def run_filter(settings: Settings) -> Path:
    pairs_path = Path(settings.paths.pairs_parquet)
    if not pairs_path.exists():
        raise FileNotFoundError(f"once `tammuz pairs` calistirin: {pairs_path}")

    df = pl.read_parquet(pairs_path)
    if "dino_similarity" not in df.columns:
        raise FileNotFoundError("once `tammuz embed` calistirin: dino_similarity kolonu yok")

    threshold = settings.embed.similarity_threshold
    keep = (df["dino_similarity"].is_null() | (df["dino_similarity"] < threshold)) & (
        df["largest_run"] >= settings.filter.min_largest_run
    )

    if settings.filter.min_change_ratio > 0:
        keep = keep & (df["change_ratio"] >= settings.filter.min_change_ratio)

    changes = df.filter(keep).with_columns(
        pl.when(pl.col("dino_similarity").is_null())
        .then(pl.lit("no_embedding"))
        .otherwise(pl.lit("dino"))
        .alias("candidate_reason"),
        pl.lit(True).alias("is_candidate"),
    )

    out = Path(settings.paths.changes_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    changes.write_parquet(out)

    console.print(
        f"[green]filter:[/green] {df.height} ciftten {changes.height} degisim adayi secildi "
        f"(esik: dino_similarity < {threshold})"
    )
    return out
