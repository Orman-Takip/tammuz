"""Arazi ortusu siniflandirma ve degisim tipi adimi.

`changes.parquet` icindeki her aday icin once ve sonra gorselini siniflandirir,
`land_cover_from` / `land_cover_to` ve bunlardan turetilen
`change_type_tr` / `change_type_en` kolonlarini ekler.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from rich.console import Console

from tammuz.config import Settings
from tammuz.models.embeddings import DinoEmbedder
from tammuz.models.landcover import LandCoverClassifier, derive_change_type, train_probe

console = Console()


def run_classify(settings: Settings) -> Path:
    changes_path = Path(settings.paths.changes_parquet)
    if not changes_path.exists():
        raise FileNotFoundError(f"once `tammuz filter` calistirin: {changes_path}")

    changes = pl.read_parquet(changes_path)
    embedder = DinoEmbedder(settings.embed.model, settings.embed.device, settings.embed.batch_size)
    try:
        probe_path = Path(settings.classify.probe_path)
        if not probe_path.exists():
            train_probe(embedder, settings.classify, Path(settings.paths.datasets_dir))
        classifier = LandCoverClassifier(embedder, probe_path, confidence_floor=settings.classify.confidence_floor)

        images_dir = Path(settings.paths.images_dir)
        lc_from: list[str] = []
        lc_to: list[str] = []
        conf_from: list[float] = []
        conf_to: list[float] = []
        type_tr: list[str] = []
        type_en: list[str] = []

        for row in changes.iter_rows(named=True):
            before = row.get("before_image") or ""
            after = row.get("after_image") or ""
            fb = classifier.classify(images_dir / Path(before).name) if before else None
            fa = classifier.classify(images_dir / Path(after).name) if after else None
            lc_from.append(fb[0] if fb else "")
            lc_to.append(fa[0] if fa else "")
            conf_from.append(fb[1] if fb else 0.0)
            conf_to.append(fa[1] if fa else 0.0)
            ctype = derive_change_type(lc_from[-1], lc_to[-1])
            type_tr.append(ctype["tr"] if ctype else "")
            type_en.append(ctype["en"] if ctype else "")
    finally:
        embedder.close()

    changes = changes.with_columns(
        pl.Series("land_cover_from", lc_from),
        pl.Series("land_cover_to", lc_to),
        pl.Series("land_cover_conf_from", conf_from),
        pl.Series("land_cover_conf_to", conf_to),
        pl.Series("change_type_tr", type_tr),
        pl.Series("change_type_en", type_en),
    )
    changes.write_parquet(changes_path)

    labeled = (changes["land_cover_from"] != "") & (changes["land_cover_to"] != "")
    console.print(
        f"[green]classify:[/green] {changes.height} adaydan {int(labeled.sum())} cift etiketlendi "
        f"({labeled.mean():.1%})"
    )
    return changes_path
