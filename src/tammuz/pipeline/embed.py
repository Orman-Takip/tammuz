"""DINOv2 embedding ve benzerlik skoru adimi.

`pairs.parquet` icindeki tum goruntuler icin embedding hesaplar, her ciftin
once/sonra kosinus benzerligini cikarir ve `pairs.parquet` uzerine
`dino_similarity` kolonunu yazar. Benzerlik 1.0'a yakin ise iki goruntu
semantik olarak ayni arazi demektir (gurultu), dusuk ise gercek bir degisim
olabilir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console

from tammuz.config import Settings
from tammuz.models.embeddings import DinoEmbedder, cosine_similarity

console = Console()


def _unique_image_paths(pairs_path: Path, images_dir: Path) -> list[Path]:
    df = pl.read_parquet(pairs_path)
    cols = [c for c in ("before_image", "after_image") if c in df.columns]
    flat = df.select(cols).to_dict(as_series=False)
    seen: set[str] = set()
    paths: list[Path] = []
    for col in cols:
        for rel in flat[col]:
            if rel and rel not in seen:
                seen.add(rel)
                paths.append(images_dir / Path(rel).name)
    return paths


def run_embed(settings: Settings) -> Path:
    pairs_path = Path(settings.paths.pairs_parquet)
    if not pairs_path.exists():
        raise FileNotFoundError(f"once `tammuz pairs` calistirin: {pairs_path}")

    images_dir = Path(settings.paths.images_dir)
    embeddings_path = Path(settings.paths.embeddings_parquet)

    # Onceden yapilmis hesaplari one ile atla (devam ettirilebilirlik).
    cached: dict[str, np.ndarray] = {}
    if embeddings_path.exists():
        emb_df = pl.read_parquet(embeddings_path)
        for row in emb_df.iter_rows(named=True):
            cached[str(row["image"])] = np.asarray(row["embedding"], dtype=np.float32)

    unique = _unique_image_paths(pairs_path, images_dir)
    to_compute = [p for p in unique if p.name not in cached]
    console.print(f"toplam benzersiz gorsel: {len(unique)}, yeni hesaplanacak: {len(to_compute)}")

    embedder = DinoEmbedder(settings.embed.model, settings.embed.device, settings.embed.batch_size)
    try:
        if to_compute:
            vecs = embedder.embed_paths(to_compute)
            new_rows = [
                {"image": p.name, "embedding": vec.tolist(), "dim": embedder.dim}
                for p, vec in zip(to_compute, vecs, strict=False)
            ]
            new_df = pl.DataFrame(new_rows)
            combined = pl.concat([emb_df, new_df]) if embeddings_path.exists() and not emb_df.is_empty() else new_df
            combined.write_parquet(embeddings_path)
            for p, vec in zip(to_compute, vecs, strict=False):
                cached[p.name] = vec
    finally:
        embedder.close()

    # Cift benzerliklerini hesapla.
    pairs = pl.read_parquet(pairs_path)
    sims = []
    for row in pairs.iter_rows(named=True):
        before = cached.get(Path(row["before_image"]).name) if row["before_image"] else None
        after = cached.get(Path(row["after_image"]).name) if row["after_image"] else None
        sims.append(cosine_similarity(before, after) if before is not None and after is not None else None)

    pairs = pairs.with_columns(pl.Series("dino_similarity", sims, dtype=pl.Float64))
    pairs.write_parquet(pairs_path)

    valid = [s for s in sims if s is not None]
    if valid:
        arr = np.asarray(valid)
        console.print(
            f"[green]embed:[/green] dino_similarity ort {arr.mean():.4f}, medyan {np.median(arr):.4f}, "
            f"<{settings.embed.similarity_threshold}: {(arr < settings.embed.similarity_threshold).mean():.1%}"
        )
    return embeddings_path
