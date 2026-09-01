"""EuroSAT veri setini Hugging Face Hub'dan indirip goruntulere acan yardimci modul.

`tanganke/eurosat` veri seti parquet dosyalari olarak dagitilir; goruntu
kolonu HF datasets'in standart `{bytes, path}` struct'idir. Bu modul parquet'i
pyarrow ile okur, goruntuleri disk'e JPEG olarak cikarir ve (yol, etiket)
listesi dondurur. Boylece goruntuler dogrudan DINOv2 embedder'a verilebilir.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

from tammuz.models.landcover import LABELS


def load_eurosat(repo_id: str, filename: str, cache_dir: Path) -> tuple[list[Path], list[int]]:
    """EuroSAT'in bir parquet bolumunu indirir ve (goruntu_yollari, etiketler) dondurur."""
    cache_dir = Path(cache_dir)
    parquet_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", cache_dir=str(cache_dir))

    table = pq.read_table(parquet_path)
    images = table.column("image").to_pylist()
    labels = table.column("label").to_pylist()

    split = "train" if "train" in filename else "test"
    out_dir = cache_dir / "eurosat" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    label_values: list[int] = []
    for idx, (raw, label) in enumerate(
        tqdm(zip(images, labels, strict=False), total=len(images), desc=f"eurosat/{split}")
    ):
        payload = raw.get("bytes") if isinstance(raw, dict) else raw
        if not payload:
            continue
        with Image.open(io.BytesIO(payload)) as img:
            rgb = img.convert("RGB")
            out_path = out_dir / f"{idx:05d}_{LABELS[label]}.jpg"
            if not out_path.exists():
                rgb.save(out_path, "JPEG", quality=92)
        paths.append(out_path)
        label_values.append(int(label))

    return paths, np.asarray(label_values)
