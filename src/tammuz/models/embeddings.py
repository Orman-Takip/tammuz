"""DINOv2 embedding modeli.

DINOv2 (facebookresearch/dinov2) self-supervised bir goruntu modelidir;
uydu/hava goruntulerine de iyi genellesir. Herakles'in kullandigi ImageNet
ResNet18'in aksine, ayni araziyi farkli aydinlatma/kompresyon altinda "ayni"
olarak tanimakta cok daha saglamdir. Bu yuzden iki tile'in gercekten ayni
arazi mi yoksa farkli arazi mi oldugunu ayirt etmek icin ideal bir sinyaldir.

timm uzerinden yuklenir (model agirliklari Hugging Face Hub'dan inilir).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console

console = Console()


class DinoEmbedder:
    """Bir goruntuyu DINOv2 embedding vektorune ceviren sinif."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32):
        import timm
        import torch

        self.torch = torch
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        console.print(f"[cyan]DINOv2 yukleniyor:[/cyan] {model_name} ({self.device})")
        self.model = timm.create_model(model_name, pretrained=True)
        self.model = self.model.to(self.device).eval()
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        self.dim = self.model.num_features

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device and device != "auto":
            return device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self, path: Path) -> np.ndarray:
        with Image.open(path) as img:
            return np.asarray(self.transform(img.convert("RGB")))

    def embed_paths(self, paths: list[Path]) -> np.ndarray:
        """Verilen goruntulerin (N, D) boyutunda float32 embedding matrisini dondurur."""
        torch = self.torch
        out = np.empty((len(paths), self.dim), dtype=np.float32)
        for start in range(0, len(paths), self.batch_size):
            batch = paths[start : start + self.batch_size]
            tensors = [self._load(p) for p in batch]
            x = torch.from_numpy(np.stack(tensors)).to(self.device)
            with torch.inference_mode():
                features = self.model.forward_features(x)
                vec = features[:, 0].cpu().numpy()
            out[start : start + len(batch)] = vec
        return out

    def embed_one(self, path: Path) -> np.ndarray:
        return self.embed_paths([path])[0]

    def close(self) -> None:
        del self.model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Iki embedding vektoru arasindaki kosinus benzerligi (0..1)."""
    if a.ndim == 2:
        a = a[0]
    if b.ndim == 2:
        b = b[0]
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / norm)
