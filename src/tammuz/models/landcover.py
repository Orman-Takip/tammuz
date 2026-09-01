"""Arazi ortusu siniflandirmasi.

Strateji: DINOv2 embedding'lerini dondurup uzerine EuroSAT ile egitilmis
kucuk bir linear baslik takmak. EuroSAT, Sentinel-2 uydu goruntuleriyle
10 sinifli arazi ortusu veri setidir; DINOv2 ozellikleriyle birlikte bu
yaklasim, Herakles'in ResNet18 tabanli siniflandiricisindan cok daha
saglam etiketler uretir.

EuroSAT verisi Hugging Face Hub'dan (`tanganke/eurosat`) parquet olarak
indirilir; model agirliklari ilk calistirmada egitilir ve
`data/models/` altina kaydedilir.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()

# tanganke/eurosat etiket sirasi
LABELS = [
    "AnnualCrop",
    "Forest",
    "Shrubland",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Genis grup kumesi: degisim turu bu gruplar uzerinden turetilir.
GROUP = {
    "Forest": "dogal",
    "Shrubland": "dogal",
    "Pasture": "dogal",
    "AnnualCrop": "tarim",
    "PermanentCrop": "tarim",
    "Residential": "yapili",
    "Industrial": "yapili",
    "Highway": "yapili",
    "River": "su",
    "SeaLake": "su",
}

CHANGE_TYPES: dict[tuple[str, str], dict[str, str]] = {
    ("dogal", "tarim"): {"tr": "Tarıma açılma", "en": "Conversion to agriculture"},
    ("dogal", "yapili"): {"tr": "Doğal alanın yapılaşması", "en": "Natural area developed"},
    ("dogal", "su"): {"tr": "Su baskını", "en": "Flooding"},
    ("tarim", "yapili"): {"tr": "Tarım arazisinin yapılaşması", "en": "Farmland developed"},
    ("tarim", "dogal"): {"tr": "Doğallaşma", "en": "Reverting to natural"},
    ("tarim", "su"): {"tr": "Su baskını", "en": "Flooding"},
    ("yapili", "dogal"): {"tr": "Yeşillendirme", "en": "Greening"},
    ("yapili", "tarim"): {"tr": "Tarıma dönüşüm", "en": "Conversion to agriculture"},
    ("su", "dogal"): {"tr": "Su çekilmesi", "en": "Water recession"},
    ("su", "tarim"): {"tr": "Su çekilmesi", "en": "Water recession"},
    ("su", "yapili"): {"tr": "Dolgu/yapılaşma", "en": "Landfill and development"},
}

TURKISH_GROUP = {"dogal": "Doğal alan", "tarim": "Tarım", "yapili": "Yapılı alan", "su": "Su"}


def derive_change_type(from_label: str, to_label: str) -> dict[str, str] | None:
    """Iki arazi etiketi arasindaki gecisin insan-okur karsiligini dondurur."""
    if not from_label or not to_label:
        return None
    fg = GROUP.get(from_label)
    tg = GROUP.get(to_label)
    if not fg or not tg or fg == tg:
        return None
    return CHANGE_TYPES.get((fg, tg))


class LandCoverClassifier:
    """DINOv2 + EuroSAT linear baslik ile bir goruntuyu siniflandiran sinif."""

    def __init__(self, embedder, probe_path: str | Path, confidence_floor: float = 0.5):
        import torch

        self.torch = torch
        self.embedder = embedder
        self.confidence_floor = confidence_floor
        self.device = embedder.device
        self.linear = torch.nn.Linear(embedder.dim, len(LABELS)).to(self.device)
        state = torch.load(probe_path, map_location=self.device)
        self.linear.load_state_dict(state)
        self.linear.eval()

    def classify(self, image_path: str | Path) -> tuple[str, float] | None:
        """Goruntuyu siniflandirir; guven esigini gecerse (etiket, guven) dondurur."""
        vec = self.embedder.embed_one(Path(image_path))
        x = self.torch.from_numpy(vec).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            logits = self.linear(x)
            probs = self.torch.softmax(logits, dim=1)[0].cpu().numpy()
        label_idx = int(np.argmax(probs))
        confidence = float(probs[label_idx])
        if confidence < self.confidence_floor:
            return None
        return LABELS[label_idx], confidence


def train_probe(embedder, config, work_dir: Path) -> Path:
    """EuroSAT uzerinde linear baslik egitir ve agirligi kaydeder. Yolu dondurur."""
    from tammuz.models.eurosat import load_eurosat

    console.print("[cyan]EuroSAT linear baslik egitiliyor...[/cyan]")
    train_images, train_labels = load_eurosat(
        repo_id=config.eurosat_repo_id,
        filename=config.eurosat_train_file,
        cache_dir=work_dir,
    )
    test_images, test_labels = load_eurosat(
        repo_id=config.eurosat_repo_id,
        filename=config.eurosat_test_file,
        cache_dir=work_dir,
    )
    console.print(f"egitim: {len(train_images)} ornek, test: {len(test_images)} ornek")

    train_vecs = embedder.embed_paths(train_images)
    test_vecs = embedder.embed_paths(test_images)

    import torch

    device = embedder.device
    linear = torch.nn.Linear(embedder.dim, len(LABELS)).to(device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=config.probe_lr)
    criterion = torch.nn.CrossEntropyLoss()

    x_train = torch.from_numpy(train_vecs).to(device)
    y_train = torch.from_numpy(np.asarray(train_labels)).to(device)
    x_test = torch.from_numpy(test_vecs).to(device)
    y_test = torch.from_numpy(np.asarray(test_labels)).to(device)

    n = len(train_vecs)
    for epoch in range(config.probe_epochs):
        perm = torch.randperm(n, device=device)
        total, correct = 0, 0
        for start in range(0, n, config.probe_batch_size):
            idx = perm[start : start + config.probe_batch_size]
            logits = linear(x_train[idx])
            loss = criterion(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += len(idx)
            correct += int((logits.argmax(1) == y_train[idx]).sum().item())
        train_acc = correct / total

        with torch.inference_mode():
            test_logits = linear(x_test)
            test_acc = float((test_logits.argmax(1) == y_test).float().mean().item())
        console.print(f"epoch {epoch + 1}/{config.probe_epochs} egitim {train_acc:.4f} test {test_acc:.4f}")

    probe_path = Path(config.probe_path)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(linear.state_dict(), probe_path)
    console.print(f"[green]Linear baslik kaydedildi:[/green] {probe_path} (test dogrulugu {test_acc:.4f})")
    return probe_path


def class_labels_json() -> str:
    return json.dumps(LABELS, ensure_ascii=False)
