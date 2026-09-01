"""Yapilandirma yukleme ve dogrulama.

default.yaml projeye gomulu temel yapilandirmadir. `config/local.yaml`
varsa uzerine biner (yerel denemeler icin, git'e girmez). Ayrica
`TAMMUZ_CONFIG` cevre degiskeniyle farkli bir dosya da isaret edilebilir.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from tammuz.paths import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
LOCAL_CONFIG = PROJECT_ROOT / "config" / "local.yaml"


class Paths(BaseModel):
    data_dir: str = "data"
    raw_db: str = "data/raw/herakles.db"
    models_dir: str = "data/models"
    datasets_dir: str = "data/datasets"
    processed_dir: str = "data/processed"
    images_dir: str = "data/processed/images"
    masks_dir: str = "data/processed/masks"
    lancedb_dir: str = "data/processed/lancedb"
    exports_dir: str = "data/processed/exports"
    pairs_parquet: str = "data/processed/pairs.parquet"
    embeddings_parquet: str = "data/processed/embeddings.parquet"
    changes_parquet: str = "data/processed/changes.parquet"


class Sync(BaseModel):
    url: str = ""
    ssh_host: str = "hetzner-ormantakip"
    ssh_db_path: str = "/opt/herakles/repo/data/herakles.db"


class Db(BaseModel):
    zoom: int = 16


class Embed(BaseModel):
    model: str = "vit_base_patch14_dinov2.lvd142m"
    batch_size: int = 32
    device: str = "auto"
    similarity_threshold: float = 0.92


class Filter(BaseModel):
    min_largest_run: int = 2
    min_change_ratio: float = 0.0
    keep_no_label: bool = True


class Classify(BaseModel):
    eurosat_repo_id: str = "tanganke/eurosat"
    eurosat_train_file: str = "data/train-00000-of-00001.parquet"
    eurosat_test_file: str = "data/test-00000-of-00001.parquet"
    probe_path: str = "data/models/eurosat_dinov2_probe.pt"
    probe_epochs: int = 12
    probe_lr: float = 0.003
    probe_batch_size: int = 256
    confidence_floor: float = 0.5


class Narrate(BaseModel):
    enabled: bool = False
    base_url: str = ""
    model: str = "deepseek-ai/DeepSeek-V4-Flash"
    api_key_env: str = "TAMMUZ_API_KEY"
    temperature: float = 0.1
    max_tokens: int = 400
    batch_size: int = 8
    two_images: bool = True


class Mask(BaseModel):
    cell_size: int = 16
    diff_threshold: int = 28
    min_area: int = 4


class ImportanceWeights(BaseModel):
    change_ratio: float = 0.3
    magnitude: float = 0.3
    land_impact: float = 0.2
    recency: float = 0.2


class Enrich(BaseModel):
    importance_weights: ImportanceWeights = Field(default_factory=ImportanceWeights)
    hotspot_eps_km: float = 15.0
    hotspot_min_samples: int = 5
    drop_ai_false: bool = False


class Vector(BaseModel):
    table: str = "changes"
    k_default: int = 6


class Serve(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8787
    max_features: int = 4000


class Settings(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    sync: Sync = Field(default_factory=Sync)
    db: Db = Field(default_factory=Db)
    embed: Embed = Field(default_factory=Embed)
    filter: Filter = Field(default_factory=Filter)
    classify: Classify = Field(default_factory=Classify)
    narrate: Narrate = Field(default_factory=Narrate)
    mask: Mask = Field(default_factory=Mask)
    enrich: Enrich = Field(default_factory=Enrich)
    vector: Vector = Field(default_factory=Vector)
    serve: Serve = Field(default_factory=Serve)


def _resolve(settings: Settings, base: Path) -> Settings:
    """Yollari proje kokune gore mutlak hale getirir."""
    for field in Paths.model_fields:
        current = getattr(settings.paths, field)
        if current and not Path(current).is_absolute():
            setattr(settings.paths, field, str((base / current).resolve()))
    settings.classify.probe_path = str((base / settings.classify.probe_path).resolve())
    return settings


def load_config() -> Settings:
    env_config = os.environ.get("TAMMUZ_CONFIG")
    base = Path(env_config).resolve() if env_config else DEFAULT_CONFIG
    if not base.exists():
        raise FileNotFoundError(f"yapilandirma dosyasi bulunamadi: {base}")

    with base.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    local = LOCAL_CONFIG
    if local.exists():
        with local.open(encoding="utf-8") as fh:
            overrides = yaml.safe_load(fh) or {}
        data = _deep_merge(data, overrides)

    settings = Settings.model_validate(data)
    return _resolve(settings, PROJECT_ROOT)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
