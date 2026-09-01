"""Proje kokune gore onemli dizinleri hesaplar.

Paket `src/` duzeninde kuruldugu icin dosya sistemi uzerindeki konum
paket konumundan cikarilir: proje koku = iki seviye yukari (src/tammuz -> repo koku).
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent


def project_root() -> Path:
    return PROJECT_ROOT


def data_dir() -> Path:
    return PROJECT_ROOT / "data"
