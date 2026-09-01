"""Goruntu byte'larini disk'e yazma yardimcilari."""

from __future__ import annotations

from pathlib import Path


def detect_extension(data: bytes) -> str:
    """Bayt imzasindan goruntu uzantisini tahmin eder (jpeg/png/webp)."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "jpg"


def write_image(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
