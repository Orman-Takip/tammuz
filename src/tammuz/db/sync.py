"""Ham Herakles veritabanini indirme / cekme.

Iki yol desteklenir:

- `--url`: Herakles DB dosyasinin public bir konumu (GitHub Release veya
  baska bir indirme adresi). Acik kaynak kullanicilar icin onerilen yol.
- `--ssh`: `~/.ssh/config` uzerinden kendi sunucumuzdan cekim. Bu yol
  ozeldir ve yalnizca ekip icinde calisir.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from rich.console import Console

from tammuz.config import Settings

console = Console()


def sync_db(settings: Settings, url: str | None = None, ssh_host: str | None = None) -> Path:
    """Veritabanini `paths.raw_db` konumuna indirir; varligini dogrular."""
    target = Path(settings.paths.raw_db)
    target.parent.mkdir(parents=True, exist_ok=True)

    source_url = url or settings.sync.url
    host = ssh_host or settings.sync.ssh_host

    if source_url:
        _download(source_url, target)
    elif host:
        _pull_via_ssh(host, settings.sync.ssh_db_path, target)
    else:
        raise RuntimeError(
            "kaynak yok: config/sync.url bos ve --ssh verilmedi. "
            "ya config/local.yaml icinde sync.url ayarlayin ya da --ssh HOST kullanin."
        )

    _validate(target)
    return target


def _download(url: str, target: Path) -> None:
    console.print(f"[cyan]Indiriliyor:[/cyan] {url}")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"gecersiz indirme adresi: {url}")

    with tempfile.NamedTemporaryFile(prefix="herakles-", suffix=".db", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with console.status("indiriliyor...") as status:
                written = 0
                with tmp_path.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        written += len(chunk)
                        if total:
                            status.update(f"indiriliyor... {written / 1e9:.2f} / {total / 1e9:.2f} GB")
        shutil.move(str(tmp_path), target)
    finally:
        tmp_path.unlink(missing_ok=True)


def _pull_via_ssh(host: str, remote_path: str, target: Path) -> None:
    console.print(f"[cyan]Sunucudan cekiliyor:[/cyan] {host}:{remote_path}")
    cmd = ["scp", f"{host}:{remote_path}", str(target)]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(f"scp basarisiz: {' '.join(cmd)}")


def _validate(path: Path) -> None:
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.execute("SELECT COUNT(*) FROM change_events")
        conn.close()
    except sqlite3.Error as exc:
        raise RuntimeError(f"indirilen dosya gecerli bir Herakles DB degil: {exc}") from exc

    size = path.stat().st_size
    console.print(f"[green]Hazir:[/green] {path} ({size / 1e9:.2f} GB)")
