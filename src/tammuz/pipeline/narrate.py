"""Vizyon LLM ile sebep ve dogrulama adimi (istege bagli).

Once/sonra uydu gorselini OpenAI uyumlu bir vizyon modeline gonderir ve
degisimin gercek olup olmadigini, turunu ve insan-okur sebebini yapilandirilmis
JSON olarak alir. OpenAI uyumlu her uc nokta calisir: DeepSeek, Ollama, vLLM,
LM Studio gibi.

Varsayilan olarak kapalidir (`narrate.enabled: false`); acmak icin:
  config/local.yaml -> narrate.enabled: true ve api key ayarlayin.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import polars as pl
from rich.console import Console

from tammuz.config import Settings

console = Console()

SYSTEM_PROMPT = (
    "Sen bir uydu goruntusu analizcisisin. Sana ayni konumun iki farkli tarihli "
    "uydu goruntusu verilecek (once ve sonra). Gorevin: aradaki farkin gercek bir "
    "arazi degisimi mi yoksa gurultu mu (mevsimsel, bulut, cekim farki, kompresyon) "
    "olduguna karar vermek. Yalnizca asagidaki JSON semasinda yanit ver: "
    '{"is_real": true|false, "change_type": "construction|agriculture|deforestation|'
    'water|mining|other", "cause_tr": "kisa turkce aciklama", "confidence": 0.0}'
)

VALID_TYPES = {"construction", "agriculture", "deforestation", "water", "mining", "other"}


def _image_data_url(path: Path) -> str:
    ext = path.suffix.lstrip(".") or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/{ext};base64,{data}"


def _parse_response(text: str) -> dict | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    if payload.get("change_type") not in VALID_TYPES:
        payload["change_type"] = ""
    payload["is_real"] = bool(payload.get("is_real"))
    payload["confidence"] = float(payload.get("confidence") or 0.0)
    return payload


def run_narrate(settings: Settings, limit: int | None = None) -> Path:
    if not settings.narrate.enabled:
        console.print("[yellow]narrate kapali (narrate.enabled: false). Atlaniyor.[/yellow]")
        return Path(settings.paths.changes_parquet)

    import openai

    changes_path = Path(settings.paths.changes_parquet)
    if not changes_path.exists():
        raise FileNotFoundError(f"once `tammuz filter` calistirin: {changes_path}")

    changes = pl.read_parquet(changes_path)
    if limit:
        changes = changes.head(limit)

    client = openai.OpenAI(
        base_url=settings.narrate.base_url or None,
        api_key=os_env(settings.narrate.api_key_env) or "sk-none",
    )

    images_dir = Path(settings.paths.images_dir)
    results: list[dict] = []
    for row in changes.iter_rows(named=True):
        event_id = int(row["event_id"])
        content: list[dict] = [
            {"type": "text", "text": "Asagidaki iki uydu goruntusu arasindaki degisimi degerlendir."}
        ]
        for side in ("before_image", "after_image"):
            rel = row.get(side) or ""
            if not rel:
                continue
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(images_dir / Path(rel).name)}})

        try:
            response = client.chat.completions.create(
                model=settings.narrate.model,
                temperature=settings.narrate.temperature,
                max_tokens=settings.narrate.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            )
            parsed = _parse_response(response.choices[0].message.content or "")
            results.append(
                {
                    "event_id": event_id,
                    "ai_is_real": bool(parsed["is_real"]) if parsed else None,
                    "ai_change_type": parsed["change_type"] if parsed else "",
                    "ai_cause_tr": parsed.get("cause_tr", "") if parsed else "",
                    "ai_confidence": parsed["confidence"] if parsed else 0.0,
                    "ai_model": settings.narrate.model,
                }
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]LLM hatasi event {event_id}: {exc}[/red]")
            results.append(
                {
                    "event_id": event_id,
                    "ai_is_real": None,
                    "ai_change_type": "",
                    "ai_cause_tr": "",
                    "ai_confidence": 0.0,
                    "ai_model": settings.narrate.model,
                }
            )

    result_df = pl.DataFrame(results)
    changes = changes.join(result_df, on="event_id", how="left")
    changes.write_parquet(changes_path)

    processed = changes["ai_is_real"].is_not_null().sum()
    console.print(f"[green]narrate:[/green] {int(processed)} degisim analiz edildi")
    return changes_path


def os_env(name: str) -> str:
    import os

    return os.environ.get(name, "")
