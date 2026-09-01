"""Tammuz komut satiri arayuzu."""

from __future__ import annotations

import typer
import uvicorn

from tammuz import __version__
from tammuz.config import load_config

app = typer.Typer(help="Tammuz: arazi degisimi isleme ve yayinlama araci.", no_args_is_help=True)


def _settings():
    return load_config()


@app.command()
def sync(
    url: str | None = typer.Option(None, help="Ham DB indirme adresi (config/sync.url'u ezer)."),
    ssh: str | None = typer.Option(None, help="Sunucudan cekilecek SSH host adi (config/sync.ssh_host'u ezer)."),
):
    """Ham Herakles veritabanini indir."""
    from tammuz.db.sync import sync_db

    sync_db(_settings(), url=url, ssh_host=ssh)


@app.command()
def pairs():
    """Degisim ciftlerini ve gorselleri cikar."""
    from tammuz.pipeline.pairs import build_pairs

    build_pairs(_settings())


@app.command()
def embed():
    """DINOv2 embedding ve benzerlik skoru hesapla."""
    from tammuz.pipeline.embed import run_embed

    run_embed(_settings())


@app.command()
def filter():
    """Gurultu ele ve degisim adaylarini sec."""
    from tammuz.pipeline.filter import run_filter

    run_filter(_settings())


@app.command()
def classify():
    """Arazi ortusu siniflandirmasi ve degisim tipi."""
    from tammuz.pipeline.classify import run_classify

    run_classify(_settings())


@app.command()
def mask():
    """Degisim bolgesi poligonlarini cikar."""
    from tammuz.pipeline.mask import run_mask

    run_mask(_settings())


@app.command()
def narrate(limit: int | None = typer.Option(None, help="Istege bagli: yalnizca ilk N kaydi isle.")):
    """Vizyon LLM ile sebep ve dogrulama (istege bagli)."""
    from tammuz.pipeline.narrate import run_narrate

    run_narrate(_settings(), limit=limit)


@app.command()
def enrich():
    """Episode, onem skoru ve hotspot kumeleme."""
    from tammuz.pipeline.enrich import run_enrich

    run_enrich(_settings())


@app.command()
def vector():
    """LanceDB vektor indeksini kur."""
    from tammuz.vector.store import build_index

    build_index(_settings())


@app.command()
def export(release: bool = typer.Option(False, help="GitHub Release icin arsiv de uret.")):
    """GeoJSON, NDJSON ve GeoParquet cikar."""
    from tammuz.pipeline.export import run_export

    run_export(_settings(), release=release)


@app.command()
def models():
    """Model agirliklarini onceden indir (embed/classify'nin ilk calistirmasinda da yapilir)."""
    from tammuz.models.embeddings import DinoEmbedder

    s = _settings()
    embedder = DinoEmbedder(s.embed.model, s.embed.device, s.embed.batch_size)
    embedder.close()
    typer.echo("DINOv2 modeli hazir.")


@app.command()
def pipeline(
    skip: list[str] = typer.Option(
        [], "--skip", help="Atlanacak adim (ornegin --skip mask). Birden fazla verilebilir."
    ),
):
    """Islem hattini uc uca calistir (sync, pairs, embed, filter, classify, mask, enrich, vector, export)."""
    from tammuz.pipeline.classify import run_classify
    from tammuz.pipeline.embed import run_embed
    from tammuz.pipeline.enrich import run_enrich
    from tammuz.pipeline.export import run_export
    from tammuz.pipeline.filter import run_filter
    from tammuz.pipeline.mask import run_mask
    from tammuz.pipeline.pairs import build_pairs
    from tammuz.vector.store import build_index

    settings = _settings()
    steps = {
        "sync": (lambda: _sync_only(settings)),
        "pairs": lambda: build_pairs(settings),
        "embed": lambda: run_embed(settings),
        "filter": lambda: run_filter(settings),
        "classify": lambda: run_classify(settings),
        "mask": lambda: run_mask(settings),
        "enrich": lambda: run_enrich(settings),
        "vector": lambda: build_index(settings),
        "export": lambda: run_export(settings),
    }
    for name, step in steps.items():
        if name in skip:
            typer.echo(f"atlanan adim: {name}")
            continue
        typer.echo(f"==> {name}")
        step()


def _sync_only(settings):
    from tammuz.db.sync import sync_db

    return sync_db(settings)


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Dinlenecek adres."),
    port: int | None = typer.Option(None, help="Dinlenecek port."),
    reload: bool = typer.Option(False, help="Gelistirme modu (uvicorn --reload)."),
):
    """Harita uygulamasini baslat."""
    s = _settings()
    uvicorn.run(
        "tammuz.serve.app:create_app",
        host=host or s.serve.host,
        port=port or s.serve.port,
        reload=reload,
        factory=True,
    )


@app.command()
def version():
    """Surum bilgisi."""
    typer.echo(f"tammuz {__version__}")


if __name__ == "__main__":
    app()
