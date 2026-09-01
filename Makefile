PYTHON ?= python3
VENV   := .venv
PIP    := $(VENV)/bin/pip
RUN    := $(VENV)/bin/tammuz

.DEFAULT_GOAL := help

.PHONY: help setup dev-install install ml install-ml sync db pairs embed filter classify mask narrate enrich vector export pipeline serve test lint fmt models clean release

help: ## Bu yardimi goster
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

setup: ## Sanal ortam + tum bagimliliklar
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[ml,dev]"

dev-install: ## Sadece gelistirme bagimliliklari (ML olmadan)
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -e ".[dev]"

sync: ## Ham herakles.db dosyasini indir/cikar
	$(RUN) sync

pairs: ## Degisim ciftlerini ve gorselleri cikar
	$(RUN) pairs

embed: ## DINOv2 embedding + benzerlik skoru
	$(RUN) embed

filter: ## Gurultu elemesi, degisim adaylarini sec
	$(RUN) filter

classify: ## Arazi ortusu siniflandirmasi + degisim tipi
	$(RUN) classify

mask: ## Degisim bolgesi poligonlari
	$(RUN) mask

narrate: ## (Istege bagli) Vizyon LLM ile sebep/aciklama
	$(RUN) narrate

enrich: ## Episode, hotspot, onem skoru
	$(RUN) enrich

vector: ## LanceDB vektor indeksi
	$(RUN) vector

export: ## GeoJSON / NDJSON / GeoParquet export
	$(RUN) export

serve: ## Harita uygulamasini baslat
	$(RUN) serve

pipeline: sync pairs embed filter classify mask enrich vector export ## Tum islem hattini uc uca calistir
	@echo "Islem hatti tamamlandi."

test: ## Birim testleri
	$(VENV)/bin/pytest

lint: ## Ruff ile statik kontrol
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests

fmt: ## Ruff formatla
	$(VENV)/bin/ruff format src tests
	$(VENV)/bin/ruff check --fix src tests

models: ## Model agirliklarini hazirla (ML ortaminda)
	$(RUN) models

release: ## GitHub Release icin arsiv uret
	$(RUN) export --release

clean: ## Uretilen verileri temizle
	rm -rf data/processed data/models data/datasets
