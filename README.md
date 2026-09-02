# Tammuz

<p align="center">
  <img src="docs/assets/tammuz.png" alt="Tammuz logosu" width="200">
</p>

Adını, her yıl ölüp baharla yeniden doğan Mezopotamya bereket tanrısından alır. Tıpkı tanrı gibi
bu sistem de toprağın ölümünü ve dirilişini kayıt altına alır: orman kesilir, tarla asfalta
döner, kıyı doldurulur, yıkılan bir yer yeniden yeşerir. [Tammuz'un hikayesini oku](STORY.md)

Türkiye'de Esri Wayback uydu görüntüleri üzerinden otomatik tespit edilen arazi değişimlerini
(Herakles projesinin çıktısı) işleyen, sınıflandıran, filtreleyen ve yayınlayan açık kaynak araç.

Herakles'in tespit ettiği ham değişim kayıtları, web'de çalışan hafif modellerle üretildiği için
önemli ölçüde gürültü içerir (mevsimsel döngü, bulut, çekim farkı, kompresyon). Tammuz, bu ham
kayıtları lokalde çok daha güçlü modellerle yeniden işler; gürültünün içindeki gerçek değişimi
ayıklar ve ona yeniden anlam kazandırır:

1. **DINOv2** ile her değişimin önce/sonra görüntüsünü gömüntüleyip semantik benzerliği hesaplar,
   gürültüyü eler.
2. **DINOv2 + EuroSAT** ile arazi örtüsünü sınıflandırır ve değişim türünü türetir.
3. İsteğe bağlı **vizyon LLM** (DeepSeek, Ollama, vLLM, LM Studio) ile değişimin sebebini ve
   gerçekliğini doğrular.
4. Değişim bölgesini piksel farkından poligona çevirir.
5. Episode, önem skoru ve hotspot kümeleme ile zenginleştirir.
6. Sonucu **GeoJSON, NDJSON, GeoParquet** ve **LanceDB vektör indeksi** olarak yayınlar.
7. Bir **harita uygulaması** sunar: tür, tarih ve zaman filtreleriyle tüm değişimler.

## Neden Tammuz?

Mezopotamya'da Tammuz her yıl ölür ve baharda yeniden doğar; toprak onunla kurur, onunla yeşerir.
Bu sistem de aynı döngüyü izler. Herakles, uydu görüntülerinde toprağın "ölümünü" (kesilen orman,
asfalta dönen tarla, dolan kıyı) bulur; Tammuz ise bu ham kayıtları işleyerek onları yeniden
"diriltir": gürültüyü eler, değişimin türünü ve sebebini söyler, insanların görebileceği bir
haritaya dönüştürür. Mevsim-eşleştirilmiş yıllık karşılaştırmalar da Dumuzid'in yıllık
ölüm-diriliş ritminin ta kendisidir. Mitin tamamı: [STORY.md](STORY.md)

## Kurulum

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[ml,dev]"   # ML adımları için .[ml], sadece sunucu için .[dev]
```

Makefile üzerinden:

```bash
make setup          # sanal ortam + tum bagimliliklar
```

## Kullanım

Önce ham Herakles veritabanını indirin (üç yoldan biri):

```bash
bash scripts/fetch-db.sh                    # GitHub Release'ten parcali indirme (oneri, ~11 GB)
tammuz sync --url https://ornek-adres/herakles.db     # public konum
tammuz sync --ssh hetzner-ormantakip                  # kendi sunucumuzdan
```

Ham veritabanı, GitHub Releases üzerinden 2 GB'lık parçalar halinde yayınlanır
(`raw-db-*` etiketleri). `scripts/fetch-db.sh` parçaları indirip birleştirir ve
sha256 ile doğrular; böylece aynı komut hem yerelde hem bulutta çalışır.

Sonra işlem hattını uçtan uca çalıştırın:

```bash
tammuz pipeline
```

Veya adım adım (her adım bir öncekine dayanır):

```bash
tammuz pairs       # ciftleri ve gorselleri cikar
tammuz embed       # DINOv2 embedding + benzerlik
tammuz filter      # gurultu eleme
tammuz classify    # arazi ortusu + degisim turu
tammuz mask        # degisim bolgesi poligonu
tammuz narrate     # (istege bagli) vizyon LLM
tammuz enrich      # episode, onem, hotspot
tammuz vector      # LanceDB indeksi
tammuz export      # GeoJSON/NDJSON/GeoParquet
```

Harita uygulamasını başlatın:

```bash
tammuz serve
# http://127.0.0.1:8787
```

Detaylı dokümantasyon: [docs/](docs/)

## Mimari özeti

```
herakles.db (ham)  ->  pairs.parquet  ->  changes.parquet  ->  exports/  +  lancedb/  ->  web uygulamasi
       sync                pairs            embed/filter/...      export/vector            serve
```

Veri modeli, işlem hattı adımları ve sunucu katmanı hakkında ayrıntılar için:
[docs/architecture.md](docs/architecture.md), [docs/pipeline.md](docs/pipeline.md),
[docs/data-model.md](docs/data-model.md), [docs/serving.md](docs/serving.md).

## Lisans

Kod MIT lisansı ile dağıtılır. İşlenmiş veri (export çıktıları) ayrıca CC BY 4.0 ile yayınlanır;
bkz. her release arşivinin içindeki lisans notu.

## Katkı

- Kod kalitesi: `ruff` (lint + format) ve `pytest` CI'da çalışır.
- Yeni bir model/çıktı eklerken `tests/` altına sentetik veriyle test ekleyin.
- İşlem hattı adımları `src/tammuz/pipeline/` altında, birbirinden bağımsız fonksiyonlardır.
