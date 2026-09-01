# Mimari

Tammuz, Herakles'in ürettiği ham arazi değişimi kayıtlarını işleyen, zenginleştiren ve yayınlayan
uc uca bir hattır. Bu belge, hattın genel akışını ve her katmanın sorumluluğunu anlatır.

## Genel akış

```
Herakles DB (ham)
      |
      | tammuz sync   (SQLite dosyasini indir)
      v
pairs.parquet + data/processed/images
      |
      | tammuz embed  (DINOv2 embedding + kosinus benzerligi)
      v
pairs.parquet + dino_similarity
      |
      | tammuz filter (gurultu eleme)
      v
changes.parquet
      |
      | classify -> mask -> narrate -> enrich
      v
changes.parquet (zenginlestirilmis)
      |
      +-> export   -> degisimler.geojson / .ndjson / .geoparquet / summary.json
      +-> vector   -> lancedb/ (vektor indeksi)
      +-> serve    -> FastAPI + Leaflet harita uygulamasi
```

## Katmanlar

### 1. Veri katmani (`src/tammuz/db/`)

- `sync.py`: Ham `herakles.db` dosyasını ya public bir adresten (URL) ya da SSH üzerinden kendi
  sunucumuzdan indirir. Veritabanının geçerli olduğunu doğrular.
- `reader.py`: `change_events` ve `tile_snapshots` tablolarını salt-okunur okur. Görüntüler BLOB
  olarak döndürülür. Hiçbir yazma işlemi yapmaz; ham veri aynen korunur.

### 2. Model katmani (`src/tammuz/models/`)

- `embeddings.py`: DINOv2 (timm üzerinden) ile bir görüntüyü semantik vektöre çevirir. Herakles'in
  ImageNet ResNet18'inin aksine DINOv2, uydu/hava görüntülerinde aynı araziyi "aynı" tanımakta
  çok daha sağlamdır.
- `landcover.py`: DINOv2 özellikleri üzerine EuroSAT ile eğitilmiş küçük bir linear başlık ile
  10 sınıflı arazi örtüsü sınıflandırması yapar. Değişim türü, iki sınıf arasındaki geniş grup
  geçişinden türetilir (doğal, tarım, yapılı, su).
- `eurosat.py`: EuroSAT verisini Hugging Face Hub'dan parquet olarak indirip görüntülere açar.

### 3. İşlem hattı (`src/tammuz/pipeline/`)

Her adım `pairs.parquet` veya `changes.parquet` üzerinde çalışır ve aynı dosyayı zenginleştirir.
Adımlar sıralıdır; her biri bir öncekinin çıktısına dayanır. Ayrıntı: [pipeline.md](pipeline.md).

### 4. Vektör katmani (`src/tammuz/vector/`)

LanceDB içinde her değişimin "sonra" (değişim sonrası) görüntüsünün DINOv2 embedding'i
indekslenir. Böylece benzer değişimleri aramak, kümelemek ve "bu değişime benzeyenler" diye
sorgulamak mümkün olur. LanceDB yerel ve parquet tabanlı olduğu için sunucu kurulumu gerektirmez.

### 5. Sunucu katmani (`src/tammuz/serve/`)

FastAPI uygulaması, `changes.parquet`'i DuckDB üzerinden filtreler ve harita uygulamasına servis
eder. Statik ön yüz Leaflet ile haritayı çizer. Ayrıntı: [serving.md](serving.md).

## Neden her adım ayrı bir komut?

- Her adım bağımsız çalıştırılabilir ve yeniden başlatılabilir (embed hesaplanmış vektörleri atlar).
- ML adımları (embed, classify) ağır; sunucu (serve) hafiftir. Böylece sunucu kurulumunda ML
  bağımlılıkları yüklenmek zorunda değildir.
- Hangi adımın ne ürettiği net biçimde izlenebilir.
