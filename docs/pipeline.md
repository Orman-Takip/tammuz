# İşlem hattı

İşlem hattı adımları, çalışma sırasına göre aşağıdadır. Her adım `tammuz <ad>` komutuyla veya
`tammuz pipeline` ile çalıştırılır.

## 1. sync

Ham Herakles veritabanını indirir. Üç kaynak desteklenir:

- `bash scripts/fetch-db.sh`: Ham DB'nin GitHub Release'ten parçalı indirilip
  birleştirilmesi (önerilen yol, hem yerelde hem bulutta aynı çalışır).
- `tammuz sync --url <adres>`: Tek parça halinde public indirme adresi.
- `tammuz sync --ssh <host>`: `~/.ssh/config` üzerinden kendi sunucumuzdan
  `scp` ile çeker (özel kullanım).

Ham veritabanı GitHub Releases üzerinden `raw-db-*` etiketiyle, her biri
~2 GB olan parçalar halinde yayınlanır (`herakles.db.part-aa` ... `part-af`).
`scripts/fetch-db.sh` bunları indirir, sırayla birleştirir ve sha256 ile
doğrular.

## 2. pairs

Her `change_events` satırı için önce/sonra görüntülerini `tile_snapshots`'tan okur ve `images/`
dizinine yazar. Görüntüler `(col_row_releaseid)` ile adlandırıldığı için birden fazla olayın
paylaştığı aynı görüntü yalnızca bir kez yazılır. Tüm metadata `pairs.parquet` içine aktarılır.

Çıktı: `pairs.parquet`, `data/processed/images/`

## 3. embed

`pairs.parquet` içindeki tüm benzersiz görüntüler için DINOv2 embedding hesaplar ve her çiftin
önce/sonra kosinüs benzerliğini `dino_similarity` olarak yazar. Önceden hesaplanmış vektörler
`embeddings.parquet` içinde saklanır ve bir sonraki çalıştırmada atlanır.

`dino_similarity` 1.0'a yakınsa iki görüntü semantik olarak aynı araziyi gösteriyordur; bu,
mevsim/bulut/kompresyon gürültüsünün ayırt edici işaretidir.

## 4. filter

`dino_similarity < esik` koşulunu sağlayan ve `largest_run >= min_largest_run` olan çiftleri
"değişim adayı" olarak seçer ve `changes.parquet` içine yazar. Elenen çiftler `pairs.parquet`
içinde kayıtlı kalır; böylece eşik değişiklikleri sonrası yeniden filtreleme mümkündür.

## 5. classify

Her adayın önce ve sonra görüntüsünü arazi örtüsü sınıflandırıcısından geçirir. EuroSAT üzerinde
eğitilen linear başlık ilk çalıştırmada hazırlanır ve `data/models/` içine kaydedilir. İki sınıf
arasındaki geniş grup geçişi `change_type_tr` ve `change_type_en` olarak yazılır.

Güven eşiğinin altında kalan etiketler boş bırakılır (yanlış etiket yaymamak için).

## 6. mask

Önce/sonra gri tonlamalı görüntüler arasındaki mutlak farkı hücreler halinde eşikler ve bağlantılı
bölgeleri bulur. Her bölge için tile'in gerçek lon/lat sınırlarına hizalanmış bir dikdörtgen
poligon üretir. Poligon `geometry_mask` kolonuna, overlay maskesi `masks/` dizinine yazılır.

## 7. narrate (isteğe bağlı)

`config/narrate.enabled: true` olduğunda OpenAI uyumlu bir vizyon modeline önce/sonra görüntüsü
gönderilir ve yapılandırılmış JSON döner:

```json
{
  "is_real": true,
  "change_type": "construction",
  "cause_tr": "Bölgede yeni konut alanları oluşmuş",
  "confidence": 0.9
}
```

DeepSeek, Ollama, vLLM veya LM Studio gibi her OpenAI uyumlu uç nokta kullanılabilir
(`narrate.base_url` ve `narrate.api_key_env`). API anahtarı ortam değişkeninden okunur.

## 8. enrich

- **Episode**: Aynı tile üzerinde zamanla örtüşen değişimler tek bir süreklilik hikayesinde
  birleştirilir. Örneğin yıllar içinde yayılan bir yapılaşma her yıl ayrı kaydedilmişse, tek
  episode olarak etiketlenir.
- **Önem skoru**: Değişim oranı, büyüklük (1 - dino benzerliği), arazi etkisi ve güncellik
  ağırlıklı toplanarak 0..1 arası bir değer üretilir. İnsanlar en önemli değişimleri sıralayabilir.
- **Hotspot**: DBSCAN ile konum bazlı kümeleme yapılır; bölgesel değişim noktaları bulunur.

## 9. vector

LanceDB içinde her değişimin "sonra" görüntüsünün embedding'i indekslenir. Benzer değişim arama
ve kümeleme bu indeks üzerinden yapılır.

## 10. export

`changes.parquet`'i GeoJSON, NDJSON, GeoParquet ve summary.json olarak `data/processed/exports/`
içine yazar. `--release` ile tümü (maskeler dahil) tek bir tar.gz arşivine toplanır.

## Çalıştırma sırası ve eşikler

Tüm eşikler ve model seçenekleri `config/default.yaml` içindedir; yerel değişiklikler için
`config/local.yaml` kullanın (merge edilir, git'e girmez).
