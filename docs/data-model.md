# Veri modeli

Bu belge, işlem hattı boyunca üretilen ana veri dosyalarını ve sütunlarını tanımlar.

## pairs.parquet

Herakles DB'sindeki her değişim olayı için bir satır. `tammuz pairs` tarafından üretilir.

| Sütun | Açıklama |
|---|---|
| event_id | Olay kimliği |
| tile_col, tile_row | z16 tile koordinatları |
| from_release_id, from_date | Önce sürümü ve tarihi |
| to_release_id, to_date | Sonra sürümü ve tarihi |
| detected_at | Tespit zamanı |
| ssim_score | Yapısal benzerlik skoru (1.0 aynı) |
| change_ratio | Pencere bazlı değişim oranı |
| largest_run | En büyük bağlantılı değişim bloğu (pencere) |
| land_windows, water_windows | Kullanılan pencere sayıları |
| ssim_min, ssim_p05 | En düşük ve %5 dilim SSIM |
| embedding_similarity | Herakles'in ImageNet embedding benzerliği |
| from_land_cover, to_land_cover | Herakles'in zayıf EuroSAT etiketleri |
| before_image, after_image | Görüntü dosya yolları (processed altında) |

`embed` adımı `dino_similarity` sütununu ekler.

## changes.parquet

`filter` adımından itibaren üretilen, aday ve zenginleştirilmiş veri seti. pairs'in tüm sütunlarına
ek olarak:

| Sütun | Kaynak adım | Açıklama |
|---|---|---|
| dino_similarity | embed | DINOv2 önce/sonra benzerliği |
| is_candidate, candidate_reason | filter | Adaylık işareti |
| land_cover_from/to | classify | Bizim arazi örtüsü etiketleri |
| land_cover_conf_from/to | classify | Etiket güvenleri |
| change_type_tr/en | classify | Değişim türü (TR/EN) |
| geometry_mask | mask | Değişim bölgesi poligonu (GeoJSON) |
| mask_image | mask | Overlay maskesi yolu |
| ai_is_real, ai_change_type, ai_cause_tr, ai_confidence | narrate | Vizyon LLM değerlendirmesi |
| geometry | enrich | Tile sınır poligonu (GeoJSON) |
| centroid_lon, centroid_lat | enrich | Tile merkezi |
| episode_id, episode_index, episode_count | enrich | Episode bilgisi |
| importance | enrich | 0..1 önem skoru |
| hotspot_id, hotspot_centroid_lon/lat | enrich | Hotspot küme bilgisi |

## Export çıktıları (`data/processed/exports/`)

- `degisimler.geojson`: FeatureCollection; her feature bir poligon + tüm önemli metadata.
- `degisimler.ndjson`: Aynı veri, satır satır.
- `degisimler.geoparquet`: GeoParquet 1.0, WKB geometri, EPSG:4326, zstd sıkıştırma.
- `summary.json`: Toplamlar, tür dağılımı, yıl dağılımı, tarih aralığı.

## LanceDB vektör indeksi

Her değişim için "sonra" görüntüsünün DINOv2 embedding'i `vector` sütununda, metadata
(`tile_col`, `tile_row`, `centroid_lon/lat`, `to_date`, `change_type_tr/en`, `importance`,
`episode_id`, `has_mask`) yanında saklanır.

## Görüntüler

- `data/processed/images/`: Önce/sonra görüntüleri. Adlandırma `col_row_releaseid.ext`.
- `data/processed/masks/`: Her değişimin overlay maskesi, `event_id.png`.
