# Harita uygulaması (serve)

`tammuz serve` komutu, `changes.parquet` + görüntüler + LanceDB indeksi üzerinden bir FastAPI
uygulaması ve Leaflet tabanlı ön yüz başlatır. Varsayılan adres `http://127.0.0.1:8787`.

## Sorgu mimarisi

Harita ön yüzü, görünüm penceresini z16 tile aralığına çevirir ve arka uca filtreli sorgular
atar. Arka uç, `changes.parquet`'i DuckDB üzerinde filtreler. Bu yaklaşım iki nedenden dolayı
seçildi:

- Değişimler tile ızgarasına bağlıdır; görünüm penceresi doğrudan bir tile aralığına karşılık
  gelir. Geometri hesaplamasına gerek kalmadan hızlı filtreleme yapılır.
- Yüz binlerce kayıt olsa bile DuckDB, dar aralıklı sorguları milisaniyeler içinde döndürür.

Düşük zoom seviyelerinde kayıt sayısı çok büyük olduğundan ön yüz bir `grid` (hücre sayımları)
katmanına geçer; yüksek zoomda gerçek değişim poligonlarını gösterir. Eşik `serve.max_features`.

## API uçları

| Uç | Açıklama |
|---|---|
| GET / | Ön yüz (statik dosyalar) |
| GET /api/overview | Toplam, tür dağılımı, yıl dağılımı, tarih aralığı |
| GET /api/count | Görünüm penceresindeki kayıt sayısı |
| GET /api/changes | Filtreli GeoJSON feature'ları |
| GET /api/grid | Hücre bazlı sayım katmanı |
| GET /api/event/{event_id} | Tek değişim detayı |
| GET /api/tile/{col}/{row} | Bir tile'daki tüm değişimler |
| GET /api/image?event_id&side | Önce/sonra görüntüsü |
| GET /api/mask?event_id | Değişim bölgesi maskesi |
| GET /api/similar?event_id&k | Benzer değişimler (LanceDB) |

## Ön yüz özellikleri

- Tür filtresi (pill'ler), tarih aralığı filtresi
- Zaman oynatma: yıl kaydırıcısı ile değişimlerin yıllar içindeki oluşumunu izleme
- Yalnızca AI onaylı değişimler anahtarı (narrate çalıştırıldıysa)
- Değişim detay paneli: önce/sonra görüntü, maske, episode bilgisi, benzer değişimler
- Tür bazlı renk lejantı

## Bağımlılıklar

Sunucu katmanı ML bağımlılığı gerektirmez (torch/timm yalnızca `embed` ve `classify` için
gerekir). Bu yüzden sunucuyu `pip install -e ".[dev]"` ile de çalıştırabilirsiniz; yalnızca
işlenmiş `changes.parquet` + görüntüler + `lancedb/` dizini gereklidir.
