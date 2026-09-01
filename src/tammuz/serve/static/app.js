/* Tammuz harita uygulamasi. */
(function () {
  "use strict";

  var ZOOM16 = 16;
  var state = {
    types: new Set(),
    from: "",
    to: "",
    maxYear: 2026,
    onlyAI: false,
    layer: null,
  };

  var map = L.map("map", { zoomControl: true }).setView([38.9, 34.0], 7);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; CARTO',
    subdomains: "abcd",
    maxZoom: 19,
  }).addTo(map);

  var COLOR = {
    "Tarıma açılma": "#7cb342",
    "Doğal alanın yapılaşması": "#fb8c00",
    "Su baskını": "#42a5f5",
    "Tarım arazisinin yapılaşması": "#e53935",
    "Doğallaşma": "#00897b",
    "Su çekilmesi": "#4fc3f7",
    "Yeşillendirme": "#aed581",
    "Tarıma dönüşüm": "#9ccc65",
    "Dolgu/yapılaşma": "#8e24aa",
  };
  var UNKNOWN_COLOR = "#9e9e9e";

  function typeColor(t) {
    return COLOR[t] || UNKNOWN_COLOR;
  }

  function colFromLon(lon) {
    return Math.floor(((lon + 180) / 360) * Math.pow(2, ZOOM16));
  }
  function lonFromCol(c) {
    return (c / Math.pow(2, ZOOM16)) * 360 - 180;
  }
  function rowFromLat(lat) {
    var rad = (lat * Math.PI) / 180;
    var m = Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI;
    return Math.floor(((1 - m) / 2) * Math.pow(2, ZOOM16));
  }
  function latFromRow(r) {
    var y = 1 - (2 * r) / Math.pow(2, ZOOM16);
    return (Math.atan(Math.sinh(Math.PI * y)) * 180) / Math.PI;
  }

  function viewportRange() {
    var b = map.getBounds();
    return {
      min_col: colFromLon(b.getWest()),
      max_col: colFromLon(b.getEast()),
      min_row: rowFromLat(b.getNorth()),
      max_row: rowFromLat(b.getSouth()),
    };
  }

  function cellForZoom(z) {
    return Math.max(1, Math.pow(2, Math.max(0, ZOOM16 - z)));
  }

  function filterParams() {
    var p = viewportRange();
    p.types = state.types.size ? Array.from(state.types).join(",") : "";
    p.from_date = state.from || "";
    p.to_date = state.to || "";
    return p;
  }

  function qs(params) {
    var parts = [];
    Object.keys(params).forEach(function (k) {
      if (params[k]) parts.push(encodeURIComponent(k) + "=" + encodeURIComponent(params[k]));
    });
    return parts.join("&");
  }

  var gridLayer = L.layerGroup();
  var featureLayer = L.layerGroup();
  gridLayer.addTo(map);
  featureLayer.addTo(map);

  function refresh() {
    setStatus("yukleniyor...");
    var cell = cellForZoom(map.getZoom());
    var params = filterParams();

    fetch("/api/count?" + qs(params))
      .then(function (r) { return r.json(); })
      .then(function (json) {
        var count = json.count;
        var maxFeatures = 4000;
        gridLayer.clearLayers();
        featureLayer.clearLayers();
        if (cell > 1 || count > maxFeatures) {
          fetch("/api/grid?" + qs(Object.assign({ cell: cell }, params)))
            .then(function (r) { return r.json(); })
            .then(function (fc) { drawGrid(fc); setStatus(count + " degisim (ozet harita)"); });
        } else {
          fetch("/api/changes?" + qs(params))
            .then(function (r) { return r.json(); })
            .then(function (fc) { drawFeatures(fc); setStatus(count + " degisim gosteriliyor"); });
        }
      })
      .catch(function (e) { setStatus("hata: " + e.message); });
  }

  function drawGrid(fc) {
    var maxN = 1;
    fc.features.forEach(function (f) { maxN = Math.max(maxN, f.properties.count); });
    fc.features.forEach(function (f) {
      var n = f.properties.count;
      var ratio = n / maxN;
      var color = ratio > 0.5 ? "#d32f2f" : ratio > 0.2 ? "#fb8c00" : "#ffd54f";
      L.polygon(f.geometry.coordinates[0], {
        color: color,
        weight: 1,
        fillColor: color,
        fillOpacity: 0.55,
      }).addTo(gridLayer).bindPopup("<b>" + n + "</b> degisim");
    });
  }

  function drawFeatures(fc) {
    L.geoJSON(fc, {
      pointToLayer: function (feature, latlng) {
        return L.circleMarker(latlng, {
          radius: 5,
          color: "#fff",
          weight: 1,
          fillColor: typeColor(feature.properties.change_type_tr),
          fillOpacity: 0.9,
        });
      },
      onEachFeature: function (feature, layer) {
        layer.bindPopup(popupHtml(feature));
        layer.on("click", function () { showDetail(feature); });
      },
    }).addTo(featureLayer);
  }

  function popupHtml(f) {
    var p = f.properties;
    var t = p.change_type_tr || "Siniflanmamis";
    return (
      "<b>" + t + "</b><br>" +
      p.from_date + " &rarr; " + p.to_date + "<br>" +
      "<span class='muted'>tile " + p.tile_col + ", " + p.tile_row +
      " &middot; onem " + (p.importance !== null && p.importance !== undefined ? p.importance.toFixed(2) : "-") + "</span>"
    );
  }

  function showDetail(f) {
    var p = f.properties;
    var el = document.getElementById("detail");
    var body = document.getElementById("detail-body");
    var maskHtml = p.mask_image
      ? "<figure><img src='/api/mask?event_id=" + p.event_id + "' alt='degisim bolgesi'><figcaption>Degisim bolgesi</figcaption></figure>"
      : "";
    var aiHtml = "";
    if (p.ai_is_real !== null && p.ai_is_real !== undefined) {
      aiHtml = "<p><b>AI degerlendirmesi:</b> " + (p.ai_is_real ? "gercek degisim" : "gurultu") +
        (p.ai_cause_tr ? "<br>" + p.ai_cause_tr : "") + "</p>";
    }
    var episodeHtml = p.episode_count > 1
      ? "<p class='muted'>Bu tile'daki " + p.episode_count + " degisimden " + (p.episode_index + 1) + ". (episode " + p.episode_id + ")</p>"
      : "";

    body.innerHTML =
      "<div class='imgs'>" +
      "<figure><img src='/api/image?event_id=" + p.event_id + "&side=before' alt='oncesi'><figcaption>Oncesi (" + p.from_date + ")</figcaption></figure>" +
      "<figure><img src='/api/image?event_id=" + p.event_id + "&side=after' alt='sonrasi'><figcaption>Sonrasi (" + p.to_date + ")</figcaption></figure>" +
      "</div>" +
      maskHtml + aiHtml + episodeHtml +
      "<table class='meta'>" +
      "<tr><td>Tur</td><td>" + (p.change_type_tr || "-") + "</td></tr>" +
      "<tr><td>Arazi</td><td>" + (p.land_cover_from || "-") + " &rarr; " + (p.land_cover_to || "-") + "</td></tr>" +
      "<tr><td>Konum</td><td>z16 " + p.tile_col + ", " + p.tile_row + "</td></tr>" +
      "<tr><td>DINO benzerlik</td><td>" + (p.dino_similarity !== null && p.dino_similarity !== undefined ? p.dino_similarity.toFixed(3) : "-") + "</td></tr>" +
      "<tr><td>Degisim orani</td><td>" + (p.change_ratio !== null ? (p.change_ratio * 100).toFixed(1) + "%" : "-") + "</td></tr>" +
      "<tr><td>Onem</td><td>" + (p.importance !== null && p.importance !== undefined ? p.importance.toFixed(2) : "-") + "</td></tr>" +
      "</table>" +
      "<div class='similar'><h4>Benzer degisimler</h4><div id='similar-list'></div></div>";

    el.classList.remove("hidden");
    loadSimilar(p.event_id);
  }

  function loadSimilar(eventId) {
    fetch("/api/similar?event_id=" + eventId + "&k=6")
      .then(function (r) { return r.json(); })
      .then(function (items) {
        var box = document.getElementById("similar-list");
        if (!items.length) { box.innerHTML = "<p class='muted'>Benzer bulunamadi.</p>"; return; }
        box.innerHTML = items.map(function (f) {
          var p = f.properties;
          return "<button class='similar-item' data-id='" + p.event_id + "'>" +
            (p.change_type_tr || "-") + " &middot; " + p.to_date + "</button>";
        }).join("");
        box.querySelectorAll(".similar-item").forEach(function (btn) {
          btn.addEventListener("click", function () {
            var id = Number(btn.dataset.id);
            fetch("/api/event/" + id).then(function (r) { return r.json(); }).then(showDetail);
          });
        });
      });
  }

  function loadOverview() {
    fetch("/api/overview")
      .then(function (r) { return r.json(); })
      .then(function (o) {
        document.getElementById("total-label").textContent =
          o.total.toLocaleString("tr-TR") + " degisim, " + o.tiles.toLocaleString("tr-TR") + " tile";
        var pills = document.getElementById("type-pills");
        pills.innerHTML = "";
        o.change_types.forEach(function (t) {
          var b = document.createElement("button");
          b.className = "pill";
          b.textContent = t.type + " (" + t.count + ")";
          b.style.setProperty("--c", typeColor(t.type));
          b.addEventListener("click", function () {
            if (state.types.has(t.type)) { state.types.delete(t.type); b.classList.remove("on"); }
            else { state.types.add(t.type); b.classList.add("on"); }
            refresh();
          });
          pills.appendChild(b);
        });
        if (o.date_range.from) {
          document.getElementById("date-from").max = o.date_range.to;
          document.getElementById("date-to").max = o.date_range.to;
          document.getElementById("date-from").min = o.date_range.from;
          document.getElementById("date-to").min = o.date_range.from;
        }
        buildLegend(o.change_types);
      });
  }

  function buildLegend(types) {
    var el = document.getElementById("legend");
    el.innerHTML = "<b>Degisim turleri</b>" + types.map(function (t) {
      return "<div><span class='dot' style='background:" + typeColor(t.type) + "'></span>" + t.type + "</div>";
    }).join("");
    el.classList.remove("hidden");
  }

  function setStatus(msg) {
    document.getElementById("status").textContent = msg;
  }

  document.getElementById("detail-close").addEventListener("click", function () {
    document.getElementById("detail").classList.add("hidden");
  });
  document.getElementById("reset-filter").addEventListener("click", function () {
    state.types.clear();
    state.from = "";
    state.to = "";
    state.maxYear = 2026;
    document.getElementById("year-play").value = 2026;
    document.getElementById("year-label").textContent = 2026;
    document.querySelectorAll(".pill").forEach(function (b) { b.classList.remove("on"); });
    document.getElementById("date-from").value = "";
    document.getElementById("date-to").value = "";
    refresh();
  });
  document.getElementById("date-from").addEventListener("change", function (e) {
    state.from = e.target.value; refresh();
  });
  document.getElementById("date-to").addEventListener("change", function (e) {
    state.to = e.target.value; refresh();
  });
  document.getElementById("year-play").addEventListener("input", function (e) {
    var y = Number(e.target.value);
    state.maxYear = y;
    document.getElementById("year-label").textContent = y;
    state.to = y + "-12-31";
    refresh();
  });
  document.getElementById("only-ai").addEventListener("change", function (e) {
    state.onlyAI = e.target.checked;
    refresh();
  });

  var playing = null;
  document.getElementById("play-btn").addEventListener("click", function () {
    if (playing) { clearInterval(playing); playing = null; return; }
    var slider = document.getElementById("year-play");
    playing = setInterval(function () {
      var v = Number(slider.value);
      if (v >= Number(slider.max)) { clearInterval(playing); playing = null; return; }
      slider.value = v + 1;
      slider.dispatchEvent(new Event("input"));
    }, 700);
  });

  map.on("moveend", refresh);
  map.on("zoomend", refresh);

  loadOverview();
  refresh();
})();
