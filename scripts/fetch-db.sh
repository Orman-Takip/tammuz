#!/usr/bin/env bash
# Tammuz ham veritabanini GitHub Release'ten indirip birlestirir.
#
# Kullanim:
#   bash scripts/fetch-db.sh [release-tag] [hedef-yol]
#
# Varsayilan: release-tag = raw-db-20260902, hedef = data/raw/herakles.db
# Veritabani 2 GB'lik parcalar halinde yayinlanir; bu script parcalari
# indirir, birlestirir ve sha256 ile dogrular.

set -euo pipefail

REPO="Orman-Takip/tammuz"
TAG="${1:-raw-db-20260902}"
TARGET="${2:-data/raw/herakles.db}"

BASE="https://github.com/${REPO}/releases/download/${TAG}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "> Indiriliyor: ${TAG}"
for part in aa ab ac ad ae af; do
  url="${BASE}/herakles.db.part-${part}"
  echo "  ${url}"
  curl -fL --retry 5 -C - -o "${TMP}/herakles.db.part-${part}" "${url}"
done

echo "> Birlestiriliyor -> ${TARGET}"
mkdir -p "$(dirname "${TARGET}")"
cat "${TMP}"/herakles.db.part-* > "${TMP}/herakles.db"

echo "> sha256 dogrulaniyor"
curl -fL --retry 5 -o "${TMP}/herakles.db.sha256" "${BASE}/herakles.db.sha256"
expected="$(awk 'NR==1 { print $1 }' "${TMP}/herakles.db.sha256")"
actual="$(shasum -a 256 "${TMP}/herakles.db" | awk '{ print $1 }')"
if [[ "${expected}" != "${actual}" ]]; then
  echo "HATA: sha256 uyusmadi (beklenen ${expected}, gelen ${actual})" >&2
  exit 1
fi

mv "${TMP}/herakles.db" "${TARGET}"
echo "> Hazir: ${TARGET} (sha256 dogrulandi)"
