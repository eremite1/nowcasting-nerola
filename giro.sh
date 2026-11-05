#!/usr/bin/env bash
# Un giro completo: scarica, prevede, verifica, disegna. Scrive tutto in $NUOVO partendo dal giro precedente in $PUB.
#   giro.sh CARTELLA_PUBBLICATA CARTELLA_NUOVA
# Variabili: SFONDO (confini e rilievo), WRF_FILE (previsione WRF, facoltativa), CACHE_SAT (immagini satellite gia' scaricate)
set -euo pipefail
PUB="$1"
NUOVO="$2"
QUI="$(cd "$(dirname "$0")" && pwd)"
DATI="$(mktemp -d)"
rm -rf "$NUOVO" && mkdir -p "$NUOVO"
REG="$NUOVO/registro.txt"
echo "=== giro delle $(date -u +%H:%M) UTC ===" > "$REG"

python "$QUI/scarica.py" --out "$DATI" >> "$REG" 2>&1

WRF=()
if [ -n "${WRF_FILE:-}" ] && [ -f "$WRF_FILE" ]; then WRF=(--wrf "$WRF_FILE"); fi
PAG=(); [ -f "$PUB/pagella.json" ] && PAG=(--pagella "$PUB/pagella.json")
python "$QUI/nowcast.py" --dati "$DATI" "${WRF[@]}" "${PAG[@]}" --precedente "$PUB" --out "$DATI/nowcast.pkl" >> "$REG" 2>&1

python "$QUI/verifica.py" --dati "$DATI" --nowcast "$DATI/nowcast.pkl" --precedente "$PUB" --out "$NUOVO" >> "$REG" 2>&1 \
  || echo "ATTENZIONE: pagella non riuscita" >> "$REG"

python "$QUI/disegna.py" --dati "$DATI" --nowcast "$DATI/nowcast.pkl" --sfondo "${SFONDO:-}" --wrf "${WRF_FILE:-}" --out "$NUOVO" >> "$REG" 2>&1

# il satellite non deve mai bloccare il radar: se non riesce, si tengono le immagini del giro precedente
if ! timeout 480 python "$QUI/satellite.py" --cache "${CACHE_SAT:-$DATI/cache_sat}" --dati "$DATI" --precedente "$PUB" \
     --nowcast "$DATI/nowcast.pkl" --wrf "${WRF_FILE:-}" --out "$NUOVO" >> "$REG" 2>&1; then
  echo "ATTENZIONE: satellite non aggiornato in questo giro" >> "$REG"
  for z in lazio sabina; do
    [ -d "$PUB/$z/sat" ] && cp -a "$PUB/$z/sat" "$NUOVO/$z/" || true
  done
  [ -f "$PUB/satellite.json" ] && cp "$PUB/satellite.json" "$NUOVO/" || true
  [ -f "$PUB/griglia.bin" ] && cp "$PUB/griglia.bin" "$NUOVO/" || true
fi

# gara dei metodi: archivio delle previsioni di tutti i metodi, confronto con il radar vero, classifica (gara.json)
python "$QUI/gara_giro.py" --dati "$DATI" --nowcast "$DATI/nowcast.pkl" --precedente "$PUB" --out "$NUOVO" >> "$REG" 2>&1 || {
  echo "ATTENZIONE: gara dei metodi non riuscita" >> "$REG"
  for f in gara.json gara_storico.json; do [ -f "$PUB/$f" ] && cp "$PUB/$f" "$NUOVO/" || true; done
}

# radar 3D: la pioggia a 10 quote (CAPPI) sopra il rilievo; se non riesce resta quello del giro prima
python "$QUI/radar3d.py" --sfondo "${SFONDO:-}" --precedente "$PUB" --out "$NUOVO" >> "$REG" 2>&1 || {
  echo "ATTENZIONE: radar 3D non aggiornato" >> "$REG"
  [ -d "$PUB/radar3d" ] && cp -r "$PUB/radar3d" "$NUOVO/" || true
}

# radar come pluviometri virtuali (e WRF negli stessi punti) per le pagelle dei modelli contro il radar
python "$QUI/radar_punti.py" --precedente "$PUB" --out "$NUOVO" --wrf "${WRF_FILE:-}" >> "$REG" 2>&1 \
  || echo "ATTENZIONE: radar nei punti non aggiornato" >> "$REG"

python "$QUI/stazione.py" --out "$NUOVO" >> "$REG" 2>&1 || true

python "$QUI/notifiche.py" --nuovo "$NUOVO" --precedente "$PUB" >> "$REG" 2>&1 || echo "ATTENZIONE: notifiche non riuscite" >> "$REG"

cp "$QUI/pagina/index.html" "$NUOVO/index.html"
cp "$QUI/pagina/radar3d.js" "$NUOVO/radar3d.js"
cp "$DATI/diagnosi.json" "$NUOVO/" 2>/dev/null || true
touch "$NUOVO/.nojekyll"
cat > "$NUOVO/LEGGIMI.md" <<'EOT'
# Nowcasting Nerola (risultati)

Pagina: `index.html`. Radar e previsione: `indice.json`. Satellite: `satellite.json`. Pagella: `pagella.json`, `storico.csv`.
Immagine pronta dell'ultimo radar: `lazio/ultimo.png`, `sabina/ultimo.png`.

Radar: Dipartimento della Protezione Civile, piattaforma Radar-DPC (https://radar.protezionecivile.it), dati elaborati
(ritagliati, ricampionati, estrapolati e combinati con il modello).
Satellite: contains modified EUMETSAT Meteosat (MTG) data, servizio EUMETView (CC BY 4.0).
Modello: WRF su dati ECMWF IFS open data (CC BY 4.0), dati modificati.
Rilievo: Terrain Tiles (SRTM/GMTED). Confini: ISTAT (CC BY 3.0 IT), Natural Earth. Nomi: GeoNames (CC BY 4.0).
Previsione sperimentale: non usare per decisioni di sicurezza.
EOT
rm -rf "$DATI"
echo "=== giro finito alle $(date -u +%H:%M:%S) UTC ===" >> "$REG"
