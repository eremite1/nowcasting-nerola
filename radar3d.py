#!/usr/bin/env python3
"""RADAR 3D: la pioggia vista dal radar a 10 quote diverse, per vedere i temporali in volume sopra il rilievo.

La Protezione Civile pubblica, oltre al radar "al suolo", le immagini a quota costante (CAPPI_1 ... CAPPI_10, una per
ogni chilometro di quota, da 1 a 10 km). Messe una sopra l'altra formano un volume: dove l'eco arriva in alto c'e' un
temporale forte (i nuclei di grandine stanno tra 5 e 9 km), dove resta in basso piove in modo debole e uniforme.

Uscite (nella cartella pubblicata, sottocartella radar3d/):
  info.json     ora, quote, dimensioni, scala
  volume.bin    10 strati x righe x colonne, 1 byte per cella: 0 = niente, altrimenti dBZ = (v - 1) / 2 - 10
  terreno.bin   quota del suolo in metri, 2 byte per cella (intero senza segno, little-endian)
La griglia e' quella della mappa del Lazio della pagina, rimpicciolita (una cella = PASSO pixel, circa 1,4 km), cosi'
sotto il volume si puo' stendere la stessa immagine mappa.jpg.

  radar3d.py --sfondo CARTELLA_SFONDO --out CARTELLA_NUOVA
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

import disegna
import fonti
import geo
import nubi
import sfondo as sfondo_mod

LIVELLI = [f"CAPPI_{k}" for k in range(1, 11)]
# dettaglio per lo zoom: il Lazio diviso in TESSERE x TESSERE riquadri, ognuno con mappa (860 pixel, circa 70 m) e
# rilievo (circa 280 m) piu' fini, calcolati una volta sola e poi riusati (cambiare VERSIONE_DETTAGLIO per rifarli)
TESSERE = 4
VERSIONE_DETTAGLIO = 1
PASSO_DETTAGLIO = 4
QUOTE_M = [1000 * k for k in range(1, 11)]
PASSO = 5                 # pixel della mappa per cella del volume
SOGLIA_DBZ = 5.0


def riduci(a, f, modo):
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    b = a[:h, :w].reshape(h // f, f, w // f, f)
    with np.errstate(all="ignore"):
        return np.nanmax(b, axis=(1, 3)) if modo == "max" else np.nanmean(b, axis=(1, 3))


def dettaglio(cartella_sfondo, out, precedente):
    """Tessere ad alta definizione per lo zoom. Se ci sono gia' (stessa versione) nel giro precedente si copiano."""
    import shutil
    dd = out / "dettaglio"
    vecchio = Path(precedente) / "radar3d" / "dettaglio" if precedente else None
    if vecchio and (vecchio / "info.json").exists():
        try:
            if json.loads((vecchio / "info.json").read_text()).get("versione") == VERSIONE_DETTAGLIO:
                if dd.exists():
                    shutil.rmtree(dd)
                shutil.copytree(vecchio, dd)
                return json.loads((dd / "info.json").read_text())
        except Exception:  # noqa: BLE001
            pass
    if not cartella_sfondo:
        return None
    dd.mkdir(parents=True, exist_ok=True)
    sf = sfondo_mod.carica(cartella_sfondo)
    z = disegna.Zona("lazio", disegna.AREE["lazio"])
    dl, da = (z.lon1 - z.lon0) / TESSERE, (z.lat1 - z.lat0) / TESSERE
    tess = []
    for i in range(TESSERE):            # riga 0 = nord
        for j in range(TESSERE):
            riq = (z.lon0 + j * dl, z.lon0 + (j + 1) * dl, z.lat1 - (i + 1) * da, z.lat1 - i * da)
            zt = disegna.Zona(f"t{i}{j}", {"nome": "", "riquadro": riq, "larghezza": 860, "dettaglio": True})
            margine = 0.02
            zq, meta = sfondo_mod.dem((riq[0] - margine, riq[1] + margine, riq[2] - margine, riq[3] + margine), 11, 0.0008)
            dem_t = {"fine": (zq, meta[0], meta[1], meta[2])}
            disegna.fondo(zt, dem_t, dd / f"t{i}_{j}.jpg")
            disegna.confini(zt, sf, dd / "_confini.png", massimo=12)
            im = Image.open(dd / f"t{i}_{j}.jpg").convert("RGBA")
            im.alpha_composite(Image.open(dd / "_confini.png").convert("RGBA"))
            im.convert("RGB").save(dd / f"t{i}_{j}.jpg", quality=86, optimize=True)
            q = nubi.dem_su_zona(dem_t, zt)
            t = riduci(q, PASSO_DETTAGLIO, "media")
            np.clip(np.nan_to_num(t), 0, 65000).astype("<u2").tofile(dd / f"t{i}_{j}.bin")
            tess.append({"i": i, "j": j, "righe": int(t.shape[0]), "colonne": int(t.shape[1])})
    (dd / "_confini.png").unlink(missing_ok=True)
    info = {"versione": VERSIONE_DETTAGLIO, "tessere": TESSERE, "elenco": tess}
    (dd / "info.json").write_text(json.dumps(info))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sfondo", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--precedente", default="")
    a = ap.parse_args()
    z = disegna.Zona("lazio", disegna.AREE["lazio"])
    out = Path(a.out) / "radar3d"
    out.mkdir(parents=True, exist_ok=True)

    strati, ore, diag = [], [], {}
    for tipo in LIVELLI:
        try:
            t0, _ = fonti.ultimo(tipo)
            arr, g, _ = geo.leggi_geotiff(fonti.scarica(tipo, t0))
            arr = np.where(arr < -30, np.nan, arr).astype("float32")      # valori negativi grandi = nessun dato
            v = disegna.ricampiona(arr, z.mappa_da(g))
            strati.append(riduci(v, PASSO, "max"))
            ore.append(t0)
            diag[tipo] = [round(float(np.nanmin(arr)), 1), round(float(np.nanmax(arr)), 1)]
        except Exception as e:  # noqa: BLE001
            print(f"ATTENZIONE {tipo}: {e}", file=sys.stderr)
            strati.append(None)
            diag[tipo] = str(e)[:120]
    buoni = [s for s in strati if s is not None]
    if len(buoni) < 5:
        print("radar 3D: troppe quote mancanti, non aggiorno", diag, file=sys.stderr)
        sys.exit(1)
    forma = buoni[0].shape
    vol = np.zeros((len(LIVELLI),) + forma, dtype=np.uint8)
    for k, s in enumerate(strati):
        if s is None:
            continue
        q = np.where(np.isfinite(s) & (s >= SOGLIA_DBZ), np.clip(np.round((s + 10) * 2) + 1, 1, 255), 0)
        vol[k] = q.astype(np.uint8)
    vol.tofile(out / "volume.bin")

    dem = sfondo_mod.carica_dem(a.sfondo) if a.sfondo else {}
    q = nubi.dem_su_zona(dem, z) if dem else None
    terr = np.zeros(forma) if q is None else riduci(q, PASSO, "media")[:forma[0], :forma[1]]
    np.clip(np.nan_to_num(terr), 0, 65000).astype("<u2").tofile(out / "terreno.bin")

    try:
        det = dettaglio(a.sfondo, out, a.precedente)
    except Exception as e:  # noqa: BLE001
        print("ATTENZIONE: dettaglio per lo zoom non pronto:", e, file=sys.stderr)
        det = None
    latc = (z.lat0 + z.lat1) / 2
    larg_km = (z.lon1 - z.lon0) * 111.32 * np.cos(np.radians(latc))
    alt_km = (z.lat1 - z.lat0) * 110.57
    ora = max(ore)
    info = {"ora": datetime.fromtimestamp(ora / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "ora_ms": int(ora), "quote_m": QUOTE_M,
            "righe": int(forma[0]), "colonne": int(forma[1]), "larghezza_km": round(float(larg_km), 2),
            "altezza_km": round(float(alt_km), 2), "riquadro": [z.lon0, z.lon1, z.lat0, z.lat1],
            "mappa": "lazio/mappa.jpg", "codifica": "dBZ = (v - 1) / 2 - 10", "diagnosi": diag,
            "celle_con_eco": int((vol > 0).sum()), "terreno": q is not None, "confini": "lazio/confini.png",
            "dettaglio": det}
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False))
    print(f"radar 3D: {len(buoni)} quote su 10, {forma[1]}x{forma[0]} celle, {info['celle_con_eco']} con eco, "
          f"valori {diag}")


if __name__ == "__main__":
    main()
