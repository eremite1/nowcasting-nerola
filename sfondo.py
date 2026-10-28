#!/usr/bin/env python3
"""Confini, costa, laghi, fiumi, nomi dei paesi e rilievo da disegnare sotto le mappe.

  sfondo.py --scarica CARTELLA     scarica i dati (una volta) e li salva in CARTELLA
  (da mappe.py)  carica(CARTELLA), carica_dem(CARTELLA)

Fonti (gratuite):
  - Natural Earth (pubblico dominio): province, costa, laghi, fiumi
  - openpolis/geojson-italy, da dati ISTAT (CC BY 3.0 IT): confini di comuni e regioni
  - GeoNames (CC BY 4.0): nomi, abitanti e quota dei paesi
  - Terrain Tiles su AWS (Mapzen; SRTM, GMTED e altri, vedi https://registry.opendata.aws/terrain-tiles/): rilievo
Se un download non riesce le mappe escono lo stesso, con meno dettagli.
"""
import argparse
import io
import json
import math
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# riquadro tenuto nel file piccolo (piu' largo dei domini): lon_min, lon_max, lat_min, lat_max
RIQUADRO = (10.0, 15.5, 40.5, 43.8)
NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
STRATI_NE = {
    "province": "ne_10m_admin_1_states_provinces.geojson",
    "costa": "ne_10m_coastline.geojson",
    "laghi": "ne_10m_lakes.geojson",
    "fiumi": "ne_10m_rivers_lake_centerlines.geojson",
}
OPENPOLIS = "https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/"
# regioni attorno ai domini: 12 Lazio, 13 Abruzzo, 10 Umbria, 9 Toscana, 11 Marche
REGIONI = (12, 13, 10, 9, 11)
GEONAMES = "https://download.geonames.org/export/dump/IT.zip"
TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
# rilievo: zona fine (dominio a 1 km con margine) e zona larga (dominio a 3 km con margine)
DEM_FINE = {"riquadro": (12.2, 13.35, 41.7, 42.6), "zoom": 10, "passo": 0.0015}
DEM_LARGO = {"riquadro": (10.9, 14.6, 40.9, 43.4), "zoom": 9, "passo": 0.004}
# sedi di comune (PPLA3), capoluoghi di provincia (PPLA2) e di regione (PPLA)
CODICI_SEDE = {"PPLA", "PPLA2", "PPLA3", "PPLC"}


def scarica(url, tentativi=4):
    ultimo = None
    for n in range(tentativi):
        try:
            rich = urllib.request.Request(url, headers={"User-Agent": "wrf-nerola"})
            with urllib.request.urlopen(rich, timeout=180) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            ultimo = e
            time.sleep(4 * (n + 1))
    raise RuntimeError(f"download fallito: {url}: {ultimo}")


def _dentro(x, y):
    return RIQUADRO[0] <= x <= RIQUADRO[1] and RIQUADRO[2] <= y <= RIQUADRO[3]


def _linee(geom, solo_bordo=True):
    """Ritorna una lista di linee [[(lon, lat), ...], ...] da una geometria GeoJSON (linee o bordi di poligoni)."""
    t, c = geom.get("type"), geom.get("coordinates")
    if t == "LineString":
        return [c]
    if t == "MultiLineString":
        return list(c)
    if t == "Polygon":
        return list(c) if solo_bordo else [c[0]]
    if t == "MultiPolygon":
        return [anello for poligono in c for anello in (poligono if solo_bordo else poligono[:1])]
    return []


def _taglia(linea, decimali=4):
    """Tiene solo la parte di linea dentro il riquadro, arrotondando i decimali per fare file piccoli."""
    out, corrente = [], []
    for x, y, *_ in linea:
        if RIQUADRO[0] - 0.3 <= x <= RIQUADRO[1] + 0.3 and RIQUADRO[2] - 0.3 <= y <= RIQUADRO[3] + 0.3:
            corrente.append((round(x, decimali), round(y, decimali)))
        elif len(corrente) > 1:
            out.append(corrente)
            corrente = []
        else:
            corrente = []
    if len(corrente) > 1:
        out.append(corrente)
    return out


def strato_ne(nome, file):
    dati = json.loads(scarica(NE + file))
    risultato = []
    for f in dati["features"]:
        g = f.get("geometry")
        p = f.get("properties") or {}
        if not g:
            continue
        if nome == "province" and (p.get("iso_a2") or p.get("ISO_A2")) != "IT":
            continue  # interessano le province italiane
        for linea in _linee(g, solo_bordo=True):
            risultato.extend(_taglia(linea))
    return risultato


def strato_openpolis(tipo):
    """Confini di comuni ('municipalities') o regioni ('regions') dalle regioni attorno ai domini."""
    risultato, errori = [], []
    if tipo == "regions":
        elenco = [f"limits_IT_regions.geojson"]
    else:
        elenco = [f"limits_R_{r}_municipalities.geojson" for r in REGIONI]
    for file in elenco:
        try:
            dati = json.loads(scarica(OPENPOLIS + file))
            for f in dati["features"]:
                g = f.get("geometry")
                if g:
                    for linea in _linee(g, solo_bordo=True):
                        risultato.extend(_taglia(linea, 4))
        except Exception as e:  # noqa: BLE001
            errori.append(f"{file}: {e}")
    if not risultato:
        raise RuntimeError("; ".join(errori) or "nessun confine")
    return risultato


def paesi():
    z = zipfile.ZipFile(io.BytesIO(scarica(GEONAMES)))
    out = []
    with z.open("IT.txt") as f:
        for riga in io.TextIOWrapper(f, encoding="utf-8"):
            c = riga.rstrip("\n").split("\t")
            if len(c) < 17 or c[6] != "P":
                continue
            lat, lon = float(c[4]), float(c[5])
            if not _dentro(lon, lat):
                continue
            pop = int(c[14] or 0)
            sede = c[7] in CODICI_SEDE
            if not sede and pop < 800:
                continue
            quota = None
            for k in (16, 15):  # quota da modello digitale del terreno (SRTM), poi quota dichiarata
                if c[k].lstrip("-").isdigit():
                    quota = int(c[k])
                    break
            nome = "Roma" if c[1] == "Rome" else c[1]
            out.append({"nome": nome, "lat": lat, "lon": lon, "pop": pop, "sede": sede, "quota": quota})
    out.sort(key=lambda p: (not p["sede"], -p["pop"]))
    return out


# ---------------------------------------------------------------- rilievo da tessere "terrarium"

def _px(lon, z):
    return (np.asarray(lon, dtype="float64") + 180.0) / 360.0 * 2 ** z


def _py(lat, z):
    r = np.radians(np.asarray(lat, dtype="float64"))
    return (1.0 - np.log(np.tan(r) + 1.0 / np.cos(r)) / math.pi) / 2.0 * 2 ** z


def _tessera(z, x, y):
    from PIL import Image
    png = scarica(TERRARIUM.format(z=z, x=x, y=y))
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype="float64")
    return a[:, :, 0] * 256.0 + a[:, :, 1] + a[:, :, 2] / 256.0 - 32768.0


def dem(riquadro, zoom, passo):
    """Quote (m) su una griglia regolare lon/lat (latitudine crescente) ottenuta dalle tessere del rilievo."""
    lon0, lon1, lat0, lat1 = riquadro
    x0, x1 = int(math.floor(_px(lon0, zoom))), int(math.floor(_px(lon1, zoom)))
    y0, y1 = int(math.floor(_py(lat1, zoom))), int(math.floor(_py(lat0, zoom)))
    lavori = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    with ThreadPoolExecutor(6) as pool:
        tessere = list(pool.map(lambda xy: _tessera(zoom, xy[0], xy[1]), lavori))
    mosaico = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256))
    for (x, y), t in zip(lavori, tessere):
        mosaico[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = t
    lons = np.arange(lon0, lon1 + passo / 2, passo)
    lats = np.arange(lat0, lat1 + passo / 2, passo)
    fx = (_px(lons, zoom) - x0) * 256.0 - 0.5
    fy = (_py(lats, zoom) - y0) * 256.0 - 0.5
    ix = np.clip(np.floor(fx).astype(int), 0, mosaico.shape[1] - 2)
    iy = np.clip(np.floor(fy).astype(int), 0, mosaico.shape[0] - 2)
    wx = np.clip(fx - ix, 0, 1)[None, :]
    wy = np.clip(fy - iy, 0, 1)[:, None]
    a = mosaico[iy][:, ix]
    b = mosaico[iy][:, ix + 1]
    c = mosaico[iy + 1][:, ix]
    d = mosaico[iy + 1][:, ix + 1]
    z = (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy
    return np.round(z).astype("int16"), (float(lon0), float(lat0), float(passo))


def costruisci(cartella):
    cartella = Path(cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    risultato, errori, avvisi = {}, [], []
    for nome, file in STRATI_NE.items():
        try:
            risultato[nome] = strato_ne(nome, file)
            print(f"{nome}: {len(risultato[nome])} linee")
        except Exception as e:  # noqa: BLE001
            errori.append(f"{nome}: {e}")
            risultato[nome] = []
    try:
        risultato["paesi"] = paesi()
        print("paesi:", len(risultato["paesi"]))
    except Exception as e:  # noqa: BLE001
        errori.append(f"paesi: {e}")
        risultato["paesi"] = []
    # parti facoltative: se mancano si va avanti
    for chiave, tipo in (("comuni", "municipalities"), ("regioni", "regions")):
        try:
            risultato[chiave] = strato_openpolis(tipo)
            print(f"{chiave}: {len(risultato[chiave])} linee")
        except Exception as e:  # noqa: BLE001
            avvisi.append(f"{chiave}: {e}")
            risultato[chiave] = []
    arrays = {}
    for chiave, cfg in (("fine", DEM_FINE), ("largo", DEM_LARGO)):
        try:
            z, meta = dem(cfg["riquadro"], cfg["zoom"], cfg["passo"])
            arrays[chiave] = z
            arrays[chiave + "_meta"] = np.array(meta)
            print(f"rilievo {chiave}: {z.shape}, quota {z.min()}..{z.max()} m")
        except Exception as e:  # noqa: BLE001
            avvisi.append(f"rilievo {chiave}: {e}")
    if arrays:
        np.savez_compressed(cartella / "dem.npz", **arrays)
    risultato["errori"] = errori
    risultato["avvisi"] = avvisi
    nucleo = all(risultato[k] for k in ("province", "costa", "laghi", "fiumi", "paesi"))
    if nucleo:
        (cartella / "sfondo.json").write_text(json.dumps(risultato, separators=(",", ":")))
    else:
        (cartella / "sfondo_parziale.json").write_text(json.dumps(risultato, separators=(",", ":")))
    for e in errori + avvisi:
        print("ATTENZIONE:", e, file=sys.stderr)
    return nucleo


def carica(cartella):
    """Sfondo pronto all'uso; se non c'e' (o e' solo parziale) usa quello che c'e', altrimenti uno vuoto."""
    vuoto = {"province": [], "costa": [], "laghi": [], "fiumi": [], "comuni": [], "regioni": [], "paesi": [],
             "errori": ["sfondo non disponibile"], "avvisi": []}
    for nome in ("sfondo.json", "sfondo_parziale.json"):
        p = Path(cartella) / nome
        if p.exists():
            try:
                s = json.loads(p.read_text())
                for k, v in vuoto.items():
                    s.setdefault(k, v)
                return s
            except Exception:  # noqa: BLE001
                pass
    return vuoto


def carica_dem(cartella):
    """{'fine': (griglia, lon0, lat0, passo), 'largo': (...)} oppure {} se manca il rilievo."""
    p = Path(cartella) / "dem.npz"
    if not p.exists():
        return {}
    z = np.load(p)
    out = {}
    for k in ("fine", "largo"):
        if k in z.files:
            lon0, lat0, passo = z[k + "_meta"]
            out[k] = (z[k].astype("float64"), float(lon0), float(lat0), float(passo))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scarica", required=True, help="cartella dove mettere i dati")
    a = ap.parse_args()
    ok = costruisci(a.scarica)
    print("sfondo completo" if ok else "sfondo incompleto: le mappe usciranno con meno dettagli")


if __name__ == "__main__":
    main()
