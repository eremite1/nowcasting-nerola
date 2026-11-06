#!/usr/bin/env python3
"""Il radar come "stazioni meteo virtuali" per le pagelle dei modelli.

Le pagelle confrontano la pioggia prevista dai modelli con quella misurata dalle stazioni. Le stazioni pero' sono poche
e la pioggia, soprattutto quella dei temporali, puo' cadere tutta in mezzo a due stazioni. Il radar invece vede
dappertutto: qui si prende la pioggia caduta in ogni ora (prodotto SRT1 della Protezione Civile) su una griglia di
punti fissi sul Lazio, uno ogni 0,15 gradi (circa 15 km), come se in ogni punto ci fosse un pluviometro.

Ad ogni giro:
  1. radar_punti.csv: pioggia caduta ora per ora nei punti (ultimi 8 giorni), stesso formato delle osservazioni delle
     stazioni (lo legge la pagella di meteo-mappe, che confronta tutti i modelli con questi punti).
  2. wrf_punti.csv: la pioggia prevista ora per ora nei punti dall'ultima corsa di WRF Nerola, nello stesso formato delle
     previsioni delle Mappe Lazio: cosi' anche WRF Nerola entra nella stessa pagella, con le stesse regole.

  radar_punti.py --precedente CARTELLA_PUBBLICATA --out CARTELLA_NUOVA [--wrf d01_tutto.npz]
"""
import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import fonti
import geo

LAT = np.round(np.arange(41.25, 42.9001, 0.15), 2)
LON = np.round(np.arange(11.70, 13.9501, 0.15), 2)
GIORNI = 8
ORE_INDIETRO = 4          # ore passate che si provano a recuperare se mancano
CAMPI_OSS = ["id", "nome", "lat", "lon", "quota", "ora", "T", "Td", "V", "P"]
CAMPI_PREV = ["corsa", "valido", "ore", "passo", "id", "T", "Td", "V", "P", "h"]


def punti():
    out = []
    for la in LAT:
        for lo in LON:
            out.append((f"R{la * 100:04.0f}_{lo * 100:04.0f}", float(la), float(lo)))
    return out


def iso_ora(t):
    return t.strftime("%Y-%m-%dT%H:00Z")


def campiona(campo, griglia, pp):
    """Media su 3x3 km attorno a ogni punto (il radar non e' un pluviometro: un po' di media toglie il rumore)."""
    c, r = griglia.indici(np.array([p[2] for p in pp]), np.array([p[1] for p in pp]))
    out = []
    for ci, ri in zip(np.round(c).astype(int), np.round(r).astype(int)):
        if not (1 <= ri < campo.shape[0] - 1 and 1 <= ci < campo.shape[1] - 1):
            out.append(None)
            continue
        v = campo[ri - 1:ri + 2, ci - 1:ci + 2]
        v = v[np.isfinite(v) & (v >= 0)]
        out.append(round(float(v.mean()), 2) if v.size >= 5 else None)
    return out


# griglia comune della pagella unica (stessa di meteo-mappe/motore/verifica.py): 0,025 gradi sul Lazio, riga 0 = sud
G_LON0, G_LAT0, G_PASSO, G_NX, G_NY = 11.30, 41.05, 0.025, 118, 86


def su_griglia_comune(a, g):
    """Pioggia oraria del radar sulla griglia comune (media dei pixel radar nella cella), mm x 10, 65535 = no dato."""
    from scipy import ndimage
    ok = np.isfinite(a)
    k = 3
    s = ndimage.uniform_filter(np.where(ok, a, 0.0), k, mode="constant")
    n = ndimage.uniform_filter(ok.astype("float64"), k, mode="constant")
    m = np.where(n > 0.5, s / np.maximum(n, 1e-9), np.nan)
    lon = G_LON0 + G_PASSO * (np.arange(G_NX) + 0.5)
    lat = G_LAT0 + G_PASSO * (np.arange(G_NY) + 0.5)
    LO, LA = np.meshgrid(lon, lat)
    c, r = g.indici(LO, LA)
    c, r = np.round(c).astype(int), np.round(r).astype(int)
    dentro = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
    v = np.full(LO.shape, np.nan)
    v[dentro] = m[r[dentro], c[dentro]]
    return np.where(np.isfinite(v), np.clip(np.round(v * 10), 0, 65534), 65535).astype("<u2")


def archivio_orario(precedente, out, nuove, adesso):
    """radar_orario.npz: le mappe orarie degli ultimi GIORNI giorni sulla griglia comune (per la pagella unica)."""
    tutte = {}
    vecchio = Path(precedente) / "radar_orario.npz" if precedente else None
    if vecchio and vecchio.exists():
        try:
            z = np.load(vecchio)
            tutte = {str(t): z["mm"][k] for k, t in enumerate(z["ore"])}
        except Exception as e:  # noqa: BLE001
            print("archivio orario precedente illeggibile:", e, file=sys.stderr)
    tutte.update(nuove)
    limite = iso_ora(adesso - timedelta(days=GIORNI))
    ore = sorted(t for t in tutte if t >= limite)
    if ore:
        np.savez_compressed(Path(out) / "radar_orario.npz", ore=np.array(ore), mm=np.stack([tutte[t] for t in ore]))
    return len(ore)


def osservazioni(precedente, out, adesso):
    pp = punti()
    griglie = {}
    righe, viste = [], set()
    vecchio = Path(precedente) / "radar_punti.csv" if precedente else None
    limite = iso_ora(adesso - timedelta(days=GIORNI))
    if vecchio and vecchio.exists():
        for r in csv.DictReader(open(vecchio, newline="")):
            if r["ora"] >= limite:
                righe.append(r)
                viste.add(r["ora"])
    nuove = 0
    ultima = adesso.replace(minute=0, second=0, microsecond=0)
    # ore gia' nell'archivio orario sulla griglia comune; se l'archivio non c'e' ancora si recuperano 2 giorni
    gia_griglia = set()
    va = Path(precedente) / "radar_orario.npz" if precedente else None
    if va and va.exists():
        try:
            gia_griglia = {str(x) for x in np.load(va)["ore"]}
        except Exception:  # noqa: BLE001
            pass
    indietro = ORE_INDIETRO if gia_griglia else 48
    for k in range(indietro):
        t = ultima - timedelta(hours=k)
        if iso_ora(t) in viste and iso_ora(t) in gia_griglia:
            continue
        try:
            dati = fonti.scarica("SRT1", int(t.timestamp() * 1000))
        except fonti.NonTrovato:
            print(f"SRT1 {iso_ora(t)}: non ancora disponibile")
            continue
        except Exception as e:  # noqa: BLE001
            print(f"SRT1 {iso_ora(t)}: errore {e}", file=sys.stderr)
            continue
        a, g, _ = geo.leggi_geotiff(dati)
        a = np.where(a < 0, np.nan, a).astype("float64")
        if np.nanmax(a) > 500:
            print(f"SRT1 {iso_ora(t)}: valori fuori scala, scartato", file=sys.stderr)
            continue
        try:
            griglie[iso_ora(t)] = su_griglia_comune(a, g)
        except Exception as e:  # noqa: BLE001
            print(f"SRT1 {iso_ora(t)}: griglia comune non fatta: {e}", file=sys.stderr)
        for (sid, la, lo), v in zip(pp, campiona(a, g, pp)):
            if v is None or iso_ora(t) in viste:
                continue
            righe.append({"id": sid, "nome": f"Radar {la:.2f}N {lo:.2f}E".replace(".", ","), "lat": la, "lon": lo,
                          "quota": "", "ora": iso_ora(t), "T": "", "Td": "", "V": "", "P": v})
        nuove += 1
    with open(Path(out) / "radar_punti.csv", "w", newline="") as f:
        w = csv.DictWriter(f, CAMPI_OSS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(righe, key=lambda r: (r["ora"], r["id"])):
            w.writerow(r)
    print(f"radar nei punti: {nuove} ore nuove, {len(righe)} righe, {len(pp)} punti")
    print(f"radar orario sulla griglia comune: {archivio_orario(precedente, out, griglie, adesso)} ore")


def previsioni_wrf(percorso, out):
    import wrf as wrf_mod
    w = wrf_mod.PrevisioneWRF(percorso)
    pp = punti()
    # cella del WRF piu' vicina a ogni punto (solo se il punto e' dentro il dominio)
    celle = []
    for sid, la, lo in pp:
        d2 = (w.lat - la) ** 2 + ((w.lon - lo) * np.cos(np.radians(la))) ** 2
        iy, ix = np.unravel_index(int(np.argmin(d2)), d2.shape)
        dentro = 0 < iy < w.lat.shape[0] - 1 and 0 < ix < w.lat.shape[1] - 1 and d2[iy, ix] < 0.05 ** 2
        celle.append((sid, iy, ix, dentro))
    righe = []
    corsa = w.corsa.strftime("%Y-%m-%dT%H:%MZ")
    for i in range(len(w.tempi) - 1):
        fine = w.tempi[i + 1]
        ore = int(round((fine - w.corsa).total_seconds() / 3600))
        mm = w.tassi[i] * (w.tempi[i + 1] - w.tempi[i]).total_seconds() / 3600
        for sid, iy, ix, dentro in celle:
            if dentro:
                righe.append([corsa, fine.strftime("%Y-%m-%dT%H:%MZ"), ore, 1, sid, "", "", "", round(float(mm[iy, ix]), 2), ""])
    with open(Path(out) / "wrf_punti.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(CAMPI_PREV)
        cw.writerows(righe)
    print(f"WRF nei punti: corsa {corsa}, {len(righe)} righe")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precedente", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wrf", default="")
    a = ap.parse_args()
    adesso = datetime.now(timezone.utc)
    try:
        osservazioni(a.precedente, a.out, adesso)
    except Exception as e:  # noqa: BLE001
        print("ATTENZIONE: radar nei punti non aggiornato:", e, file=sys.stderr)
        vecchio = Path(a.precedente) / "radar_punti.csv" if a.precedente else None
        if vecchio and vecchio.exists():
            (Path(a.out) / "radar_punti.csv").write_bytes(vecchio.read_bytes())
        vo = Path(a.precedente) / "radar_orario.npz" if a.precedente else None
        if vo and vo.exists():
            (Path(a.out) / "radar_orario.npz").write_bytes(vo.read_bytes())
    if a.wrf and Path(a.wrf).exists():
        try:
            previsioni_wrf(a.wrf, a.out)
        except Exception as e:  # noqa: BLE001
            print("ATTENZIONE: WRF nei punti non scritto:", e, file=sys.stderr)


if __name__ == "__main__":
    main()
