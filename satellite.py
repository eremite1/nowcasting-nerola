#!/usr/bin/env python3
"""Satellite ad alta definizione: Meteosat di terza generazione (MTG, strumenti FCI e LI) dal servizio EUMETView di EUMETSAT.

  satellite.py --cache CARTELLA_CACHE --dati CARTELLA_SCARICO --sfondo CARTELLA_SFONDO --out CARTELLA_PUBBLICA
               [--precedente CARTELLA_RISULTATI_VECCHI]

1) SCARICA (solo i fotogrammi che non ha gia') le immagini delle ultime 2 ore, una ogni 10 minuti, sul Lazio:
     vis06_hrfi     visibile 0,6 micron, 500 m (solo di giorno)
     ir105_hrfi     infrarosso 10,5 micron, 1 km (giorno e notte)
     rgb_geocolour  colori reali di giorno, nubi e luci delle citta' di notte
     li_afa         fulmini (Lightning Imager: area dei lampi negli ultimi minuti)
   EUMETView disegna i pixel "a quadretti" (vicino piu' vicino): qui vengono sfumati in base alla loro dimensione
   vera e poi ricampionati, cosi' l'immagine e' nitida ma senza quadretti.

2) TARA LE TEMPERATURE: le immagini infrarosse di EUMETView sono in grigi, senza scala. Le temperature vere delle nubi
   le da' l'infrarosso della Protezione Civile (in gradi C, ma a bassa definizione). Confrontando le due immagini dello
   stesso istante (stessa distribuzione dei valori: "quantile matching") si ricava, ogni volta, la corrispondenza
   grigio -> temperatura. Cosi' il "top nubi" mostra temperature misurate, non colori a caso.

3) DISEGNA cinque prodotti per Lazio e Sabina:
     sandwich   radar + satellite (nubi basse gialle, nubi alte bianco-azzurre, terra e mare blu scuro), con i fulmini
     visibile   visibile ad alta definizione (di notte passa da solo all'infrarosso)
     colori     colori reali / luci notturne
     infrarosso infrarosso in grigi (temperatura delle nubi)
     topnubi    temperatura della cima delle nubi a colori, con i fulmini

Dati: "Contains modified EUMETSAT Meteosat data" (prodotti immagine EUMETView, licenza CC BY 4.0).
"""
import argparse
import io
import json
import math
import os
import pickle
import re
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import disegna  # noqa: E402

WMS = "https://view.eumetsat.int/geoserver/wms"
BBOX = (11.30, 14.25, 41.05, 43.20)          # lon_min, lon_max, lat_min, lat_max (= zona Lazio, la Sabina e' dentro)
LARGHEZZA = 1200
ALTEZZA = int(round(LARGHEZZA * (BBOX[3] - BBOX[2]) / (BBOX[1] - BBOX[0])))
ORE = 2.0
PASSO = 10                                     # minuti tra un'immagine e l'altra di MTG
STRATI = {
    # nome: (strato WMS, stile, dimensione vera del pixel in km sull'Italia centrale, solo di giorno?)
    "ir": ("mtg_fd:ir105_hrfi", "mtg_fd:mtg_fd_ir105_hrfi_grayscale", 1.4, False),
    "vis": ("mtg_fd:vis06_hrfi", "", 0.7, True),
    "geo": ("mtg_fd:rgb_geocolour", "", 1.4, False),
    "lampi": ("mtg_fd:li_afa", "", 0, False),
}
PRODOTTI = [("sandwich", "Radar + satellite"), ("visibile", "Visibile HD"), ("colori", "Colori reali"),
            ("infrarosso", "Infrarosso"), ("topnubi", "Top nubi")]
# cambiando VERSIONE tutte le immagini vengono ridisegnate (serve quando cambia lo stile)
VERSIONE = "v2"
TARATURA_DEFAULT = [40.0 - g * 110.0 / 255.0 for g in range(256)]   # usata solo se non c'e' mai stata una taratura


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%MZ")


def da_iso(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)


def richiesta(url, timeout=240):
    rich = urllib.request.Request(url, headers={"User-Agent": "nowcasting-nerola (www.meteonerola.it)"})
    with urllib.request.urlopen(rich, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "")


# --------------------------------------------------------------------------------------------- scarico

def ultimi_istanti():
    """Ora dell'ultima immagine disponibile per ogni strato (dalla descrizione del servizio)."""
    dati, _ = richiesta(f"{WMS}?service=WMS&version=1.3.0&request=GetCapabilities&namespace=mtg_fd", timeout=180)
    testo = dati.decode("utf-8", "replace")
    out = {}
    for nome, (strato, *_r) in STRATI.items():
        corto = strato.split(":")[1]
        m = re.search(r"<Name>(?:mtg_fd:)?" + re.escape(corto) + r"</Name>.*?<Dimension[^>]*default=\"([^\"]+)\"", testo, re.S)
        if m:
            out[nome] = datetime.strptime(m.group(1)[:16], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
    return out


def coseno_zenitale(t, lon, lat):
    """Coseno dell'angolo tra il Sole e la verticale (formule NOAA); negativo = Sole sotto l'orizzonte."""
    g = 2 * math.pi / 365.0 * (t.timetuple().tm_yday - 1 + (t.hour - 12) / 24.0)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                    - 0.040849 * math.sin(2 * g))
    minuti = t.hour * 60 + t.minute + eqt + 4 * np.asarray(lon)
    ha = np.radians(minuti / 4.0 - 180.0)
    la = np.radians(np.asarray(lat))
    return np.sin(la) * math.sin(decl) + np.cos(la) * math.cos(decl) * np.cos(ha)


def scarica_uno(nome, t, cartella):
    strato, stile, _, _ = STRATI[nome]
    url = (f"{WMS}?service=WMS&version=1.3.0&request=GetMap&layers={strato}&styles={stile}&crs=EPSG:4326"
           f"&bbox={BBOX[2]},{BBOX[0]},{BBOX[3]},{BBOX[1]}&width={LARGHEZZA}&height={ALTEZZA}"
           f"&format=image/png&transparent=true&time={t:%Y-%m-%dT%H:%M}:00Z")
    for n in range(2):
        try:
            dati, tipo = richiesta(url)
            if not dati.startswith(b"\x89PNG"):
                raise RuntimeError(f"risposta non e' un'immagine ({tipo}): {dati[:200]!r}")
            Image.open(io.BytesIO(dati)).load()
            (cartella / f"{t:%Y%m%d_%H%M}.png").write_bytes(dati)
            return True
        except Exception as e:  # noqa: BLE001
            ultimo = e
            time.sleep(5)
    print(f"  {nome} {iso(t)}: non scaricato ({ultimo})", file=sys.stderr)
    return False


def aggiorna_cache(cache, adesso_max=None):
    """Scarica solo i fotogrammi mancanti; cancella quelli troppo vecchi. Ritorna {strato: [istanti disponibili]}."""
    try:
        ultimi = ultimi_istanti()
    except Exception as e:  # noqa: BLE001
        print(f"ATTENZIONE: elenco EUMETView non disponibile ({e}), uso quello che c'e' in cache", file=sys.stderr)
        ultimi = {}
    print("ultime immagini EUMETView:", {k: iso(v) for k, v in ultimi.items()})
    lavori = []
    for nome in STRATI:
        cart = cache / nome
        cart.mkdir(parents=True, exist_ok=True)
        fine = ultimi.get(nome)
        if fine is None:
            continue
        fine = fine.replace(minute=fine.minute - fine.minute % PASSO)
        for k in range(int(ORE * 60 / PASSO) + 1):
            t = fine - timedelta(minutes=k * PASSO)
            if (cart / f"{t:%Y%m%d_%H%M}.png").exists():
                continue
            if STRATI[nome][3] and float(np.max(coseno_zenitale(t, np.array([BBOX[0], BBOX[1]]), np.array([BBOX[2], BBOX[3]])))) < -0.05:
                continue   # visibile di notte: inutile
            lavori.append((nome, t, cart))
    print(f"immagini da scaricare: {len(lavori)}")
    inizio = time.time()
    with ThreadPoolExecutor(4) as pool:
        esiti = list(pool.map(lambda l: scarica_uno(*l), lavori))
    print(f"scaricate {sum(esiti)} su {len(lavori)} in {time.time() - inizio:.0f} s")
    limite = max(ultimi.values()) - timedelta(hours=ORE + 0.5) if ultimi else None
    disponibili = {}
    for nome in STRATI:
        tt = []
        for f in sorted((cache / nome).glob("*.png")):
            t = datetime.strptime(f.stem, "%Y%m%d_%H%M").replace(tzinfo=timezone.utc)
            if limite and t < limite:
                f.unlink()
                continue
            tt.append(t)
        disponibili[nome] = tt
    return disponibili


# --------------------------------------------------------------------------------------------- lettura e taratura

def leggi(cache, nome, t):
    """Immagine scaricata come matrice: grigi 0-1 (vis, ir), RGB 0-1 (geo) o RGBA 0-1 (lampi). None se manca."""
    f = cache / nome / f"{t:%Y%m%d_%H%M}.png"
    if not f.exists():
        return None
    im = Image.open(f)
    if nome == "lampi":
        return np.asarray(im.convert("RGBA"), dtype="float32") / 255.0
    rgba = np.asarray(im.convert("RGBA"), dtype="float32") / 255.0
    trasparente = rgba[..., 3] < 0.5
    if nome == "geo":
        a = rgba[..., :3].copy()
        a[trasparente] = np.nan
        return a
    g = rgba[..., :3].mean(axis=-1)
    g[trasparente] = np.nan
    return g


def sfuma(a, km):
    """Toglie i quadretti: sfumatura gaussiana larga quanto il pixel vero del satellite."""
    if km <= 0:
        return a
    px = km / ((BBOX[1] - BBOX[0]) * 111.0 * math.cos(math.radians(42.1)) / LARGHEZZA)
    sigma = max(px / 3.0, 0.6)
    if a.ndim == 3:
        return np.stack([sfuma(a[..., c], km) for c in range(a.shape[-1])], axis=-1)
    noto = np.isfinite(a).astype("float32")
    num = ndimage.gaussian_filter(np.nan_to_num(a).astype("float32"), sigma)
    den = ndimage.gaussian_filter(noto, sigma)
    return np.where(den > 0.3, num / np.maximum(den, 1e-6), np.nan)


def lonlat_bbox():
    lon = BBOX[0] + (np.arange(LARGHEZZA) + 0.5) * (BBOX[1] - BBOX[0]) / LARGHEZZA
    lat = BBOX[3] - (np.arange(ALTEZZA) + 0.5) * (BBOX[3] - BBOX[2]) / ALTEZZA
    return np.meshgrid(lon, lat)


def taratura(cache, disponibili, dpc, vecchia):
    """Tabella di 256 valori: temperatura (gradi C) per ogni grigio dell'infrarosso EUMETView."""
    if dpc is None:
        return vecchia, {"fonte": "taratura precedente (infrarosso Protezione Civile non disponibile)"}
    tempi_dpc = {da_iso(t): c for t, c in zip(dpc["tempi"], dpc["campi"])}
    lon, lat = lonlat_bbox()
    c, r = dpc["griglia"].indici(lon[::4, ::4], lat[::4, ::4])
    mappa = (np.asarray(c, dtype="float32"), np.asarray(r, dtype="float32"))
    grigi, temp = [], []
    usati = 0
    for t in disponibili.get("ir", []):
        if t not in tempi_dpc:
            continue
        g = leggi(cache, "ir", t)
        if g is None:
            continue
        g = sfuma(g, 4.0)[::4, ::4]           # alla stessa definizione (piu' grossolana) della Protezione Civile
        T = disegna.ricampiona(tempi_dpc[t], mappa)
        # la Protezione Civile scrive +47,5 dove non ha il dato: fuori dai valori possibili per Italia e mari vicini
        m = np.isfinite(g) & np.isfinite(T) & (T > -100) & (T < 45)
        if m.sum() < 5000:
            continue
        grigi.append(g[m])
        temp.append(T[m])
        usati += 1
    if not usati:
        return vecchia, {"fonte": "taratura precedente (nessuna coppia di immagini dello stesso istante)"}
    g = np.concatenate(grigi) * 255.0
    T = np.concatenate(temp)
    p = np.linspace(0.5, 99.5, 199)
    gq = np.percentile(g, p)
    tq = np.percentile(T, 100 - p)        # piu' chiaro = piu' freddo
    gq, idx = np.unique(np.round(gq, 2), return_index=True)
    tq = tq[idx]
    tq = np.minimum.accumulate(tq)        # mai piu' caldo salendo di grigio
    if len(gq) < 10 or tq[0] - tq[-1] < 15:
        return vecchia, {"fonte": "taratura precedente (scena troppo uniforme per tarare)", "coppie": usati}
    x = np.arange(256, dtype="float64")
    tab = np.interp(x, gq, tq)
    # oltre i valori visti oggi: si prosegue con la pendenza media, limitata a valori fisici
    pend = (tq[-1] - tq[0]) / (gq[-1] - gq[0])
    tab = np.where(x < gq[0], tq[0] + (x - gq[0]) * pend, tab)
    tab = np.where(x > gq[-1], tq[-1] + (x - gq[-1]) * pend, tab)
    tab = np.clip(tab, -95, 65)
    # si mescola con la taratura precedente (piu' stabile da un giro all'altro)
    if vecchia is not None and np.asarray(vecchia)[0] < 45:
        tab = 0.5 * tab + 0.5 * np.asarray(vecchia)
    return [round(float(v), 2) for v in tab], {"fonte": "infrarosso Protezione Civile", "coppie": usati,
                                                 "grigio_min": round(float(gq[0])), "grigio_max": round(float(gq[-1])),
                                                 "temp_calda": round(float(tq[0]), 1), "temp_fredda": round(float(tq[-1]), 1)}


# --------------------------------------------------------------------------------------------- colori

def scala_topnubi(T):
    """Grigi per le nubi tiepide, colori sotto -30 gradi (cime alte, temporali), bianco sotto -75."""
    T = np.nan_to_num(T, nan=20.0)
    g = np.clip((35.0 - T) / 65.0, 0, 1) ** 0.95 * 0.85 + 0.05
    rgb = np.stack([g, g, g], axis=-1)
    fasce = [(-30.0, (0.55, 0.80, 1.00)), (-40.0, (0.15, 0.45, 0.95)), (-50.0, (0.05, 0.70, 0.25)),
             (-58.0, (0.98, 0.92, 0.10)), (-64.0, (1.00, 0.50, 0.00)), (-70.0, (0.85, 0.00, 0.10)),
             (-75.0, (0.75, 0.00, 0.75)), (-82.0, (1.00, 1.00, 1.00))]
    for (ta, ca), (tb, cb) in zip(fasce[:-1], fasce[1:]):
        m = (T <= ta) & (T > tb)
        f = ((ta - T) / (ta - tb))[m][:, None]
        rgb[m] = np.array(ca) * (1 - f) + np.array(cb) * f
    rgb[T <= -82.0] = (1.0, 1.0, 1.0)
    return rgb


def a_temp(g, tab):
    tab = np.asarray(tab)
    x = np.clip(np.nan_to_num(g, nan=0.0) * 255.0, 0, 255)
    return np.interp(x, np.arange(256), tab)


def giorno(cosz):
    """0 di notte, 1 di giorno, sfumato col Sole tra 2 e 10 gradi sull'orizzonte."""
    return np.clip((cosz - 0.035) / (0.17 - 0.035), 0, 1)


def riflettanza(vis, cosz):
    """Grigio del visibile diviso (in parte) per l'altezza del Sole: al mattino e alla sera non diventa tutto buio."""
    return np.nan_to_num(vis) / np.maximum(np.clip(cosz, 0.1, 1), 1e-3) ** 0.35


def sandwich(vis, T, cosz):
    fred = np.clip((18.0 - T) / 78.0, 0, 1) ** 1.15    # 0 = caldo (suolo), 1 = -60 gradi (cime dei temporali)
    w = giorno(cosz)[..., None]
    if vis is None:
        w = np.zeros_like(w)
        vis = np.zeros_like(T)
    # riflettanza corretta per l'altezza del Sole; suolo e mare (sotto 0,24) diventano scuri, restano le nubi
    v = np.clip((riflettanza(vis, cosz) - 0.24) / 0.55, 0, 1) ** 0.9
    di_giorno = np.stack([v * 0.97, v * 0.95, np.maximum(fred, v * 0.55)], axis=-1)
    di_notte = np.stack([fred * 0.80, fred * 0.82, fred], axis=-1)
    rgb = w * di_giorno + (1 - w) * di_notte
    base = np.array([0.04, 0.05, 0.16])               # blu notte su terra e mare senza nubi
    return np.clip(base + rgb * (1 - base), 0, 1)


def sovrapponi(rgb, rgba, alfa=1.0):
    a = rgba[..., 3:4] * alfa
    return rgb * (1 - a) + rgba[..., :3] * a


def pioggia_rgba(valori):
    """Pioggia colorata (stessa scala delle mappe radar) come RGBA 0-1."""
    out = np.zeros(valori.shape + (4,), dtype="float32")
    v = np.nan_to_num(valori, nan=0.0)
    classe = np.digitize(v, disegna.SOGLIE)
    for k, (c, a) in enumerate(zip(disegna.COLORI, disegna.ALFA), start=1):
        m = classe == k
        out[m, :3] = np.array(disegna.hex_rgb(c)) / 255.0
        out[m, 3] = min(1.0, a / 255.0 + 0.1)
    return out


def jpeg(rgb, percorso):
    # leggera "maschera di contrasto": i bordi delle nubi diventano piu' netti senza inventare dettagli
    rgb = np.asarray(rgb, dtype="float32")
    morbido = np.stack([ndimage.gaussian_filter(rgb[..., c], 1.4) for c in range(3)], axis=-1)
    rgb = rgb + 0.55 * (rgb - morbido)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB").save(percorso, quality=84, optimize=True, progressive=True)


# --------------------------------------------------------------------------------------------- tutto

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dati", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precedente", default="")
    ap.add_argument("--senza-scarico", action="store_true", help="usa solo quello che c'e' gia' in cache (prove)")
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out)
    prec = Path(a.precedente) if a.precedente else None
    if a.senza_scarico:
        disponibili = {n: sorted(datetime.strptime(f.stem, "%Y%m%d_%H%M").replace(tzinfo=timezone.utc)
                                 for f in (cache / n).glob("*.png")) for n in STRATI}
    else:
        disponibili = aggiorna_cache(cache)
    with open(f"{a.dati}/radar.pkl", "rb") as f:
        radar = pickle.load(f)
    dpc = None
    if Path(f"{a.dati}/satellite.pkl").exists():
        with open(f"{a.dati}/satellite.pkl", "rb") as f:
            dpc = pickle.load(f)
    vecchia = None
    for p in ([prec / "satellite_taratura.json"] if prec else []) + [cache / "taratura.json"]:
        if p.exists():
            try:
                vecchia = json.loads(p.read_text())["tabella"]
                if vecchia[0] >= 45:      # taratura vecchia sbagliata (fatta con il +47,5 della Protezione Civile)
                    vecchia = None
                    continue
                break
            except Exception:  # noqa: BLE001
                pass
    tab, info_tar = taratura(cache, disponibili, dpc, vecchia)
    if tab is None:
        tab = TARATURA_DEFAULT
        info_tar["fonte"] = "valori di partenza (mai tarato finora)"
    (cache / "taratura.json").write_text(json.dumps({"tabella": tab, "info": info_tar}))
    (out).mkdir(parents=True, exist_ok=True)
    (out / "satellite_taratura.json").write_text(json.dumps({"tabella": tab, "info": info_tar}))
    print("taratura infrarosso:", info_tar)

    tempi_radar = {da_iso(t): c for t, c in zip(radar["tempi"], radar["campi"])}
    lon_b, lat_b = lonlat_bbox()
    istanti = disponibili.get("ir", [])
    indice = {"prodotti": PRODOTTI, "taratura": info_tar, "zone": {}}
    for chiave, cfg in disegna.AREE.items():
        z = disegna.Zona(chiave, cfg)
        cart = out / chiave / "sat"
        cart.mkdir(parents=True, exist_ok=True)
        # dalla griglia di EUMETView ai pixel della zona
        mx = ((z.lon - BBOX[0]) / (BBOX[1] - BBOX[0]) * LARGHEZZA - 0.5).astype("float32")
        my = ((BBOX[3] - z.lat) / (BBOX[3] - BBOX[2]) * ALTEZZA - 0.5).astype("float32")
        mappa_b = (mx, my)
        mappa_r = z.mappa_da(radar["griglia"])
        voce = {p: [] for p, _ in PRODOTTI}
        fatti = 0
        for t in istanti:
            nomi = {p: f"{p}_{VERSIONE}_{t:%Y%m%d_%H%M}.jpg" for p, _ in PRODOTTI}
            # il radar dello stesso istante deve esserci per il "sandwich", altrimenti lo si rifara' al prossimo giro
            r = tempi_radar.get(t)
            gia = prec and all((prec / chiave / "sat" / n).exists() for n in nomi.values()) \
                and not (prec / chiave / "sat" / f"rifare_{t:%Y%m%d_%H%M}").exists()
            if gia:
                for p, n in nomi.items():
                    shutil.copy2(prec / chiave / "sat" / n, cart / n)
                    voce[p].append({"file": f"sat/{n}", "ora": iso(t)})
                continue
            g = leggi(cache, "ir", t)
            if g is None:
                continue
            ir = disegna.ricampiona(sfuma(g, STRATI["ir"][2]), mappa_b)
            T = a_temp(ir, tab)
            cosz = coseno_zenitale(t, z.lon, z.lat)
            vis = leggi(cache, "vis", t)
            vis = disegna.ricampiona(sfuma(vis, STRATI["vis"][2]), mappa_b) if vis is not None else None
            lampi = leggi(cache, "lampi", t)
            if lampi is not None:
                lampi = np.stack([cv2.remap(lampi[..., c], mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                  for c in range(4)], axis=-1)
            # 1 sandwich
            rgb = sandwich(vis, T, cosz)
            if r is not None:
                rgb = sovrapponi(rgb, pioggia_rgba(disegna.ricampiona(r, mappa_r)), 0.9)
            if lampi is not None:
                rgb = sovrapponi(rgb, lampi)
            jpeg(rgb, cart / nomi["sandwich"])
            # 2 visibile (di notte: infrarosso)
            irg = np.clip((35.0 - T) / 105.0, 0, 1) ** 0.9
            w = giorno(cosz)
            if vis is not None:
                v = np.clip((riflettanza(vis, cosz) - 0.10) / 0.72, 0, 1) ** 0.8
                gv = w * v + (1 - w) * irg
            else:
                gv = irg
            jpeg(np.repeat(gv[..., None], 3, axis=-1), cart / nomi["visibile"])
            # 3 colori reali
            geo = leggi(cache, "geo", t)
            if geo is not None:
                geo = sfuma(geo, STRATI["geo"][2])
                geo = np.stack([disegna.ricampiona(geo[..., c], mappa_b) for c in range(3)], axis=-1)
                jpeg(np.nan_to_num(geo), cart / nomi["colori"])
            # 4 infrarosso
            jpeg(np.repeat(irg[..., None], 3, axis=-1), cart / nomi["infrarosso"])
            # 5 top nubi
            rgb = scala_topnubi(T)
            if lampi is not None:
                rgb = sovrapponi(rgb, lampi)
            jpeg(rgb, cart / nomi["topnubi"])
            fatti += 1
            completo = r is not None and geo is not None
            for p, n in nomi.items():
                if (cart / n).exists():
                    voce[p].append({"file": f"sat/{n}", "ora": iso(t)})
            if not completo:
                # mancava qualcosa (radar o colori): si rifara' al prossimo giro
                (cart / f"rifare_{t:%Y%m%d_%H%M}").write_text("")
        indice["zone"][chiave] = voce
        print(f"{z.nome}: satellite {len(istanti)} istanti, disegnati ora {fatti}")
    (out / "satellite.json").write_text(json.dumps(indice, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
