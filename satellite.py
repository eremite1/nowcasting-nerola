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
import nubi  # noqa: E402
import previsione_nubi  # noqa: E402
import riassunto  # noqa: E402

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
    # infrarosso su una zona molto piu' grande (circa 2 km per pixel), solo per la previsione: le nubi che nelle
    # prossime 6 ore entreranno nella mappa adesso sono FUORI dalla mappa, e vanno viste per poterle far entrare
    "largo": ("mtg_fd:ir105_hrfi", "mtg_fd:mtg_fd_ir105_hrfi_grayscale", 0, False),
}
BBOX_LARGO = (BBOX[0] - 2.6, BBOX[1] + 2.6, BBOX[2] - 2.0, BBOX[3] + 2.0)
LARGHEZZA_LARGO = 400
ALTEZZA_LARGO = int(round(LARGHEZZA_LARGO * (BBOX_LARGO[3] - BBOX_LARGO[2]) / (BBOX_LARGO[1] - BBOX_LARGO[0])
                          / math.cos(math.radians(42.1))))
# prodotti "per esperti" (oltre alla mappa con le nubi, che e' il prodotto principale)
PRODOTTI = [("visibile", "Visibile HD"), ("colori", "Colori reali"), ("infrarosso", "Infrarosso"), ("topnubi", "Top nubi")]
# cambiando VERSIONE tutte le immagini vengono ridisegnate (serve quando cambia lo stile)
VERSIONE = "v3"
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
    bb, lg, al = (BBOX_LARGO, LARGHEZZA_LARGO, ALTEZZA_LARGO) if nome == "largo" else (BBOX, LARGHEZZA, ALTEZZA)
    url = (f"{WMS}?service=WMS&version=1.3.0&request=GetMap&layers={strato}&styles={stile}&crs=EPSG:4326"
           f"&bbox={bb[2]},{bb[0]},{bb[3]},{bb[1]}&width={lg}&height={al}"
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

def strato_nubi(alfa, lum, T, pk):
    """Nubi e loro ombra come immagine trasparente (RGBA 0-255), da mettere sopra la mappa."""
    a = np.clip(np.nan_to_num(alfa), 0, 1)
    dy, dx = 2.0 / pk, 1.4 / pk                      # ombra spostata di circa 2 km a sud e 1,4 km a est
    om = ndimage.gaussian_filter(ndimage.shift(a, (dy, dx), order=1, mode="nearest"), max(1.0, 0.8 / pk))
    s = 0.40 * om
    A = a + s * (1 - a)
    lum = np.clip(np.nan_to_num(lum, nan=0.85), 0, 1)
    rgb = np.stack([lum, lum, lum], axis=-1)
    if T is not None:
        freddo = np.clip((-25.0 - np.nan_to_num(T, nan=0.0)) / 40.0, 0, 1)[..., None]
        rgb = rgb * (1 - 0.10 * freddo) + np.array([0.86, 0.92, 1.0]) * 0.10 * freddo
    rgb = rgb * (a / np.maximum(A, 1e-6))[..., None]   # dove c'e' solo ombra il colore e' nero (trasparente)
    out = np.zeros(a.shape + (4,), dtype=np.uint8)
    out[..., :3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(A * 255, 0, 255).astype(np.uint8)
    return out


def webp(rgba, percorso, qualita=72):
    Image.fromarray(rgba, "RGBA").save(percorso, "WEBP", quality=qualita, method=4)


def campi_zona(cache, t, tab, mappa_b, z, pk):
    """Temperatura, riflettanza, nuvolosita' e luminosita' delle nubi sui pixel della zona."""
    g = leggi(cache, "ir", t)
    if g is None:
        return None
    T = a_temp(disegna.ricampiona(sfuma(g, STRATI["ir"][2]), mappa_b), tab)
    cosz = coseno_zenitale(t, z.lon, z.lat)
    v = leggi(cache, "vis", t)
    vis = disegna.ricampiona(sfuma(v, STRATI["vis"][2]), mappa_b) if v is not None else None
    R = riflettanza(vis, cosz) if vis is not None else None
    alfa, lum = nubi.copertura(T, R, giorno(cosz), pk)
    return {"T": T, "vis": vis, "cosz": cosz, "alfa": alfa, "lum": lum}


def campi_larghi(cache, seq, mezza, tab, lon_m, lat_m):
    """Nubi sulla zona grande (solo infrarosso, circa 2 km) per gli istanti della previsione. Dentro la mappa si usano
    le nubi vere della mappa (visibile + infrarosso); fuori, quelle dell'infrarosso rese simili (stessa distribuzione)
    a quelle della mappa nella parte in comune. Ritorna il dizionario per previsione_nubi.prevedi o None."""
    dx = (BBOX_LARGO[1] - BBOX_LARGO[0]) / LARGHEZZA_LARGO
    dy = (BBOX_LARGO[3] - BBOX_LARGO[2]) / ALTEZZA_LARGO
    lon_l = BBOX_LARGO[0] + (np.arange(LARGHEZZA_LARGO) + 0.5) * dx
    lat_l = BBOX_LARGO[3] - (np.arange(ALTEZZA_LARGO) + 0.5) * dy
    LO, LA = np.meshgrid(lon_l, lat_l)
    pk_l = dx * 111.0 * math.cos(math.radians(42.1))
    # posizione dei pixel larghi nella griglia della mappa (meta' definizione)
    H, W = lon_m.shape
    l0, l1 = lon_m[0, 0], lon_m[0, -1]
    a0, a1 = lat_m[0, 0], lat_m[-1, 0]
    jx = ((LO - l0) / (l1 - l0) * (W - 1)).astype("float32")
    iy = ((LA - a0) / (a1 - a0) * (H - 1)).astype("float32")
    dentro = (jx >= 0) & (jx <= W - 1) & (iy >= 0) & (iy <= H - 1)
    fattore = max(pk_l / ((l1 - l0) / (W - 1) * 111.0 * math.cos(math.radians(42.1))), 1.0)
    campi = []
    for t in seq:
        g = leggi(cache, "largo", t)
        if g is None:
            return None
        T = a_temp(g, tab)
        alfa_ir, _ = nubi.copertura(T, None, np.zeros(T.shape), pk_l)
        alfa_ir = ndimage.gaussian_filter(alfa_ir, 1.5)    # niente quadretti
        al_m, lu_m = mezza[t]
        sig = fattore / 2.5
        a_m = cv2.remap(ndimage.gaussian_filter(al_m, sig).astype("float32"), jx, iy, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
        l_m = cv2.remap(ndimage.gaussian_filter(lu_m, sig).astype("float32"), jx, iy, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
        m = dentro & np.isfinite(a_m)
        if m.sum() > 2000:
            p = np.linspace(0, 100, 101)
            qi, qm = np.percentile(alfa_ir[m], p), np.percentile(a_m[m], p)
            qi, k = np.unique(qi, return_index=True)
            alfa_f = np.interp(alfa_ir, qi, qm[k]) if qi.size > 1 else alfa_ir
        else:
            alfa_f = alfa_ir
        alfa = np.where(m, a_m, alfa_f)
        lum = np.where(m, l_m, 0.80 + 0.20 * alfa_f)
        campi.append((alfa.astype("float64"), lum.astype("float64")))
    return {"campi": campi, "lon": LO, "lat": LA, "passo_km": pk_l}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dati", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nowcast", required=True)
    ap.add_argument("--wrf", default="")
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
    with open(a.nowcast, "rb") as f:
        nc = pickle.load(f)
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
    out.mkdir(parents=True, exist_ok=True)
    (out / "satellite_taratura.json").write_text(json.dumps({"tabella": tab, "info": info_tar}))
    print("taratura infrarosso:", info_tar)
    wrf_prev = None
    if a.wrf and Path(a.wrf).exists():
        try:
            import wrf as wrf_mod
            wrf_prev = wrf_mod.PrevisioneWRF(a.wrf)
        except Exception as e:  # noqa: BLE001
            print("ATTENZIONE: WRF non leggibile per le nubi:", e, file=sys.stderr)

    t0 = da_iso(nc["tempi"][0])
    istanti = disponibili.get("ir", [])
    indice = {"prodotti": PRODOTTI, "taratura": info_tar, "zone": {}}

    # ---- fulmini (ultime 2 ore)
    fulmini = []
    for t in disponibili.get("lampi", []):
        im = leggi(cache, "lampi", t)
        if im is None:
            continue
        for fl in riassunto.fulmini_da_immagine(im, BBOX, LARGHEZZA, ALTEZZA):
            fl["ora"] = iso(t)
            fl["t"] = t
            fulmini.append(fl)
    indice["fulmini"] = [{k: v for k, v in fl.items() if k != "t"} for fl in fulmini]

    # ---- immagini osservate, zona per zona
    zone = {k: disegna.Zona(k, c) for k, c in disegna.AREE.items()}
    mezza = {}      # Lazio a meta' definizione: istante -> (alfa, lum), per la previsione delle nubi
    copertura_oss = {}
    for chiave, z in zone.items():
        cart = out / chiave / "sat"
        cart.mkdir(parents=True, exist_ok=True)
        pk = nubi.passo_km(z)
        mx = ((z.lon - BBOX[0]) / (BBOX[1] - BBOX[0]) * LARGHEZZA - 0.5).astype("float32")
        my = ((BBOX[3] - z.lat) / (BBOX[3] - BBOX[2]) * ALTEZZA - 0.5).astype("float32")
        mappa_b = (mx, my)
        voce = {"nubi": [], "nubiprev": [], "esperti": {p: [] for p, _ in PRODOTTI}}
        fatti = 0
        for t in istanti:
            nomi = {p: f"{p}_{VERSIONE}_{t:%Y%m%d_%H%M}.jpg" for p, _ in PRODOTTI}
            nomi["nubi"] = f"nubi_{VERSIONE}_{t:%Y%m%d_%H%M}.webp"
            serve_campo = chiave == "lazio" and t in istanti[-4:]
            gia = prec and all((prec / chiave / "sat" / n).exists() for n in nomi.values()) \
                and not (prec / chiave / "sat" / f"rifare_{t:%Y%m%d_%H%M}").exists()
            cz = None
            if gia:
                for n in nomi.values():
                    shutil.copy2(prec / chiave / "sat" / n, cart / n)
            else:
                cz = campi_zona(cache, t, tab, mappa_b, z, pk)
                if cz is None:
                    continue
                T, vis, cosz = cz["T"], cz["vis"], cz["cosz"]
                webp(strato_nubi(cz["alfa"], cz["lum"], T, pk), cart / nomi["nubi"])
                irg = np.clip((35.0 - T) / 105.0, 0, 1) ** 0.9
                w = giorno(cosz)
                if vis is not None:
                    v = np.clip((riflettanza(vis, cosz) - 0.10) / 0.72, 0, 1) ** 0.8
                    gv = w * v + (1 - w) * irg
                else:
                    gv = irg
                jpeg(np.repeat(gv[..., None], 3, axis=-1), cart / nomi["visibile"])
                geo = leggi(cache, "geo", t)
                if geo is not None:
                    geo = sfuma(geo, STRATI["geo"][2])
                    geo = np.stack([disegna.ricampiona(geo[..., c], mappa_b) for c in range(3)], axis=-1)
                    jpeg(np.nan_to_num(geo), cart / nomi["colori"])
                else:
                    (cart / f"rifare_{t:%Y%m%d_%H%M}").write_text("")
                jpeg(np.repeat(irg[..., None], 3, axis=-1), cart / nomi["infrarosso"])
                jpeg(scala_topnubi(T), cart / nomi["topnubi"])
                fatti += 1
            if serve_campo:
                if cz is None:
                    cz = campi_zona(cache, t, tab, mappa_b, z, pk)
                if cz is not None:
                    al = cv2.resize(np.nan_to_num(cz["alfa"]).astype("float32"), (z.W // 2, z.H // 2), interpolation=cv2.INTER_AREA)
                    lu = cv2.resize(np.nan_to_num(cz["lum"], nan=0.85).astype("float32"), (z.W // 2, z.H // 2), interpolation=cv2.INTER_AREA)
                    mezza[t] = (al.astype("float64"), lu.astype("float64"))
            voce["nubi"].append({"file": f"sat/{nomi['nubi']}", "ora": iso(t)})
            for p, _ in PRODOTTI:
                if (cart / nomi[p]).exists():
                    voce["esperti"][p].append({"file": f"sat/{nomi[p]}", "ora": iso(t)})
        indice["zone"][chiave] = voce
        print(f"{z.nome}: satellite {len(istanti)} istanti, disegnati ora {fatti}")

    # ---- previsione delle nubi (fino a +6 ore da adesso)
    zl = zone["lazio"]
    pk_mezza = nubi.passo_km(zl) * 2
    lon_m = cv2.resize(zl.lon.astype("float32"), (zl.W // 2, zl.H // 2), interpolation=cv2.INTER_AREA).astype("float64")
    lat_m = cv2.resize(zl.lat.astype("float32"), (zl.W // 2, zl.H // 2), interpolation=cv2.INTER_AREA).astype("float64")
    prev_nubi, info_nubi = {}, {}
    seq = sorted(mezza)
    seq = [t for t in seq if (seq[-1] - t).total_seconds() <= 1800 and (seq[-1] - t).total_seconds() % 600 == 0] if seq else []
    if seq:
        t_ult = seq[-1]
        fine = t0 + timedelta(minutes=360)
        tempi_voluti = []
        t = t_ult + timedelta(minutes=10)
        while t <= fine:
            tempi_voluti.append(t)
            t += timedelta(minutes=10)
        try:
            try:
                largo = campi_larghi(cache, seq, mezza, tab, lon_m, lat_m)
            except Exception as e:  # noqa: BLE001
                largo = None
                print("ATTENZIONE: zona grande per la previsione non disponibile:", e, file=sys.stderr)
            prev_nubi, info_nubi = previsione_nubi.prevedi([mezza[t] for t in seq], t_ult, tempi_voluti, wrf_prev, lon_m, lat_m,
                                                           pk_mezza, largo=largo)
        except Exception as e:  # noqa: BLE001
            info_nubi = {"errore": str(e)[:300]}
            print("ATTENZIONE: previsione nubi non riuscita:", e, file=sys.stderr)
        print("previsione nubi:", info_nubi)
        for chiave, z in zone.items():
            cart = out / chiave / "sat"
            pk = nubi.passo_km(z)
            # dai pixel della zona alla griglia Lazio a meta' definizione
            jx = (((z.lon - zl.lon0) / (zl.lon1 - zl.lon0) * zl.W - 0.5) / 2).astype("float32")
            iy = (((zl.lat1 - z.lat) / (zl.lat1 - zl.lat0) * zl.H - 0.5) / 2).astype("float32")
            for t, (al, lu) in sorted(prev_nubi.items()):
                A = cv2.remap(al.astype("float32"), jx, iy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                L = cv2.remap(lu.astype("float32"), jx, iy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                # le nubi previste sono per natura meno dettagliate: immagine a meta' definizione (file 4 volte piu' leggero)
                A = cv2.resize(A, (z.W // 2, z.H // 2), interpolation=cv2.INTER_AREA)
                L = cv2.resize(L, (z.W // 2, z.H // 2), interpolation=cv2.INTER_AREA)
                n = f"nubiprev_{t:%Y%m%d_%H%M}.webp"
                webp(strato_nubi(A, L, None, pk * 2), cart / n, qualita=65)
                indice["zone"][chiave]["nubiprev"].append({"file": f"sat/{n}", "ora": iso(t),
                                                           "minuti_dal_satellite": int((t - t_ult).total_seconds() // 60)})
    indice["previsione_nubi"] = info_nubi

    # ---- griglia per "tocca la mappa" e riassunto di Nerola
    passi = len(nc["tempi"])
    g_radar = nc["griglia"]
    lon_g = riassunto.G_LON0 + (np.arange(riassunto.G_NX) + 0.5) * riassunto.G_PASSO
    lat_g = riassunto.G_LAT1 - (np.arange(riassunto.G_NY) + 0.5) * riassunto.G_PASSO
    LO, LA = np.meshgrid(lon_g, lat_g)
    cr, rr = g_radar.indici(LO, LA)
    mappa_g = (np.asarray(cr, dtype="float32"), np.asarray(rr, dtype="float32"))
    pioggia_g = np.zeros((passi, riassunto.G_NY, riassunto.G_NX), dtype=np.uint8)
    for k in range(passi):
        # massimo locale (circa 3 km) prima di campionare: un rovescio piccolo non deve sparire
        campo = ndimage.maximum_filter(np.nan_to_num(nc["campi"][k]), size=3)
        v = disegna.ricampiona(campo, mappa_g)
        pioggia_g[k] = np.clip(np.nan_to_num(v) * 10, 0, 255).astype(np.uint8)
    nubi_g = np.zeros_like(pioggia_g)
    jx = (((LO - zl.lon0) / (zl.lon1 - zl.lon0) * zl.W - 0.5) / 2).astype("float32")
    iy = (((zl.lat1 - LA) / (zl.lat1 - zl.lat0) * zl.H - 0.5) / 2).astype("float32")
    ultimo_oss = mezza[seq[-1]][0] if seq else None
    nubi_nerola = []
    jn = ((riassunto.NEROLA[1] - zl.lon0) / (zl.lon1 - zl.lon0) * zl.W - 0.5) / 2
    inr = ((zl.lat1 - riassunto.NEROLA[0]) / (zl.lat1 - zl.lat0) * zl.H - 0.5) / 2
    raggio = max(1, int(round(3.0 / pk_mezza)))
    def a_nerola(campo):
        j, i = int(round(jn)), int(round(inr))
        return float(np.nanmean(campo[max(0, i - raggio):i + raggio + 1, max(0, j - raggio):j + raggio + 1]))
    for k in range(passi):
        tk = t0 + timedelta(minutes=10 * k)
        campo = None
        if prev_nubi:
            vicino = min(prev_nubi, key=lambda t: abs((t - tk).total_seconds()))
            if abs((vicino - tk).total_seconds()) <= 300:
                campo = prev_nubi[vicino][0]
        if campo is None:
            campo = ultimo_oss
        if campo is not None:
            nubi_g[k] = np.clip(cv2.remap(campo.astype("float32"), jx, iy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * 100,
                                0, 100).astype(np.uint8)
            nubi_nerola.append((tk, a_nerola(campo)))
    if ultimo_oss is not None and nubi_nerola:
        nubi_nerola[0] = (t0, a_nerola(ultimo_oss))
    (out / "griglia.bin").write_bytes(pioggia_g.tobytes() + nubi_g.tobytes())
    indice["griglia"] = {"lon0": riassunto.G_LON0, "lat1": riassunto.G_LAT1, "passo": riassunto.G_PASSO,
                         "nx": riassunto.G_NX, "ny": riassunto.G_NY, "passi": passi, "t0": iso(t0), "minuti": 10,
                         "file": "griglia.bin", "contenuto": "pioggia mm/h x10 [passi,ny,nx] poi nuvolosita % [passi,ny,nx], uint8"}
    # pioggia a Nerola (massimo entro 3 km) e pioggia vicina adesso
    cN, rN = g_radar.indici(np.array([riassunto.NEROLA[1]]), np.array([riassunto.NEROLA[0]]))
    cN, rN = int(round(float(cN[0]))), int(round(float(rN[0])))
    pn = [float(np.nanmax(np.nan_to_num(nc["campi"][k][max(0, rN - 3):rN + 4, max(0, cN - 3):cN + 4]))) for k in range(passi)]
    vicina = None
    adesso = np.nan_to_num(nc["campi"][0])
    yy, xx = np.nonzero(adesso >= 0.2)
    if len(yy):
        d = np.hypot((yy - rN) * g_radar.dy / 1000, (xx - cN) * g_radar.dx / 1000)
        k = int(np.argmin(d))
        if d[k] <= 40:
            lo_p, la_p = g_radar.proj.indietro(g_radar.x0 + xx[k] * g_radar.dx, g_radar.y0 - yy[k] * g_radar.dy)
            vicina = (float(d[k]), riassunto.verso(*riassunto.NEROLA, float(la_p), float(lo_p)), float(adesso[yy[k], xx[k]]))
    tipo_nerola, gr = None, None
    if Path(f"{a.dati}/tipo.pkl").exists():
        with open(f"{a.dati}/tipo.pkl", "rb") as f:
            tdat = pickle.load(f)
        tipo_nerola, gr = tdat.get("nerola"), tdat.get("grandine")
        extra_n = {k: tdat[k] for k in ("temperatura", "cape", "quota_neve") if k in tdat}
    else:
        extra_n = {}
    indice["nerola"] = riassunto.riassunto_nerola(t0, pn, vicina, nubi_nerola, fulmini, None, info_nubi, tipo_nerola, gr)
    indice["nerola"].update(extra_n)
    if tipo_nerola:
        indice["nerola"]["tipo_10min"] = tipo_nerola
    indice["nerola"]["pioggia_10min"] = [round(v, 1) for v in pn]
    indice["nerola"]["nubi_10min"] = [round(c, 2) for _, c in nubi_nerola]
    indice["t0"] = iso(t0)
    print("Nerola:", indice["nerola"])
    (out / "satellite.json").write_text(json.dumps(indice, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
