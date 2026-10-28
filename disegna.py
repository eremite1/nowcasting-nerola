#!/usr/bin/env python3
"""Disegna le immagini del nowcasting per la pagina web.

  disegna.py --dati CARTELLA_SCARICO --nowcast nowcast.pkl --sfondo CARTELLA_SFONDO --out CARTELLA_PUBBLICA

Per ogni zona (Lazio e Sabina) scrive:
  <zona>/fondo.png                    rilievo e mare (sotto a tutto)
  <zona>/confini.png                  confini, fiumi, nomi dei paesi (trasparente, sopra a tutto)
  <zona>/radar/AAAAMMGG_HHMM.png      pioggia osservata (trasparente), ultime 2 ore
  <zona>/previsione/AAAAMMGG_HHMM.png pioggia prevista (trasparente), prossime 6 ore
  <zona>/satellite/AAAAMMGG_HHMM.png  infrarosso del satellite, ultime 3 ore
  <zona>/ultimo.png                   immagine completa dell'ultimo radar (da usare anche fuori dalla pagina)
Tutti i livelli di una zona hanno le stesse dimensioni in pixel: la pagina li sovrappone esattamente.
"""
import argparse
import json
import math
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sfondo as sfondo_mod  # noqa: E402

ROMA = ZoneInfo("Europe/Rome")
NEROLA = (42.13, 12.77)
AREE = {
    "lazio": {"nome": "Lazio", "riquadro": (11.30, 14.25, 41.05, 43.20), "larghezza": 860, "dettaglio": False},
    "sabina": {"nome": "Sabina", "riquadro": (12.34, 13.20, 41.81, 42.45), "larghezza": 860, "dettaglio": True},
}
# Classi di intensita' della pioggia (mm/h) e colori (scala radar classica: azzurri, blu, verdi, gialli, rossi, viola)
SOGLIE = [0.2, 0.5, 1, 2, 4, 7, 10, 15, 25, 40, 60, 100]
COLORI = ["#a6e3ff", "#5fc8ff", "#1e90ff", "#0a4fd0", "#00a84f", "#7acc00", "#ffe600", "#ffa000", "#ff3c00",
          "#c8001e", "#ff00c8", "#8a00ff"]
ALFA = [150, 185] + [215] * 10
GRIGIO_NODATO = ("#9aa3ad", 90)
CAPOLUOGHI = [("Rieti", 42.404, 12.857), ("Roma", 41.893, 12.483), ("Viterbo", 42.417, 12.105), ("Latina", 41.467, 12.903),
              ("Frosinone", 41.640, 13.343), ("Terni", 42.563, 12.643), ("L'Aquila", 42.350, 13.399),
              ("Grosseto", 42.760, 11.113), ("Perugia", 43.112, 12.389), ("Chieti", 42.351, 14.168),
              ("Pescara", 42.464, 14.214), ("Teramo", 42.659, 13.704), ("Isernia", 41.593, 14.233),
              ("Ascoli Piceno", 42.854, 13.575), ("Caserta", 41.073, 14.333)]


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


# ------------------------------------------------------------------------------------------ geometria delle zone

class Zona:
    def __init__(self, chiave, cfg):
        self.chiave = chiave
        self.nome = cfg["nome"]
        self.lon0, self.lon1, self.lat0, self.lat1 = cfg["riquadro"]
        self.dettaglio = cfg["dettaglio"]
        latc = (self.lat0 + self.lat1) / 2
        self.W = cfg["larghezza"]
        self.H = int(round(self.W * (self.lat1 - self.lat0) / ((self.lon1 - self.lon0) * math.cos(math.radians(latc)))))
        # centri dei pixel (riga 0 = in alto = nord)
        lons = self.lon0 + (np.arange(self.W) + 0.5) * (self.lon1 - self.lon0) / self.W
        lats = self.lat1 - (np.arange(self.H) + 0.5) * (self.lat1 - self.lat0) / self.H
        self.lon, self.lat = np.meshgrid(lons, lats)

    def mappa_da(self, griglia):
        """Coordinate (colonna, riga) nella griglia data di ogni pixel dell'immagine, per cv2.remap."""
        c, r = griglia.indici(self.lon, self.lat)
        return np.asarray(c, dtype="float32"), np.asarray(r, dtype="float32")

    def in_pixel(self, lon, lat):
        x = (np.asarray(lon) - self.lon0) / (self.lon1 - self.lon0) * self.W
        y = (self.lat1 - np.asarray(lat)) / (self.lat1 - self.lat0) * self.H
        return x, y


def ricampiona(campo, mappa, lineare=True):
    """Porta un campo sulla griglia di pixel dell'immagine. NaN dove manca il dato o fuori dalla griglia."""
    a = np.asarray(campo, dtype="float32")
    noto = np.isfinite(a).astype("float32")
    a = np.nan_to_num(a, nan=0.0)
    mx, my = mappa
    v = cv2.remap(a, mx, my, cv2.INTER_LINEAR if lineare else cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    n = cv2.remap(noto, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    v = np.where(n > 0.5, v / np.maximum(n, 1e-6), np.nan)
    return v


def ammorbidisci(campo, sigma):
    """Il satellite ha pixel di 4-5 km ripetuti sulla griglia di 1 km (quadrettoni): una leggera sfumatura li toglie."""
    from scipy import ndimage
    a = np.asarray(campo, dtype="float64")
    noto = np.isfinite(a).astype("float64")
    num = ndimage.gaussian_filter(np.nan_to_num(a), sigma)
    den = ndimage.gaussian_filter(noto, sigma)
    return np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)


# ------------------------------------------------------------------------------------------ livelli trasparenti

def tavolozza_pioggia():
    pal = [0, 0, 0] + list(hex_rgb(GRIGIO_NODATO[0]))
    alfa = [0, GRIGIO_NODATO[1]]
    for c, a in zip(COLORI, ALFA):
        pal += list(hex_rgb(c))
        alfa.append(a)
    return pal, alfa


def salva_pioggia(valori, percorso):
    """Immagine a tavolozza: indice 0 = trasparente (non piove), 1 = grigio (nessun dato), poi le classi."""
    idx = np.zeros(valori.shape, dtype=np.uint8)
    nodato = ~np.isfinite(valori)
    v = np.nan_to_num(valori, nan=0.0)
    classe = np.digitize(v, SOGLIE)        # 0 sotto la prima soglia
    idx[classe > 0] = classe[classe > 0] + 1
    idx[nodato] = 1
    im = Image.fromarray(idx, "P")
    pal, alfa = tavolozza_pioggia()
    im.putpalette(pal + [0] * (768 - len(pal)))
    im.save(percorso, optimize=True, transparency=bytes(alfa))


def a_kelvin(a, diagnosi=None):
    """Il prodotto IR_108 puo' essere in kelvin, in gradi o in conteggi 0-255: lo riconosce dai valori."""
    med = float(np.nanmedian(a)) if np.isfinite(a).any() else 280.0
    if 150 <= med <= 350:
        unita = "K"
        t = a
    elif -110 <= med <= 70 and np.nanmax(a) < 100:
        unita = "gradi C"
        t = a + 273.15
    else:
        unita = "conteggi (scala supposta: 0 = caldo, 255 = freddo)"
        t = 320.0 - np.asarray(a) * (130.0 / 255.0)
    if diagnosi is not None:
        diagnosi["satellite_unita"] = unita
    return t


def colori_ir(tk):
    """Infrarosso: grigi (nubi chiare, suolo scuro) e colori per le cime fredde dei temporali (sotto -40 C)."""
    t = np.nan_to_num(tk, nan=300.0)
    g = np.clip((308.0 - t) / (308.0 - 213.0), 0, 1) ** 0.9
    rgb = np.stack([g, g, g], axis=-1) * 0.92 + 0.04
    # scala colorata da -40 C (233 K) a -80 C (193 K)
    fasce = [(233.0, (0.20, 0.55, 1.0)), (223.0, (0.10, 0.80, 0.30)), (213.0, (1.0, 0.90, 0.10)),
             (203.0, (1.0, 0.35, 0.05)), (193.0, (0.85, 0.0, 0.60)), (180.0, (1.0, 1.0, 1.0))]
    for (ta, ca), (tb, cb) in zip(fasce[:-1], fasce[1:]):
        m = (t <= ta) & (t > tb)
        f = ((ta - t) / (ta - tb))[m][:, None]
        rgb[m] = np.array(ca) * (1 - f) + np.array(cb) * f
    rgb[t <= 180.0] = (1.0, 1.0, 1.0)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------ fondo e confini

def ritaglia_dem(dem, zona, margine=0.02):
    for chiave in ("fine", "largo"):
        if chiave not in dem:
            continue
        z, l0, a0, passo = dem[chiave]
        ny, nx = z.shape
        if l0 <= zona.lon0 - margine and l0 + (nx - 1) * passo >= zona.lon1 + margine and a0 <= zona.lat0 - margine \
                and a0 + (ny - 1) * passo >= zona.lat1 + margine:
            return z, l0, a0, passo
    return None


def rilievo(zona, dem):
    """Rilievo ombreggiato (luce da nord-ovest) con tinta leggera per la quota, mare azzurrino."""
    sel = ritaglia_dem(dem, zona)
    if sel is None:
        base = np.zeros((zona.H, zona.W, 3))
        base[:] = (0.93, 0.95, 0.90)
        return base
    z, l0, a0, passo = sel
    mx = ((zona.lon - l0) / passo).astype("float32")
    my = ((zona.lat - a0) / passo).astype("float32")
    q = cv2.remap(z.astype("float32"), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype("float64")
    dy = (zona.lat1 - zona.lat0) / zona.H * 111000.0
    dx = (zona.lon1 - zona.lon0) / zona.W * 111000.0 * math.cos(math.radians(42.1))
    gy = -np.gradient(q, axis=0) / dy      # righe verso sud: il gradiente verso nord ha il segno opposto
    gx = np.gradient(q, axis=1) / dx
    ve = 1.6
    pend = np.arctan(ve * np.hypot(gx, gy))
    esp = np.arctan2(gy, -gx)
    zen, az = np.radians(45.0), np.radians(315.0)
    hs = np.clip(np.cos(zen) * np.cos(pend) + np.sin(zen) * np.sin(pend) * np.cos(az - esp), 0, 1)
    piano = np.cos(zen)
    quota = np.clip(q / 1600.0, 0, 1)
    basso = np.array([0.93, 0.95, 0.90])
    alto = np.array([0.84, 0.78, 0.68])
    base = basso * (1 - quota[..., None]) + alto * quota[..., None]
    base = np.where((q <= 0)[..., None], np.array([0.80, 0.88, 0.96]), base)
    ombra = np.clip((piano - hs) / piano, 0, 1) * 0.42
    luce = np.clip((hs - piano) / (1 - piano), 0, 1) * 0.22
    base = base * (1 - ombra[..., None]) + luce[..., None] * (1 - base)
    return np.clip(base, 0, 1)


def figura(zona):
    fig = plt.figure(figsize=(zona.W / 100, zona.H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, zona.W)
    ax.set_ylim(zona.H, 0)
    ax.axis("off")
    return fig, ax


def linee_px(zona, linee):
    out = []
    for ln in linee:
        a = np.asarray(ln, dtype="float64")
        if a.ndim != 2 or len(a) < 2:
            continue
        x, y = zona.in_pixel(a[:, 0], a[:, 1])
        if x.max() < -20 or x.min() > zona.W + 20 or y.max() < -20 or y.min() > zona.H + 20:
            continue
        out.append(np.column_stack([x, y]))
    return out


def candidati(zona, sf):
    elenco = [("Nerola", NEROLA[0], NEROLA[1], True)]
    dentro = lambda la, lo: zona.lat0 + 0.01 < la < zona.lat1 - 0.01 and zona.lon0 + 0.015 < lo < zona.lon1 - 0.015  # noqa: E731
    if zona.dettaglio:
        paesi = [p for p in sf.get("paesi", []) if dentro(p["lat"], p["lon"]) and p["nome"] != "Nerola" and len(p["nome"]) <= 16]
        grandi = sorted([p for p in paesi if p["pop"] >= 15000], key=lambda p: -p["pop"])
        altri = sorted([p for p in paesi if p["pop"] < 15000 and p["sede"]],
                       key=lambda p: (p["lat"] - NEROLA[0]) ** 2 + ((p["lon"] - NEROLA[1]) * 0.74) ** 2)
        elenco += [(p["nome"], p["lat"], p["lon"], False) for p in grandi + altri]
    else:
        elenco += [(n, la, lo, False) for n, la, lo in CAPOLUOGHI if dentro(la, lo)]
    return elenco


def confini(zona, sf, percorso, massimo=40):
    fig, ax = figura(zona)
    fig.patch.set_alpha(0)
    stili = [("comuni", dict(colors="#3d3d3d", linewidths=0.35, alpha=0.45)) if zona.dettaglio else None,
             ("province", dict(colors="#1a1a1a", linewidths=0.9, linestyles=(0, (5, 3)), alpha=0.85)),
             ("regioni", dict(colors="#1a1a1a", linewidths=1.5, alpha=0.9)),
             ("laghi", dict(colors="#2a6cc7", linewidths=0.8)),
             ("fiumi", dict(colors="#2a6cc7", linewidths=0.7, alpha=0.8)),
             ("costa", dict(colors="#1a1a1a", linewidths=1.0))]
    for s in stili:
        if s and sf.get(s[0]):
            ax.add_collection(LineCollection(linee_px(zona, sf[s[0]]), zorder=3, **s[1]))
    occupati = []
    scritti = 0
    for nome, la, lo, grande in candidati(zona, sf):
        if scritti >= massimo:
            break
        x, y = (float(v) for v in zona.in_pixel(lo, la))
        fs = 12.5 if grande else (8.5 if zona.dettaglio else 10)
        w = len(nome) * fs * 0.62 + 6
        h = fs * 1.35 + 2
        posto = None
        for dx, dy, ha, va in ((7, -4, "left", "bottom"), (7, 4, "left", "top"), (-7, -4, "right", "bottom"), (-7, 4, "right", "top")):
            x0 = x + dx if ha == "left" else x + dx - w
            y0 = y + dy - h if va == "bottom" else y + dy
            box = (x0, y0, x0 + w, y0 + h)
            if box[0] < 2 or box[2] > zona.W - 2 or box[1] < 2 or box[3] > zona.H - 2:
                continue
            if any(not (box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3]) for o in occupati):
                continue
            posto = (dx, dy, ha, va, box)
            break
        if posto is None:
            continue
        dx, dy, ha, va, box = posto
        occupati.append(box)
        occupati.append((x - 5, y - 5, x + 5, y + 5))
        ax.plot([x], [y], marker="*" if grande else "o", color="#d6341f" if grande else "#111111",
                markersize=15 if grande else 4.5, markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        t = ax.text(x + dx, y + dy, nome, ha=ha, va=va, fontsize=fs * 0.75, fontweight="bold", color="#111111", zorder=6)
        t.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])
        scritti += 1
    ax.text(zona.W - 8, zona.H - 8, "www.meteonerola.it", ha="right", va="bottom", fontsize=10, fontweight="bold", color="#111111",
            zorder=7, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#333333", linewidth=0.8, alpha=0.9))
    fig.savefig(percorso, dpi=100, transparent=True)
    plt.close(fig)


def fondo(zona, dem, percorso):
    rgb = (rilievo(zona, dem) * 255).astype(np.uint8)
    Image.fromarray(rgb, "RGB").quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(percorso, optimize=True)


# ------------------------------------------------------------------------------------------ immagine completa

def ora_locale(iso):
    t = datetime.strptime(iso, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    return t.astimezone(ROMA)


def composito(cartella, file_pioggia, titolo, sottotitolo, percorso):
    """fondo + pioggia + confini + una fascia con titolo e legenda: immagine unica pronta da pubblicare."""
    base = Image.open(cartella / "fondo.png").convert("RGBA")
    if file_pioggia:
        base.alpha_composite(Image.open(file_pioggia).convert("RGBA"))
    base.alpha_composite(Image.open(cartella / "confini.png").convert("RGBA"))
    W, H = base.size
    alto, basso = 58, 46
    tela = Image.new("RGB", (W, H + alto + basso), "white")
    tela.paste(base.convert("RGB"), (0, alto))
    d = ImageDraw.Draw(tela)
    try:
        grassetto = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        normale = ImageFont.truetype("DejaVuSans.ttf", 15)
        piccolo = ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        grassetto = normale = piccolo = ImageFont.load_default()
    d.text((12, 6), titolo, fill="#111111", font=grassetto)
    d.text((12, 34), sottotitolo, fill="#333333", font=normale)
    # legenda
    x = 12
    y = H + alto + 8
    passo = (W - 24 - 60) / len(COLORI)
    d.text((x, y + 2), "mm/h", fill="#333333", font=piccolo)
    x += 40
    for i, c in enumerate(COLORI):
        d.rectangle([x + i * passo, y, x + (i + 1) * passo - 2, y + 14], fill=c)
        d.text((x + i * passo, y + 17), f"{SOGLIE[i]:g}", fill="#333333", font=piccolo)
    d.text((W - 12, H + alto + basso - 4), "Radar: Protezione Civile (DPC)", fill="#6b7480", font=piccolo, anchor="rb")
    tela.quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(percorso, optimize=True)


# ------------------------------------------------------------------------------------------ tutto

def nome_file(iso):
    return datetime.strptime(iso, "%Y-%m-%dT%H:%MZ").strftime("%Y%m%d_%H%M") + ".png"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dati", required=True)
    ap.add_argument("--nowcast", required=True)
    ap.add_argument("--sfondo", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    with open(f"{a.dati}/radar.pkl", "rb") as f:
        radar = pickle.load(f)
    with open(a.nowcast, "rb") as f:
        nc = pickle.load(f)
    sat = None
    if Path(f"{a.dati}/satellite.pkl").exists():
        with open(f"{a.dati}/satellite.pkl", "rb") as f:
            sat = pickle.load(f)
    sf = sfondo_mod.carica(a.sfondo) if a.sfondo else sfondo_mod.carica("/inesistente")
    dem = sfondo_mod.carica_dem(a.sfondo) if a.sfondo else {}
    diagnosi = {}
    indice = {"aggiornato": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "t0": nc["tempi"][0],
              "info": nc["info"], "soglie": SOGLIE, "colori": COLORI, "zone": {}}
    t0 = nc["tempi"][0]
    for chiave, cfg in AREE.items():
        z = Zona(chiave, cfg)
        cart = out / chiave
        for sotto in ("radar", "previsione", "satellite"):
            (cart / sotto).mkdir(parents=True, exist_ok=True)
        fondo(z, dem, cart / "fondo.png")
        confini(z, sf, cart / "confini.png", massimo=45 if z.dettaglio else 20)
        voce = {"nome": z.nome, "larghezza": z.W, "altezza": z.H, "riquadro": [z.lon0, z.lon1, z.lat0, z.lat1],
                "radar": [], "previsione": [], "satellite": []}
        mappa = z.mappa_da(radar["griglia"])
        for t, campo in zip(radar["tempi"], radar["campi"]):
            if t > t0:
                continue
            f = nome_file(t)
            salva_pioggia(ricampiona(campo, mappa), cart / "radar" / f)
            voce["radar"].append({"file": f"radar/{f}", "ora": t})
        mappa_nc = z.mappa_da(nc["griglia"])
        for k, (t, campo) in enumerate(zip(nc["tempi"], nc["campi"])):
            if k == 0:
                continue
            f = nome_file(t)
            salva_pioggia(ricampiona(campo, mappa_nc), cart / "previsione" / f)
            voce["previsione"].append({"file": f"previsione/{f}", "ora": t, "minuti": k * 10, "peso_radar": nc["pesi"][k]})
        if sat is not None:
            mappa_s = z.mappa_da(sat["griglia"])
            for t, campo in zip(sat["tempi"], sat["campi"]):
                tk = a_kelvin(ricampiona(ammorbidisci(campo, 1.8), mappa_s), diagnosi)
                f = nome_file(t)
                Image.fromarray(colori_ir(tk), "RGB").quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE) \
                    .save(cart / "satellite" / f, optimize=True)
                voce["satellite"].append({"file": f"satellite/{f}", "ora": t})
        ultimo = voce["radar"][-1] if voce["radar"] else None
        loc = ora_locale(t0)
        gg = ["lunedi'", "martedi'", "mercoledi'", "giovedi'", "venerdi'", "sabato", "domenica"][loc.weekday()]
        composito(cart, cart / ultimo["file"] if ultimo else None, f"Radar pioggia - {z.nome}",
                  f"{gg} {loc:%d/%m} ore {loc:%H:%M} (ora italiana)", cart / "ultimo.png")
        indice["zone"][chiave] = voce
        print(f"{z.nome}: {z.W}x{z.H} px, radar {len(voce['radar'])}, previsione {len(voce['previsione'])}, "
              f"satellite {len(voce['satellite'])}")
    indice["diagnosi_disegno"] = diagnosi
    (out / "indice.json").write_text(json.dumps(indice, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
