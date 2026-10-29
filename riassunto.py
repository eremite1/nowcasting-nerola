"""Il riassunto "Nerola adesso", i fulmini e la griglia per "tocca un punto della mappa".

  - fulmini: dalle immagini del Lightning Imager (MTG) si ricavano le posizioni dei lampi (centro di ogni macchia)
  - Nerola: pioggia ora e nelle prossime 6 ore, fulmine piu' vicino, nuvolosita' ora e piu' avanti, in frasi semplici
  - griglia.bin: pioggia (mm/h x 10) e nuvolosita' (%) ogni 10 minuti da adesso a +6 ore, su una griglia di circa 4 km
"""
import math
from datetime import timedelta

import numpy as np
from scipy import ndimage

NEROLA = (42.1607, 12.7856)       # lat, lon (centro del paese)
PUNTI = ["nord", "nord-est", "est", "sud-est", "sud", "sud-ovest", "ovest", "nord-ovest"]
# griglia per "tocca la mappa": lon 11,30-14,25, lat 41,05-43,20, passo 0,04 gradi (circa 3,3 x 4,4 km)
G_LON0, G_LAT1, G_PASSO = 11.30, 43.20, 0.04
G_NX = int(round((14.25 - 11.30) / G_PASSO))
G_NY = int(round((43.20 - 41.05) / G_PASSO))


def km(la1, lo1, la2, lo2):
    dy = (la2 - la1) * 111.2
    dx = (lo2 - lo1) * 111.2 * math.cos(math.radians((la1 + la2) / 2))
    return math.hypot(dx, dy)


def verso(la1, lo1, la2, lo2):
    """Da che parte sta il punto 2 rispetto al punto 1 (8 direzioni)."""
    dy = la2 - la1
    dx = (lo2 - lo1) * math.cos(math.radians(la1))
    ang = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    return PUNTI[int((ang + 22.5) // 45) % 8]


def fulmini_da_immagine(rgba, bbox, larghezza, altezza):
    """Posizioni dei lampi: ogni macchia colorata dell'immagine LI e' un gruppo di lampi."""
    a = rgba[..., 3] > 0.3
    if not a.any():
        return []
    etichette, n = ndimage.label(a)
    out = []
    for k, (r, c) in enumerate(ndimage.center_of_mass(a, etichette, range(1, n + 1)), start=1):
        area = int((etichette == k).sum())
        lon = bbox[0] + (c + 0.5) / larghezza * (bbox[1] - bbox[0])
        lat = bbox[3] - (r + 0.5) / altezza * (bbox[3] - bbox[2])
        out.append({"lat": round(float(lat), 4), "lon": round(float(lon), 4), "pixel": area})
    return out


def intensita(mmh):
    if mmh < 1:
        return "debole"
    if mmh < 4:
        return "moderata"
    if mmh < 15:
        return "forte"
    return "molto forte"


def cielo(c):
    if c < 0.12:
        return "sereno"
    if c < 0.35:
        return "poco nuvoloso"
    if c < 0.65:
        return "nuvoloso"
    return "coperto"


def ora_it(t):
    from zoneinfo import ZoneInfo
    return t.astimezone(ZoneInfo("Europe/Rome")).strftime("%H:%M")


def riassunto_nerola(t0, pioggia_nerola, pioggia_vicina, nubi_nerola, fulmini, mov_pioggia, mov_nubi, tipo_nerola=None, grandine=None):
    """pioggia_nerola: mm/h ogni 10 minuti da t0 (indice 0 = osservato adesso) a +6 h (max entro 3 km da Nerola);
    pioggia_vicina: (distanza km, direzione, mm/h) della pioggia piu' vicina adesso entro 40 km, o None;
    nubi_nerola: lista (istante, copertura 0-1) osservata e prevista; fulmini: lista con 'ora','lat','lon'."""
    frasi = {}
    p = np.asarray(pioggia_nerola)
    tn = list(tipo_nerola or [])
    nome_tipo = lambda k: ("Neve" if tn[k] == 2 else "Pioggia mista a neve" if tn[k] == 1 else "Pioggia") if k < len(tn) else "Pioggia"  # noqa: E731
    piove = p >= 0.2
    if piove[0]:
        fine = next((k for k in range(1, len(p)) if not piove[k]), None)
        verbo = "Sta nevicando" if tn and tn[0] == 2 else "Pioggia mista a neve" if tn and tn[0] == 1 else "Sta piovendo"
        frasi["pioggia"] = f"{verbo} a Nerola ({intensita(p[0])})" + (
            f": dovrebbe smettere verso le {ora_it(t0 + timedelta(minutes=10 * fine))}" if fine else ", e continuera' a lungo")
        frasi["livello_pioggia"] = "ora"
    else:
        inizio = next((k for k in range(1, len(p)) if piove[k]), None)
        if inizio is not None:
            picco = float(p[inizio:inizio + 6].max())
            minuti = 10 * inizio
            quando = f"tra circa {minuti} minuti" if minuti < 90 else f"verso le {ora_it(t0 + timedelta(minutes=minuti))}"
            frasi["pioggia"] = f"{nome_tipo(inizio)} in arrivo a Nerola {quando} ({intensita(picco)})"
            frasi["livello_pioggia"] = "presto" if minuti <= 60 else "dopo"
        else:
            frasi["pioggia"] = "Nessuna pioggia prevista a Nerola nelle prossime 6 ore"
            frasi["livello_pioggia"] = "no"
            if pioggia_vicina:
                d, dove, mm = pioggia_vicina
                frasi["pioggia"] += f" (ora piove a {round(d)} km verso {dove})"
    # grandine stimata dal radar (ultimi 30 minuti, entro 30 km)
    if grandine:
        from datetime import datetime, timezone
        vicine = []
        for g in grandine:
            tg = datetime.strptime(g["ora"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
            d = km(*NEROLA, g["lat"], g["lon"])
            if (t0 - tg).total_seconds() <= 1800 and d <= 30:
                vicine.append((d, g))
        if vicine:
            d, g = min(vicine, key=lambda x: x[0])
            frasi["grandine"] = f"Possibile grandine a {round(d)} km verso {verso(*NEROLA, g['lat'], g['lon'])} (probabilita' {g['poh']}%)"
    # fulmini (ultimi 30 minuti)
    recenti = [f for f in fulmini if (t0 - f["t"]).total_seconds() <= 1800]
    if recenti:
        vicino = min(recenti, key=lambda f: km(*NEROLA, f["lat"], f["lon"]))
        d = km(*NEROLA, vicino["lat"], vicino["lon"])
        quanto = max(0, round((t0 - vicino["t"]).total_seconds() / 60))
        frasi["fulmini"] = f"Fulmine piu' vicino: {round(d)} km verso {verso(*NEROLA, vicino['lat'], vicino['lon'])}, {quanto} minuti fa"
        frasi["livello_fulmini"] = "vicino" if d < 15 else ("medio" if d < 40 else "lontano")
    else:
        frasi["fulmini"] = "Nessun fulmine negli ultimi 30 minuti"
        frasi["livello_fulmini"] = "no"
    # cielo
    if nubi_nerola:
        ora_c = nubi_nerola[0][1]
        dopo = [c for t, c in nubi_nerola if (t - nubi_nerola[0][0]).total_seconds() >= 2 * 3600]
        s = f"Cielo {cielo(ora_c)}"
        if dopo:
            c2 = dopo[0]
            if cielo(c2) != cielo(ora_c):
                s += f", {cielo(c2)} tra un paio d'ore"
        if mov_nubi and mov_nubi.get("direzione") is not None and mov_nubi.get("velocita_kmh", 0) > 5:
            s += f"; nubi da {PUNTI[int((mov_nubi['direzione'] + 22.5) // 45) % 8]} a {round(mov_nubi['velocita_kmh'])} km/h"
        frasi["cielo"] = s
        frasi["copertura"] = round(float(ora_c), 2)
    return frasi


def griglia_coarse(campo_fn):
    """Valori di un campo nei centri della griglia per "tocca la mappa". campo_fn(lon, lat) -> matrice."""
    lon = G_LON0 + (np.arange(G_NX) + 0.5) * G_PASSO
    lat = G_LAT1 - (np.arange(G_NY) + 0.5) * G_PASSO
    LO, LA = np.meshgrid(lon, lat)
    return campo_fn(LO, LA)
