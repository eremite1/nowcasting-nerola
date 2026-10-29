"""Satellite "nubi su mappa vera": le nubi vengono separate dal suolo e disegnate, bianche e con la loro ombra,
sopra una mappa a colori naturali (rilievo, pianure, montagne, mare). Di giorno il dettaglio viene dal visibile
(500 m), di notte dall'infrarosso (1 km). Cosi' l'immagine e' luminosa e leggibile a qualsiasi ora.

Come si separano le nubi dal suolo (tutto misurato, niente soglie fisse sul colore):
  - INFRAROSSO: una nube e' piu' fredda del suolo sotto di lei. La temperatura del suolo sereno si stima come la
    temperatura piu' alta nei dintorni (entro circa 40 km); quanto piu' un punto e' freddo rispetto a questa,
    tanto piu' e' nuvoloso.
  - VISIBILE: una nube riflette piu' luce del suolo. La luminosita' del suolo sereno si stima come la piu' bassa nei
    dintorni (entro circa 25 km); quanto piu' un punto e' luminoso rispetto a questa, tanto piu' e' nuvoloso.
  Di giorno contano soprattutto il visibile (vede anche nebbie e nubi basse), di notte solo l'infrarosso.
"""
import math

import numpy as np
from scipy import ndimage


# ------------------------------------------------------------------------------------------------- mappa di base

def mappa_naturale(quote, mare, passo_km):
    """Colori naturali per quota (pianure verdi, colline verde-oliva, montagne marroni, cime grigio chiaro),
    mare blu (piu' chiaro vicino alla costa), con ombreggiatura del rilievo (luce da nord-ovest)."""
    q = np.asarray(quote, dtype="float64")
    tappe = [(0, (0.42, 0.55, 0.33)), (200, (0.50, 0.60, 0.36)), (500, (0.60, 0.62, 0.40)), (900, (0.62, 0.56, 0.42)),
             (1400, (0.60, 0.52, 0.44)), (1900, (0.72, 0.70, 0.67)), (2500, (0.90, 0.90, 0.90))]
    rgb = np.zeros(q.shape + (3,))
    qq = np.clip(q, 0, 2500)
    for (a, ca), (b, cb) in zip(tappe[:-1], tappe[1:]):
        m = (qq >= a) & (qq <= b)
        f = ((qq - a) / (b - a))[m][:, None]
        rgb[m] = np.array(ca) * (1 - f) + np.array(cb) * f
    # mare: blu profondo al largo, piu' chiaro sotto costa
    dist = ndimage.distance_transform_edt(mare) * passo_km
    vicino = np.clip(1 - dist / 25.0, 0, 1)[..., None]
    blu = np.array([0.09, 0.24, 0.42]) * (1 - vicino) + np.array([0.16, 0.38, 0.55]) * vicino
    rgb = np.where(mare[..., None], blu, rgb)
    # ombreggiatura
    gy, gx = np.gradient(np.where(mare, 0, q), passo_km * 1000.0)
    gy = -gy   # righe verso sud
    pend = np.arctan(2.0 * np.hypot(gx, gy))
    esp = np.arctan2(gy, -gx)
    zen, az = np.radians(45.0), np.radians(315.0)
    hs = np.clip(np.cos(zen) * np.cos(pend) + np.sin(zen) * np.sin(pend) * np.cos(az - esp), 0, 1)
    luce = (hs / np.cos(zen))[..., None]
    rgb = np.where(mare[..., None], rgb, np.clip(rgb * (0.55 + 0.45 * luce), 0, 1))
    return rgb


# ------------------------------------------------------------------------------------------------- nubi

def _filtro_locale(a, km, passo_km, funzione):
    n = max(3, int(round(km / passo_km)) | 1)
    # si lavora su una versione rimpicciolita (piu' veloce), poi si riporta alla dimensione originale
    f = max(1, n // 9)
    piccolo = a[::f, ::f]
    r = funzione(piccolo, size=max(3, n // f))
    r = ndimage.gaussian_filter(r, max(1.0, n / f / 3))
    return ndimage.zoom(r, (a.shape[0] / r.shape[0], a.shape[1] / r.shape[1]), order=1)[:a.shape[0], :a.shape[1]]


def copertura(T, R, giorno, passo_km):
    """Opacita' delle nubi (0 = sereno, 1 = nube spessa) e luminosita' delle nubi."""
    T = np.where(np.isfinite(T), T, np.nanmedian(T) if np.isfinite(T).any() else 10.0)
    t_suolo = _filtro_locale(ndimage.gaussian_filter(T, 2), 40, passo_km, ndimage.maximum_filter)
    c_ir = np.clip((t_suolo - T - 4.0) / 30.0, 0, 1) ** 0.8
    if R is None or giorno.max() <= 0:
        alfa = c_ir
        lum = 0.80 + 0.20 * c_ir
        return alfa, lum
    R = np.nan_to_num(R)
    r_suolo = _filtro_locale(ndimage.gaussian_filter(R, 2), 25, passo_km, ndimage.minimum_filter)
    c_vis = np.clip((R - r_suolo - 0.04) / 0.32, 0, 1) ** 0.85
    di_giorno = np.maximum(c_vis, 0.55 * c_ir)
    alfa = giorno * di_giorno + (1 - giorno) * c_ir
    lum_giorno = np.clip(0.62 + 0.55 * R, 0, 1)            # la tessitura delle nubi viene dal visibile
    lum = giorno * lum_giorno + (1 - giorno) * (0.80 + 0.20 * c_ir)
    return alfa, lum


def componi(base, alfa, lum, T=None, ombra_px=(5, 7)):
    """Mappa + ombra delle nubi + nubi."""
    # ombra: le nubi spostate verso sud-est e sfumate scuriscono il suolo (piu' alta la nube, piu' lontana l'ombra)
    dy, dx = ombra_px
    om = ndimage.shift(alfa, (dy, dx), order=1, mode="nearest")
    om = ndimage.gaussian_filter(om, 3)
    out = base * (1 - 0.40 * om[..., None])
    # nubi bianche, appena azzurrate quelle molto fredde (alte)
    bianco = np.stack([lum, lum, lum], axis=-1)
    if T is not None:
        freddo = np.clip((-25.0 - np.nan_to_num(T, nan=0.0)) / 40.0, 0, 1)[..., None]
        bianco = bianco * (1 - 0.10 * freddo) + np.array([0.86, 0.92, 1.0]) * 0.10 * freddo
    a = np.clip(alfa, 0, 1)[..., None]
    return np.clip(out * (1 - a) + bianco * a, 0, 1)


def dem_su_zona(dem, zona):
    """Quote del rilievo sui pixel della zona (dalla griglia piu' fine che la copre)."""
    import cv2
    for chiave in ("fine", "largo"):
        if chiave not in dem:
            continue
        z, l0, a0, passo = dem[chiave]
        ny, nx = z.shape
        if l0 <= zona.lon0 and l0 + (nx - 1) * passo >= zona.lon1 and a0 <= zona.lat0 and a0 + (ny - 1) * passo >= zona.lat1:
            mx = ((zona.lon - l0) / passo).astype("float32")
            my = ((zona.lat - a0) / passo).astype("float32")
            return cv2.remap(z.astype("float32"), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype("float64")
    return None


def passo_km(zona):
    return (zona.lon1 - zona.lon0) * 111.0 * math.cos(math.radians(42.1)) / zona.W
