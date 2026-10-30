"""Evoluzione guidata dal WRF: si parte dalla realta' (ultimo scatto del radar o del satellite) e la si fa evolvere
per 6 ore seguendo il modello WRF.

Ad ogni passo di 10 minuti, per ogni punto:
 1. MOVIMENTO: il punto "arriva" da dove lo porta il vento che spinge le nubi e i temporali (vento medio tra 850 e
    500 hPa del WRF, circa 1.500-5.500 m). Nei primissimi minuti si usa il movimento misurato sul radar adesso, poi
    si passa al vento del WRF (che sa come cambiera': un fronte che arriva, un cambio di direzione).
 2. EVOLUZIONE: lungo questo percorso si applica alla pioggia vera la stessa variazione che ha la pioggia nel WRF
    (in decibel, cioe' in proporzione): se nel WRF la pioggia cresce del doppio, cresce del doppio anche quella vera;
    se nel WRF si esaurisce, si esaurisce; se nel WRF nasce una cella dove prima non pioveva, nasce anche qui.
    La pioggia del WRF contiene gia' l'effetto di tutto cio' che il modello calcola (energia per i temporali,
    umidita', moti verticali, convergenza, rilievo): e' cosi' che "tutti i parametri" vengono applicati alla realta'.
 3. Il WRF usato e' quello gia' corretto adesso con radar e satellite (posizione e intensita', vedi fusione.py),
    cosi' la variazione viene presa nel posto giusto.
Nelle prime ore quindi la previsione e' la realta' di adesso con l'evoluzione del WRF; con il passare delle ore
diventa sempre piu' simile al WRF corretto, in modo naturale (non con pesi decisi a mano).
"""
import math

import cv2
import numpy as np
from scipy import ndimage

MAX_TENDENZA = 30.0       # dB: al massimo 1000 volte in piu' o in meno in un passo (evita valori assurdi)


def _remap(campo, px, py, bordo):
    return cv2.remap(np.asarray(campo, dtype="float32"), px, py, cv2.INTER_LINEAR, borderMode=bordo)


def evolvi(x0, noto0, campi_wrf, mov_oss, mov_wrf, passi, minuti_passo=10, sigma=5.0, minimo=None, tau_mov=30.0,
           massimo=None):
    """x0: campo osservato adesso (gia' nella scala in cui la variazione e' additiva: decibel per la pioggia).
    noto0: dove x0 e' noto. campi_wrf[k]: campo del WRF corretto al passo k (stessa scala), k=0..passi (None se manca).
    mov_oss: (u, v) movimento osservato in pixel per passo. mov_wrf(k): (u, v) del WRF al passo k, o None.
    minimo: valore che vuol dire "niente" (dB di non pioggia), per tenere asciutto cio' che resta asciutto.
    Ritorna [campo al passo k], [dove e' noto], e un dizionario di informazioni."""
    ny, nx = x0.shape
    xx, yy = np.meshgrid(np.arange(nx, dtype="float32"), np.arange(ny, dtype="float32"))
    px, py = xx.copy(), yy.copy()
    X0 = np.where(noto0, x0, minimo if minimo is not None else np.nanmin(x0)).astype("float32")
    W0 = campi_wrf.get(0)
    W0s = None if W0 is None else ndimage.gaussian_filter(np.nan_to_num(W0, nan=minimo), sigma).astype("float32")
    uo, vo = (a.astype("float32") for a in mov_oss)
    uscite, validi, info = [], [], {"passi_con_wrf": 0, "passi_senza_wrf": 0}
    for k in range(1, passi + 1):
        minuti = k * minuti_passo
        a = math.exp(-minuti / tau_mov)
        mw = mov_wrf(k) if mov_wrf else None
        if mw is None:
            u, v = uo, vo
        else:
            u = a * uo + (1 - a) * mw[0]
            v = a * vo + (1 - a) * mw[1]
        du = _remap(u, px, py, cv2.BORDER_REPLICATE)
        dv = _remap(v, px, py, cv2.BORDER_REPLICATE)
        px = px - du
        py = py - dv
        xdep = _remap(X0, px, py, cv2.BORDER_REPLICATE)
        ndep = _remap(noto0.astype("float32"), px, py, cv2.BORDER_CONSTANT) > 0.5
        Wk = campi_wrf.get(k)
        if W0s is not None and Wk is not None:
            Wks = ndimage.gaussian_filter(np.nan_to_num(Wk, nan=minimo), sigma).astype("float32")
            w0dep = _remap(W0s, px, py, cv2.BORDER_REPLICATE)
            tend = np.clip(Wks - w0dep, -MAX_TENDENZA, MAX_TENDENZA)
            x = xdep + tend
            # dove la realta' arriva da fuori dell'area vista, c'e' solo il WRF corretto
            x = np.where(ndep, x, Wks)
            info["passi_con_wrf"] += 1
        else:
            x = xdep
            info["passi_senza_wrf"] += 1
        if minimo is not None:
            x = np.maximum(x, minimo)
        if massimo is not None:
            x = np.minimum(x, massimo)
        uscite.append(x)
        validi.append(ndep | (Wk is not None and W0s is not None))
    return uscite, validi, info


def vento_in_pixel(u_ms, v_ms, km_pixel, minuti_passo=10):
    """Vento (m/s, verso est e verso nord) -> pixel per passo sulla griglia (colonne verso est, righe verso SUD)."""
    f = minuti_passo * 60.0 / (km_pixel * 1000.0)
    return u_ms * f, -v_ms * f
