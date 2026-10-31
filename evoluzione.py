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
MOVIMENTO LISCIO: il campo di movimento deve essere liscio (le nubi e la pioggia si spostano come un insieme, non
punto per punto), altrimenti sommando 36 passi le forme si stirano in strisce. Si usa il vento in quota del WRF
(campo gia' liscio) o il movimento misurato su larga scala.
QUANTO PESA IL WRF: la variazione presa dal WRF viene moltiplicata per l'accordo misurato adesso tra WRF corretto e
osservazioni (correlazione, da 0 a 1): un WRF che oggi non ci prende non deforma la realta'.
"""
import math

import cv2
import numpy as np
from scipy import ndimage

MAX_TENDENZA = 30.0       # dB: al massimo 1000 volte in piu' o in meno in un passo (evita valori assurdi)


def _remap(campo, px, py, bordo):
    return cv2.remap(np.asarray(campo, dtype="float32"), px, py, cv2.INTER_LINEAR, borderMode=bordo)


def evolvi(x0, noto0, campi_wrf, mov_oss, mov_wrf, passi, minuti_passo=10, sigma=5.0, minimo=None, tau_mov=30.0,
           massimo=None, peso_tendenza=1.0, max_tendenza=MAX_TENDENZA, dentro=None,
           movimenti=None, mov_fisso=None, liscia_mov=0.0):
    """x0: campo osservato adesso (gia' nella scala in cui la variazione e' additiva: decibel per la pioggia).
    noto0: dove x0 e' noto. campi_wrf[k]: campo del WRF corretto al passo k (stessa scala), k=0..passi (None se manca).
    mov_oss: (u, v) movimento osservato in pixel per passo. mov_wrf(k): (u, v) del WRF al passo k, o None.
    minimo: valore che vuol dire "niente" (dB di non pioggia), per tenere asciutto cio' che resta asciutto.
    Ritorna [campo al passo k], [dove e' noto], e un dizionario di informazioni."""
    ny, nx = x0.shape
    xx, yy = np.meshgrid(np.arange(nx, dtype="float32"), np.arange(ny, dtype="float32"))
    px, py = xx.copy(), yy.copy()
    X0 = np.where(noto0, x0, minimo if minimo is not None else np.nanmin(x0)).astype("float32")

    def liscia(W):
        """Lisciatura che rispetta i bordi del WRF: fuori dal suo territorio resta NaN (non diventa 'niente')."""
        v = np.isfinite(W).astype("float32")
        num = ndimage.gaussian_filter(np.nan_to_num(W).astype("float32"), sigma)
        den = ndimage.gaussian_filter(v, sigma)
        return np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan).astype("float32"), v

    W0 = campi_wrf.get(0)
    W0s = None if W0 is None else liscia(W0)[0]
    uo, vo = (a.astype("float32") for a in mov_oss)
    uscite, validi, info = [], [], {"passi_con_wrf": 0, "passi_senza_wrf": 0}
    for k in range(1, passi + 1):
        minuti = k * minuti_passo
        a = math.exp(-minuti / tau_mov)
        Wk = campi_wrf.get(k)
        Wks, vk = (None, None) if Wk is None else liscia(Wk)
        mw = mov_wrf(k) if (mov_wrf and mov_fisso is None) else None
        if mov_fisso is not None:
            # movimento imposto da fuori (stesso movimento della zona grande: nessuno "strappo" dove si uniscono)
            u, v = (np.asarray(a_, "float32") for a_ in mov_fisso(k))
        elif mw is None:
            u, v = uo, vo
        else:
            # dove c'e' il WRF il suo movimento, fuori dal suo territorio quello misurato (passaggio sfumato)
            pw = np.ones_like(uo) if vk is None else np.clip(ndimage.gaussian_filter(vk, 20) * 2 - 1, 0, 1)
            mwu = np.where(np.isfinite(mw[0]), mw[0], uo)
            mwv = np.where(np.isfinite(mw[1]), mw[1], vo)
            u = a * uo + (1 - a) * (pw * mwu + (1 - pw) * uo)
            v = a * vo + (1 - a) * (pw * mwv + (1 - pw) * vo)
        if liscia_mov > 0 and mov_fisso is None:
            # movimento senza strappi: dove si passa dal movimento del WRF a quello misurato le nubi non si stirano
            u = ndimage.gaussian_filter(u, liscia_mov)
            v = ndimage.gaussian_filter(v, liscia_mov)
        if movimenti is not None:
            movimenti.append((u, v))
        du = _remap(u, px, py, cv2.BORDER_REPLICATE)
        dv = _remap(v, px, py, cv2.BORDER_REPLICATE)
        px = px - du
        py = py - dv
        # la realta' che arriva da fuori dell'immagine: si prende il bordo (non il vuoto)
        xdep = _remap(X0, px, py, cv2.BORDER_REPLICATE)
        ndep = _remap(noto0.astype("float32"), px, py, cv2.BORDER_CONSTANT) > 0.5
        if W0s is not None and Wks is not None:
            w0dep = cv2.remap(W0s, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
            ok = np.isfinite(Wks) & np.isfinite(w0dep)
            # la variazione del WRF conta quanto il WRF sta azzeccando adesso (peso_tendenza tra 0 e 1);
            # fuori dal territorio del WRF nessuna variazione inventata
            tend = np.where(ok, np.clip(np.nan_to_num(Wks) - np.nan_to_num(w0dep), -max_tendenza, max_tendenza), 0.0) * peso_tendenza
            # vicino al bordo del WRF la sua influenza sfuma (niente "cornice" visibile)
            tend = tend * np.clip(ndimage.gaussian_filter(ok.astype("float32"), 20) * 2 - 1, 0, 1)
            x = xdep + tend
            if minimo is not None:
                fuori = np.where(np.isfinite(Wks), minimo + (np.nan_to_num(Wks) - minimo) * peso_tendenza, xdep)
            else:
                fuori = np.where(np.isfinite(Wks), Wks, xdep)
            x = np.where(ndep, x, fuori)
            info["passi_con_wrf"] += 1
        else:
            x = xdep
            info["passi_senza_wrf"] += 1
        if minimo is not None:
            x = np.maximum(x, minimo)
        if massimo is not None:
            x = np.minimum(x, massimo)
        uscite.append(x)
        if dentro is not None:
            dentro.append(ndep)   # dove il valore viene dalla realta' vista nell'immagine
        validi.append(ndep | (np.isfinite(Wks) if (Wks is not None and W0s is not None) else False))
    return uscite, validi, info


def vento_in_pixel(u_ms, v_ms, km_pixel, minuti_passo=10):
    """Vento (m/s, verso est e verso nord) -> pixel per passo sulla griglia (colonne verso est, righe verso SUD)."""
    f = minuti_passo * 60.0 / (km_pixel * 1000.0)
    return u_ms * f, -v_ms * f
