"""Collaborazione stretta tra osservazioni (radar, satellite) e modello WRF.

1) CORREZIONE DEL WRF PUNTO PER PUNTO ("field alignment", Ravela e altri 2007; Nehrkorn e altri 2015)
   Adesso si confronta il campo del WRF con quello osservato e si calcola, zona per zona, di quanto e in che
   direzione il WRF ha sbagliato posizione (flusso ottico tra i due campi: un vettore di spostamento per ogni punto,
   non uno solo per tutta la mappa). Poi si confrontano le intensita' (quantile mapping: il valore che nel WRF sta al
   90% della distribuzione diventa quello che nel radar sta al 90%). La stessa correzione viene applicata a tutte le
   ore successive del WRF: cosi' il WRF corretto "riparte" da dove sono davvero le cose, ma conserva tutto quello che
   il modello sa (fronti, nuove celle, cambi di direzione, intensificazioni).
   La correzione si attenua lentamente con le ore (le celle nuove non hanno l'errore di quelle di adesso).

2) PIU' MOVIMENTI CONTEMPORANEI (scomposizione in scale, come nel metodo STEPS, Bowler e altri 2006)
   Il campo osservato viene separato in una parte a grande scala (il sistema: fronte, banda di pioggia, circa 12 km
   e oltre) e una a piccola scala (le singole celle). Ognuna ha il suo movimento, misurato separatamente:
   tipicamente le celle corrono con il vento in quota (per esempio da sud verso nord) mentre il sistema avanza in
   un'altra direzione (per esempio da nord-ovest verso sud-est). Le celle durano poco e si attenuano prima.
   Il movimento del sistema, con il passare dei minuti, passa gradualmente a quello previsto dal WRF corretto,
   che sa se il fronte rallentera', accelerera' o curvera'.

3) PESI IMPARATI DAI DATI
   Quanto pesare estrapolazione e WRF a ogni scadenza non lo decidiamo a mano: ogni previsione viene verificata
   (verifica.py) separatamente per le due componenti, e quando ci sono abbastanza casi di pioggia i pesi vengono
   calcolati dall'abilita' misurata (formula dei pesi ottimali r/(1-r^2)).
"""
import math

import cv2
import numpy as np
from scipy import ndimage


def _8bit(x, lo, hi):
    return np.clip((np.nan_to_num(x, nan=lo) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def flusso(a, b, lo, hi, preset=cv2.DISOPTICAL_FLOW_PRESET_MEDIUM, patch=15):
    """Flusso ottico DIS da a verso b (colonne, righe)."""
    dis = cv2.DISOpticalFlow_create(preset)
    dis.setPatchSize(patch)
    dis.setPatchStride(4)
    f = dis.calc(_8bit(a, lo, hi), _8bit(b, lo, hi), None)
    return f[..., 0].astype("float64"), f[..., 1].astype("float64")


def liscia_pesata(u, peso, sigma, riempi=None):
    """Media mobile pesata: dove il peso e' zero il vettore viene dal dintorno, e lontano da tutto dal valore medio."""
    num = ndimage.gaussian_filter(u * peso, sigma)
    den = ndimage.gaussian_filter(peso, sigma)
    media = (u * peso).sum() / max(peso.sum(), 1e-9) if riempi is None else riempi
    a = np.clip(den / 0.15, 0, 1)
    return a * np.where(den > 1e-6, num / np.maximum(den, 1e-9), media) + (1 - a) * media


def deforma(campo, du, dv, fattore=1.0):
    """Sposta un campo lungo il campo di spostamenti (all'indietro): out(x) = campo(x - fattore*d(x)). NaN fuori."""
    ny, nx = campo.shape
    xx, yy = np.meshgrid(np.arange(nx, dtype="float32"), np.arange(ny, dtype="float32"))
    mx = (xx - fattore * du).astype("float32")
    my = (yy - fattore * dv).astype("float32")
    a = np.nan_to_num(campo).astype("float32")
    n = np.isfinite(campo).astype("float32")
    out = cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    nn = cv2.remap(n, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.where(nn > 0.5, out, np.nan)


def correlazione(a, b, m):
    a, b = a[m], b[m]
    if a.size < 200 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def allinea(oss, mod, trasforma, lo, hi, sigma_campo, sigma_spost, max_spost, soglia):
    """Spostamenti (du, dv) che portano il campo del modello su quello osservato, e qualita' prima/dopo.
    trasforma: funzione che porta i valori in una scala adatta al confronto (per la pioggia i decibel)."""
    m = np.isfinite(oss) & np.isfinite(mod)
    if m.sum() < 1000:
        return None, {"motivo": "zona comune troppo piccola"}
    a = ndimage.gaussian_filter(np.where(m, trasforma(mod), lo), sigma_campo)
    b = ndimage.gaussian_filter(np.where(m, trasforma(oss), lo), sigma_campo)
    bagn_o = (np.nan_to_num(oss) >= soglia)[m].mean()
    bagn_m = (np.nan_to_num(mod) >= soglia)[m].mean()
    info = {"area_osservata": round(float(bagn_o), 3), "area_modello": round(float(bagn_m), 3)}
    if bagn_o < 0.005 or bagn_m < 0.005:
        info["motivo"] = "troppo poco segnale in uno dei due campi per stimare gli spostamenti"
        return None, info
    du, dv = flusso(a, b, lo, hi, preset=cv2.DISOPTICAL_FLOW_PRESET_MEDIUM, patch=21)
    peso = (ndimage.gaussian_filter(((a > lo + 0.05 * (hi - lo)) | (b > lo + 0.05 * (hi - lo))).astype("float64"), 3) > 0.2)
    peso = peso.astype("float64")
    du = liscia_pesata(du, peso, sigma_spost)
    dv = liscia_pesata(dv, peso, sigma_spost)
    mod_ = np.hypot(du, dv)
    troppo = mod_ > max_spost
    du[troppo] *= max_spost / mod_[troppo]
    dv[troppo] *= max_spost / mod_[troppo]
    prima = correlazione(a, b, m)
    a2 = ndimage.gaussian_filter(np.nan_to_num(trasforma(deforma(mod, du, dv)), nan=lo), sigma_campo)
    dopo = correlazione(a2, b, m)
    info.update({"correlazione_prima": None if not np.isfinite(prima) else round(prima, 3),
                 "correlazione_dopo": None if not np.isfinite(dopo) else round(dopo, 3),
                 "spostamento_medio_px": round(float(np.average(mod_, weights=peso + 1e-9)), 1),
                 "spostamento_massimo_px": round(float(mod_.max()), 1)})
    if not np.isfinite(dopo) or (np.isfinite(prima) and dopo < prima + 0.02):
        info["motivo"] = "lo spostamento non migliora l'accordo: non applicato"
        return None, info
    return (du, dv), info


def tabella_quantili(oss, mod, m, limite=3.0):
    """Corrispondenza valore del modello -> valore osservato (quantile mapping), con limiti per non esagerare."""
    o = np.sort(np.nan_to_num(oss[m]))
    w = np.sort(np.nan_to_num(mod[m]))
    if o.size < 1000:
        return None
    p = np.linspace(0, 100, 201)
    qo = np.percentile(o, p)
    qw = np.percentile(w, p)
    # solo dove il modello ha valori distinti; valori corretti tra 1/limite e limite volte quelli del modello
    qw, idx = np.unique(qw, return_index=True)
    qo = qo[idx]
    qo = np.clip(qo, qw / limite, qw * limite)
    qo = np.maximum.accumulate(qo)
    return qw, qo


def applica_tabella(campo, tabella, alfa=1.0):
    if tabella is None:
        return campo
    qw, qo = tabella
    c = np.nan_to_num(campo)
    corr = np.interp(c, qw, qo, left=qo[0], right=None)
    # oltre il massimo visto oggi: stesso rapporto dell'ultimo quantile
    alto = c > qw[-1]
    if alto.any() and qw[-1] > 0:
        corr[alto] = c[alto] * (qo[-1] / qw[-1])
    out = (1 - alfa) * c + alfa * corr
    return np.where(np.isfinite(campo), out, np.nan)


def movimento_sequenza(campi, trasforma, lo, hi, sigma_campo, sigma_mov, soglia_peso):
    """Movimento medio (u, v per passo) da una sequenza di campi equidistanti, a una data scala."""
    forma = campi[-1].shape
    U, V, P = np.zeros(forma), np.zeros(forma), np.zeros(forma)
    for a, b in zip(campi[:-1], campi[1:]):
        ta = ndimage.gaussian_filter(np.nan_to_num(trasforma(a), nan=lo), sigma_campo) if sigma_campo else np.nan_to_num(trasforma(a), nan=lo)
        tb = ndimage.gaussian_filter(np.nan_to_num(trasforma(b), nan=lo), sigma_campo) if sigma_campo else np.nan_to_num(trasforma(b), nan=lo)
        peso = ((ta > soglia_peso) | (tb > soglia_peso)).astype("float64")
        if peso.sum() < 200:
            continue
        u, v = flusso(ta, tb, lo, hi)
        U += u * peso
        V += v * peso
        P += peso
    if P.sum() < 200:
        return np.zeros(forma), np.zeros(forma), None
    media = (U.sum() / P.sum(), V.sum() / P.sum())
    u = liscia_pesata(np.divide(U, P, out=np.zeros(forma), where=P > 0), (P > 0).astype("float64"), sigma_mov, media[0])
    v = liscia_pesata(np.divide(V, P, out=np.zeros(forma), where=P > 0), (P > 0).astype("float64"), sigma_mov, media[1])
    return u, v, media


def trasporta(campo, passi, movimento_al_passo, valido0=None):
    """Trasporto semi-lagrangiano con movimento che puo' cambiare a ogni passo.
    movimento_al_passo(k) -> (u, v) in pixel per passo. Ritorna [campi], [validi]."""
    ny, nx = campo.shape
    xx, yy = np.meshgrid(np.arange(nx, dtype="float32"), np.arange(ny, dtype="float32"))
    px, py = xx.copy(), yy.copy()
    sorgente = np.nan_to_num(campo).astype("float32")
    noto = (np.isfinite(campo) if valido0 is None else valido0).astype("float32")
    uscite, validi = [], []
    for k in range(1, passi + 1):
        u, v = movimento_al_passo(k)
        du = cv2.remap(u.astype("float32"), px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dv = cv2.remap(v.astype("float32"), px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        px = px - du
        py = py - dv
        uscite.append(cv2.remap(sorgente, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        validi.append(cv2.remap(noto, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5)
    return uscite, validi


def pesi_ottimali(r_e, r_w):
    """Peso dell'estrapolazione dati i coefficienti di correlazione delle due previsioni (errori indipendenti)."""
    r_e = min(max(r_e, 0.0), 0.995)
    r_w = min(max(r_w, 0.0), 0.995)
    a = r_e / (1 - r_e * r_e)
    b = r_w / (1 - r_w * r_w)
    if a + b <= 0:
        return 0.5
    return a / (a + b)


def abilita_misurata(pagella, minuti, chiave):
    """Correlazione media misurata nei giorni scorsi per la componente 'chiave' (estrapolazione / wrf) alla scadenza,
    interpolata tra le scadenze verificate. None se i casi non bastano."""
    if not pagella:
        return None
    tab = pagella.get("abilita_componenti") or {}
    punti = []
    for s, v in tab.items():
        c = v.get(chiave)
        if c is not None and v.get(f"casi_{chiave}", v.get("casi", 0)) >= 12:
            punti.append((int(s), float(c)))
    if len(punti) < 2:
        return None
    punti.sort()
    xs = [p[0] for p in punti]
    ys = [p[1] for p in punti]
    if minuti <= xs[0]:
        return ys[0]
    if minuti >= xs[-1]:
        # oltre l'ultima scadenza verificata: per l'estrapolazione si prosegue con la stessa diminuzione, il WRF resta
        if chiave == "estrapolazione" and ys[-1] > 0.01 and ys[-2] > 0.01:
            tau = (xs[-1] - xs[-2]) / max(1e-3, math.log(ys[-2] / ys[-1]))
            return ys[-1] * math.exp(-(minuti - xs[-1]) / max(tau, 1.0))
        return ys[-1]
    return float(np.interp(minuti, xs, ys))
