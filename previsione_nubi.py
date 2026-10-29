"""Previsione delle nubi 0-6 ore: stesso metodo della pioggia (nowcast.py), applicato alla nuvolosita' del satellite.

 1. MOVIMENTO: dagli ultimi 4 scatti (ogni 10 minuti) si calcola come si spostano le nubi (flusso ottico DIS).
 2. ESTRAPOLAZIONE: l'ultimo scatto viene trasportato avanti lungo il movimento (schema semi-lagrangiano).
 3. VITA MEDIA: quanto e' affidabile l'estrapolazione oggi (scatto di 30 minuti fa trasportato fino ad adesso e
    confrontato con quello vero).
 4. WRF: la nuvolosita' del WRF (dalla radiazione infrarossa al suolo, vedi wrf.py) viene confrontata con quella vista
    dal satellite adesso, per sapere quanto e' affidabile oggi.
 5. FUSIONE: pesi ottimali r/(1-r^2) come per la pioggia: all'inizio le nubi viste dal satellite, poi sempre piu' il WRF.
Tutto si calcola su una griglia a meta' definizione della zona Lazio (circa 560 m), poi si riporta alla definizione piena.
"""
import math
from datetime import timedelta

import cv2
import numpy as np
from scipy import ndimage

import nowcast

PASSO = 10   # minuti


def _8bit(alfa, lum):
    return np.clip(np.nan_to_num(alfa) * (0.5 + 0.5 * np.nan_to_num(lum)) * 255, 0, 255).astype(np.uint8)


def movimento(campi):
    """campi: lista di (alfa, lum) a 10 minuti. Ritorna u, v (pixel ogni 10 minuti) e la velocita' media."""
    forma = campi[-1][0].shape
    U, V, P = np.zeros(forma), np.zeros(forma), np.zeros(forma)
    for (a0, l0), (a1, l1) in zip(campi[:-1], campi[1:]):
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setPatchSize(15)
        dis.setPatchStride(4)
        f = dis.calc(_8bit(a0, l0), _8bit(a1, l1), None)
        peso = ((np.nan_to_num(a0) > 0.15) | (np.nan_to_num(a1) > 0.15)).astype("float64")
        peso = ndimage.binary_dilation(peso > 0, iterations=2).astype("float64")
        U += f[..., 0] * peso
        V += f[..., 1] * peso
        P += peso
    if P.sum() < 500:
        return np.zeros(forma), np.zeros(forma), None
    media_u, media_v = U.sum() / P.sum(), V.sum() / P.sum()
    out = []
    for A in (U, V):
        num = ndimage.gaussian_filter(A, 25)
        den = ndimage.gaussian_filter(P, 25)
        locale = np.where(den > 0.05, num / np.maximum(den, 1e-9), np.nan)
        media = media_u if A is U else media_v
        peso = np.clip(den / 0.3, 0, 1)
        out.append(np.where(np.isfinite(locale), peso * locale + (1 - peso) * media, media))
    return out[0], out[1], (media_u, media_v)


def correlazione(a, b, m):
    a, b = a[m], b[m]
    if a.size < 200 or a.std() < 1e-6 or b.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def prevedi(campi, t_ultimo, tempi_voluti, wrf_prev, lon, lat, passo_km):
    """campi: lista di (alfa, lum) a mezza definizione, ogni 10 minuti, l'ultimo a t_ultimo.
    tempi_voluti: istanti (datetime) per cui serve la previsione (anche prima di adesso, se il satellite e' in ritardo).
    lon, lat: coordinate della griglia (mezza definizione). Ritorna {istante: (alfa, lum)} e un dizionario di info."""
    info = {"ultimo_satellite": t_ultimo.strftime("%Y-%m-%dT%H:%MZ")}
    alfa0, lum0 = campi[-1]
    u, v, media = movimento(campi) if len(campi) >= 2 else (np.zeros(alfa0.shape), np.zeros(alfa0.shape), None)
    if media is not None:
        kmh = math.hypot(*media) * passo_km * 6
        info["velocita_kmh"] = round(kmh, 1)
        info["direzione"] = round((math.degrees(math.atan2(-media[0], media[1])) + 360) % 360) if kmh > 2 else None
    # vita media delle nubi (estrapolazione di 30 minuti fa contro lo scatto di adesso)
    tau = 180.0
    if len(campi) >= 4:
        prev, val = nowcast.estrapola(campi[-4][0], u, v, 3)
        m = val[-1] & np.isfinite(alfa0)
        r = correlazione(ndimage.gaussian_filter(prev[-1], 4), ndimage.gaussian_filter(np.nan_to_num(alfa0), 4), m)
        if np.isfinite(r) and r > 0.05:
            tau = float(np.clip(-30.0 / math.log(min(r, 0.995)), 30.0, 480.0))
        info["correlazione_30min"] = None if not np.isfinite(r) else round(r, 3)
    info["vita_media_min"] = round(tau)
    # quanti passi servono
    passi = max(0, max(int(round((t - t_ultimo).total_seconds() / 60 / PASSO)) for t in tempi_voluti))
    if passi == 0:
        return {}, info
    a_est, valido = nowcast.estrapola(np.where(np.isfinite(alfa0), alfa0, np.nan), u, v, passi)
    l_est, _ = nowcast.estrapola(np.where(np.isfinite(lum0), lum0, np.nan), u, v, passi)
    # abilita' del WRF oggi
    c, w_info = 0.25, {"usato": False}
    if wrf_prev is not None:
        try:
            import wrf as wrf_mod
            ponte = wrf_mod.Ponte(wrf_prev, lon, lat)
            cw0 = wrf_prev.nuvolosita(t_ultimo)
            if cw0 is not None:
                cw0 = ponte.porta(cw0)
                m = np.isfinite(cw0) & np.isfinite(alfa0)
                r = correlazione(ndimage.gaussian_filter(np.nan_to_num(alfa0), 9), np.nan_to_num(cw0), m)
                c = float(np.clip(r if np.isfinite(r) else 0.05, 0.05, 0.7))
                w_info = {"usato": True, "abilita": round(c, 2), "nuvolosita_wrf": round(float(np.nanmean(cw0)), 2),
                          "nuvolosita_satellite": round(float(np.nanmean(alfa0)), 2)}
        except Exception as e:  # noqa: BLE001
            w_info = {"usato": False, "errore": str(e)[:200]}
            ponte = None
    info["wrf"] = w_info
    out = {}
    sfuma = lambda m: ndimage.gaussian_filter(m.astype("float64"), 10)  # noqa: E731  passaggi morbidi (circa 6 km)
    for t in tempi_voluti:
        k = int(round((t - t_ultimo).total_seconds() / 60 / PASSO))
        if k <= 0:
            continue
        v = sfuma(valido[k - 1])                    # 1 dove le nubi arrivano da zone viste, 0 dove arrivano da fuori
        a = np.nan_to_num(a_est[k - 1]).astype("float64")
        lm = np.where(valido[k - 1], l_est[k - 1], 0.85).astype("float64")
        cw = None
        if w_info.get("usato"):
            cw = wrf_prev.nuvolosita(t)
            cw = ponte.porta(cw) if cw is not None else None
        if cw is not None:
            zw = sfuma(np.isfinite(cw))              # 1 dentro il WRF, sfuma verso il bordo
            cwf = np.nan_to_num(cw) ** 1.3          # nuvolosita' del modello resa come opacita' (le velature leggere restano leggere)
            w = nowcast.peso_radar(k * PASSO, tau, c)
            # dove l'estrapolazione non sa (nubi in arrivo da fuori) si usa il WRF; altrove la fusione con i pesi ottimali
            fuso = w * a + (1 - w) * cwf
            a = zw * (v * fuso + (1 - v) * cwf) + (1 - zw) * v * a
            lm = zw * (w * lm + (1 - w) * (0.80 + 0.15 * cwf)) + (1 - zw) * lm
        else:
            a = v * a
        out[t] = (np.clip(a, 0, 1), np.clip(lm, 0, 1))
    return out, info


__all__ = ["prevedi", "movimento", "timedelta"]
