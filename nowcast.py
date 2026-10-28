#!/usr/bin/env python3
"""Previsione a brevissimo termine (0-6 ore) della pioggia: radar della Protezione Civile + modello WRF di Nerola.

  nowcast.py --dati CARTELLA_SCARICO --wrf d01_tutto.npz --out nowcast.pkl

Come funziona (tutto calcolato, nessuna regola "a occhio"):

 1. MOVIMENTO. Dagli ultimi 4 scatti del radar (uno ogni 10 minuti) si calcola, punto per punto, in che direzione e a che
    velocita' si sposta la pioggia (flusso ottico, metodo DIS: Kroeger e altri, 2016). Il campo di velocita' viene
    lisciato e, dove non piove, riempito con il movimento medio della zona.
 2. ESTRAPOLAZIONE. L'ultimo scatto viene trasportato in avanti lungo quel movimento, ogni 10 minuti fino a 6 ore
    (schema semi-lagrangiano all'indietro, come nel metodo S-PROG/STEPS: Seed 2003, Bowler e altri 2006). Ogni punto
    della previsione "va a prendere" la pioggia nel punto da cui arriva, senza sbavare il campo.
 3. QUANTO DURA LA PIOGGIA. Si misura quanto e' affidabile l'estrapolazione oggi: si trasporta lo scatto di 30 minuti fa
    fino ad adesso e lo si confronta con quello vero (correlazione). Da qui la "vita media" delle strutture di pioggia
    (tau): con temporali e' breve (30-60 min), con piogge estese e' lunga (2-4 ore).
 4. CORREZIONE DEL WRF. La pioggia prevista dal WRF per adesso viene confrontata con il radar di adesso:
    - spostamento: se il WRF ha messo la pioggia un po' piu' in la', si trova lo scostamento (correlazione di fase)
      e lo si toglie a tutte le ore successive (solo se migliora davvero l'accordo e se e' entro 40 km);
    - intensita': rapporto tra la pioggia media del radar e quella del WRF (limitato tra 1/3 e 3);
    - abilita': correlazione tra radar e WRF corretto (quanto ci azzecca oggi il WRF).
 5. FUSIONE. Ad ogni scadenza si combinano estrapolazione e WRF con i pesi ottimali per due previsioni con errori
    indipendenti: peso proporzionale a r/(1-r^2), dove r e' l'abilita' di ciascuna. All'inizio conta quasi solo il radar,
    poi il peso passa gradualmente al WRF, tanto piu' in fretta quanto piu' la pioggia e' "di breve vita".
    Infine la distribuzione delle intensita' viene ripristinata (probability matching, come in STEPS) per non avere
    una pioggia "annacquata" su un'area troppo grande.
"""
import argparse
import math
import pickle
import sys
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
from scipy import ndimage

PASSO_MIN = 10          # minuti tra un fotogramma e l'altro della previsione
ORE = 6
SOGLIA = 0.1            # mm/h: sotto e' "non piove"
VEL_MAX = 25.0          # km ogni 10 minuti (150 km/h): oltre e' sicuramente un errore del flusso ottico
TAU_DEFAULT = 90.0      # minuti, quando non c'e' abbastanza pioggia per misurarla
C_DEFAULT = 0.3         # abilita' del WRF quando non si puo' misurare


def tempo(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------------------------------- movimento

def in_db(r):
    """Pioggia in decibel (10*log10 R): il flusso ottico lavora meglio su una scala logaritmica."""
    r = np.nan_to_num(np.asarray(r, dtype="float64"), nan=0.0)
    return np.where(r >= SOGLIA, 10 * np.log10(np.maximum(r, SOGLIA)), -15.0)


def in_8bit(r):
    d = in_db(r)
    return np.clip((d + 15.0) * (255.0 / 35.0), 0, 255).astype(np.uint8)


def flusso_coppia(a, b):
    """Spostamento (colonne, righe) che porta l'immagine a sull'immagine b."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setPatchSize(15)
    dis.setPatchStride(4)
    dis.setFinestScale(1)
    f = dis.calc(in_8bit(a), in_8bit(b), None)
    return f[..., 0].astype("float64"), f[..., 1].astype("float64")


def liscia_pesata(campo, peso, sigma):
    num = ndimage.gaussian_filter(campo * peso, sigma, mode="nearest")
    den = ndimage.gaussian_filter(peso, sigma, mode="nearest")
    return num, den


def movimento(campi):
    """Campo di velocita' (u verso destra, v verso il basso) in pixel ogni 10 minuti, dagli ultimi scatti (ogni 10 min)."""
    forma = campi[-1].shape
    us, vs, pesi = [], [], []
    for a, b in zip(campi[:-1], campi[1:]):
        pioggia = (np.nan_to_num(a) >= SOGLIA) | (np.nan_to_num(b) >= SOGLIA)
        if pioggia.sum() < 200:
            continue
        u, v = flusso_coppia(a, b)
        peso = ndimage.binary_dilation(pioggia, iterations=3).astype("float64")
        peso *= np.isfinite(a) & np.isfinite(b)
        us.append(u * peso)
        vs.append(v * peso)
        pesi.append(peso)
    if not pesi:
        return np.zeros(forma), np.zeros(forma), {"pixel_pioggia": 0, "velocita_media_kmh": 0.0, "direzione": None}
    P = np.sum(pesi, axis=0)
    U = np.sum(us, axis=0)
    V = np.sum(vs, axis=0)
    # lisciatura a due scale: locale (20 km) dove piove, d'insieme (80 km) per riempire i vuoti
    nu, du = liscia_pesata(np.divide(U, P, out=np.zeros(forma), where=P > 0), (P > 0).astype("float64"), 20)
    nv, _ = liscia_pesata(np.divide(V, P, out=np.zeros(forma), where=P > 0), (P > 0).astype("float64"), 20)
    gu, gd = liscia_pesata(np.divide(U, P, out=np.zeros(forma), where=P > 0), (P > 0).astype("float64"), 80)
    gv, _ = liscia_pesata(np.divide(V, P, out=np.zeros(forma), where=P > 0), (P > 0).astype("float64"), 80)
    media_u = (U.sum() / P.sum())
    media_v = (V.sum() / P.sum())
    lu = np.where(du > 1e-3, nu / np.maximum(du, 1e-9), np.nan)
    lv = np.where(du > 1e-3, nv / np.maximum(du, 1e-9), np.nan)
    ggu = np.where(gd > 1e-4, gu / np.maximum(gd, 1e-9), media_u)
    ggv = np.where(gd > 1e-4, gv / np.maximum(gd, 1e-9), media_v)
    # peso del campo locale: 1 dove c'e' pioggia vicina, sfuma verso il campo d'insieme
    a = np.clip(du / 0.2, 0, 1)
    u = np.where(np.isfinite(lu), a * lu + (1 - a) * ggu, ggu)
    v = np.where(np.isfinite(lv), a * lv + (1 - a) * ggv, ggv)
    vel = np.hypot(u, v)
    troppo = vel > VEL_MAX
    u[troppo] *= VEL_MAX / vel[troppo]
    v[troppo] *= VEL_MAX / vel[troppo]
    kmh = math.hypot(media_u, media_v) * 6.0   # 1 pixel = 1 km, ogni 10 minuti -> x6 = km/h
    # direzione di PROVENIENZA, come per il vento (le righe crescono verso sud)
    direz = (math.degrees(math.atan2(-media_u, media_v)) + 360) % 360 if kmh > 1 else None
    return u, v, {"pixel_pioggia": int((P > 0).sum() / max(len(pesi), 1)), "velocita_media_kmh": round(kmh, 1),
                  "direzione": None if direz is None else round(direz)}


# ---------------------------------------------------------------------------------------------------- estrapolazione

def estrapola(campo0, u, v, passi):
    """Trasporta campo0 lungo (u, v) per `passi` volte. Ritorna (campi, valido): valido=False dove la pioggia
    arriverebbe da fuori dell'area vista dal radar."""
    ny, nx = campo0.shape
    xx, yy = np.meshgrid(np.arange(nx, dtype="float32"), np.arange(ny, dtype="float32"))
    sorgente = np.nan_to_num(campo0, nan=0.0).astype("float32")
    noto = np.isfinite(campo0).astype("float32")
    u32, v32 = u.astype("float32"), v.astype("float32")
    px, py = xx.copy(), yy.copy()
    uscite, valide = [], []
    for _ in range(passi):
        # all'indietro lungo la traiettoria: il punto "viene da" dove la velocita' lo porta indietro
        du = cv2.remap(u32, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dv = cv2.remap(v32, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        px = px - du
        py = py - dv
        r = cv2.remap(sorgente, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        n = cv2.remap(noto, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        uscite.append(np.where(r >= SOGLIA, r, 0.0).astype("float32"))
        valide.append(n > 0.5)
    return uscite, valide


def correlazione(a, b, maschera):
    a = a[maschera]
    b = b[maschera]
    if a.size < 50 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def vita_media(campi, u, v):
    """Correlazione tra lo scatto di 30 min fa trasportato fino ad adesso e lo scatto di adesso -> tau (minuti)."""
    if len(campi) < 4:
        return TAU_DEFAULT, None
    vecchio, adesso = campi[-4], campi[-1]
    prev, valido = estrapola(vecchio, u, v, 3)
    m = valido[-1] & np.isfinite(adesso) & ((np.nan_to_num(adesso) >= SOGLIA) | (prev[-1] >= SOGLIA))
    if m.sum() < 300:
        return TAU_DEFAULT, None
    # stessa scala del confronto con il WRF (campi lisciati di circa 10 km), cosi' le due abilita' sono confrontabili
    a = ndimage.gaussian_filter((in_db(prev[-1]) + 15) * valido[-1], 6)
    b = ndimage.gaussian_filter((in_db(adesso) + 15) * np.isfinite(adesso), 6)
    r = correlazione(a, b, m)
    if not np.isfinite(r) or r <= 0.05:
        return 20.0, r
    tau = -30.0 / math.log(min(r, 0.995))
    return float(np.clip(tau, 20.0, 300.0)), r


# ---------------------------------------------------------------------------------------------------- WRF

def correggi_wrf(radar0, wrf0):
    """Confronta WRF e radar adesso. Ritorna (spostamento colonne, righe), fattore di intensita', abilita' c, note."""
    m = np.isfinite(wrf0) & np.isfinite(radar0)
    note = {}
    if m.sum() < 1000:
        return (0.0, 0.0), 1.0, C_DEFAULT, {"motivo": "zona comune troppo piccola"}
    r = np.where(m, np.nan_to_num(radar0), 0.0)
    w = np.where(m, np.nan_to_num(wrf0), 0.0)
    bagnato_r = (r >= SOGLIA)[m].mean()
    bagnato_w = (w >= SOGLIA)[m].mean()
    note.update({"area_pioggia_radar": round(float(bagnato_r), 3), "area_pioggia_wrf": round(float(bagnato_w), 3)})
    if bagnato_r < 0.01 and bagnato_w < 0.01:
        note["motivo"] = "quasi nessuna pioggia ne' nel radar ne' nel WRF"
        return (0.0, 0.0), 1.0, C_DEFAULT, note
    # campi lisciati (10 km) su scala logaritmica, per confrontare le strutture e non i singoli pixel
    # (in_db + 15) vale 0 dove non piove: cosi' il bordo della zona del WRF non crea un falso "accordo"
    lr = ndimage.gaussian_filter((in_db(r) + 15) * m, 6)
    lw = ndimage.gaussian_filter((in_db(w) + 15) * m, 6)
    c_prima = correlazione(lr, lw, m)
    sposta = (0.0, 0.0)
    if bagnato_r >= 0.01 and bagnato_w >= 0.01:
        rr, cc = np.where(m)
        r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
        a = (lw[r0:r1, c0:c1] - lw[r0:r1, c0:c1].mean()).astype("float32")
        b = (lr[r0:r1, c0:c1] - lr[r0:r1, c0:c1].mean()).astype("float32")
        fin = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (dx, dy), _ = cv2.phaseCorrelate(a, b, fin)
        if math.hypot(dx, dy) <= 40:
            ws = sposta_campo(w, dx, dy)
            mm = m & np.isfinite(ws)
            c_dopo = correlazione(lr, ndimage.gaussian_filter((in_db(np.nan_to_num(ws)) + 15) * mm, 6), mm)
            note["correlazione_prima"] = None if not np.isfinite(c_prima) else round(c_prima, 3)
            note["correlazione_dopo"] = None if not np.isfinite(c_dopo) else round(c_dopo, 3)
            if np.isfinite(c_dopo) and (not np.isfinite(c_prima) or c_dopo > c_prima + 0.05):
                sposta = (float(dx), float(dy))
                w = np.where(mm, np.nan_to_num(ws), 0.0)
                m = mm
    fattore = 1.0
    if bagnato_r >= 0.01 and bagnato_w >= 0.01:
        zona = m & ((r >= SOGLIA) | (w >= SOGLIA))
        if zona.sum() > 100 and w[zona].mean() > 0:
            fattore = float(np.clip(r[zona].mean() / w[zona].mean(), 1 / 3, 3))
    lw2 = ndimage.gaussian_filter((in_db(w * fattore) + 15) * m, 6)
    c = correlazione(lr, lw2, m)
    if not np.isfinite(c):
        c = 0.05
    c = float(np.clip(c, 0.05, 0.7))
    return sposta, fattore, c, note


def sposta_campo(campo, dx, dy):
    """Trasla un campo di (dx colonne, dy righe); dove si scopre il bordo mette NaN."""
    a = np.nan_to_num(campo, nan=0.0).astype("float32")
    noto = np.isfinite(campo).astype("float32")
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    n = cv2.warpAffine(noto, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.where(n > 0.99, out, np.nan)


def peso_radar(minuti, tau, c):
    """Peso dell'estrapolazione: abilita' del radar r=exp(-t/tau), del WRF c; peso ottimale ~ r/(1-r^2)."""
    r = min(math.exp(-minuti / tau), 0.999)
    a = r / (1 - r * r)
    b = c / (1 - c * c)
    return a / (a + b)


def fondi(estr, valido, wrfc, w):
    """Unisce estrapolazione e WRF con peso w (al radar) e ripristina la distribuzione delle intensita'."""
    out = np.where(valido, estr, np.nan).astype("float64")
    if wrfc is None:
        return out
    zona_wrf = np.isfinite(wrfc)
    solo_wrf = zona_wrf & ~valido
    out[solo_wrf] = wrfc[solo_wrf]
    m = zona_wrf & valido
    if m.sum() > 0:
        e = estr[m].astype("float64")
        f = wrfc[m]
        lineare = w * e + (1 - w) * f
        # probability matching: valori ordinati come nel campo lineare, ma con le intensita' "medie" (quantili pesati)
        bersaglio = w * np.sort(e) + (1 - w) * np.sort(f)
        ordine = np.argsort(lineare, kind="stable")
        nuovo = np.empty_like(lineare)
        nuovo[ordine] = bersaglio
        nuovo[nuovo < SOGLIA] = 0.0
        out[m] = nuovo
    return out


# ---------------------------------------------------------------------------------------------------- tutto insieme

def calcola(radar, wrf_prev=None, ponte_fn=None):
    campi_tutti = [c for c in radar["campi"]]
    tempi = [tempo(t) for t in radar["tempi"]]
    t0 = tempi[-1]
    # ultimi scatti a 10 minuti di distanza esatta (per il movimento servono 4, per la vita media lo scatto di 30 min fa)
    per_tempo = dict(zip(tempi, campi_tutti))
    seq = []
    for k in range(3, -1, -1):
        t = t0 - timedelta(minutes=PASSO_MIN * k)
        if t in per_tempo:
            seq.append(per_tempo[t])
    if len(seq) < 2:
        raise RuntimeError("servono almeno 2 scatti radar consecutivi a 10 minuti")
    info = {"t0": t0.strftime("%Y-%m-%dT%H:%MZ"), "scatti_per_il_movimento": len(seq)}
    u, v, im = movimento(seq)
    info["movimento"] = im
    tau, r30 = vita_media(seq, u, v) if len(seq) == 4 else (TAU_DEFAULT, None)
    info["vita_media_min"] = round(tau)
    info["correlazione_30min"] = None if r30 is None or not np.isfinite(r30) else round(r30, 3)
    passi = ORE * 60 // PASSO_MIN
    estr, valide = estrapola(seq[-1], u, v, passi)
    radar0 = seq[-1]
    wrf_info = {"usato": False}
    sposta, fattore, c = (0.0, 0.0), 1.0, C_DEFAULT
    ponte = None
    if wrf_prev is not None and ponte_fn is not None:
        ponte = ponte_fn(wrf_prev)
        w0 = wrf_prev.campo(t0)
        if w0 is not None:
            wrf0 = ponte.porta(w0)
            sposta, fattore, c, note = correggi_wrf(radar0, wrf0)
            wrf_info = {"usato": True, "corsa": wrf_prev.corsa.strftime("%Y-%m-%d %H UTC"),
                        "spostamento_km": [round(sposta[0], 1), round(-sposta[1], 1)], "fattore_intensita": round(fattore, 2),
                        "abilita": round(c, 2), **note}
        else:
            wrf_info = {"usato": False, "motivo": f"la corsa WRF {wrf_prev.corsa:%Y-%m-%d %H} UTC non copre l'ora attuale"}
    info["wrf"] = wrf_info
    previsioni, pesi = [np.where(np.isfinite(radar0), np.nan_to_num(radar0), np.nan).astype("float32")], [1.0]
    for k in range(1, passi + 1):
        minuti = k * PASSO_MIN
        t = t0 + timedelta(minutes=minuti)
        wrfc = None
        if wrf_info["usato"]:
            wc = wrf_prev.campo(t)
            if wc is not None:
                wrfc = ponte.porta(wc)
                if sposta != (0.0, 0.0):
                    wrfc = sposta_campo(wrfc, *sposta)
                wrfc = wrfc * fattore
        w = peso_radar(minuti, tau, c) if wrfc is not None else 1.0
        previsioni.append(fondi(estr[k - 1], valide[k - 1], wrfc, w).astype("float32"))
        pesi.append(round(w, 3))
    info["peso_radar"] = {f"+{k * PASSO_MIN}": pesi[k] for k in (1, 3, 6, 12, 18, 24, 30, 36)}
    return {"campi": np.stack(previsioni), "tempi": [(t0 + timedelta(minutes=k * PASSO_MIN)).strftime("%Y-%m-%dT%H:%MZ")
                                                    for k in range(passi + 1)],
            "pesi": pesi, "griglia": radar["griglia"], "info": info, "u": u.astype("float32"), "v": v.astype("float32")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dati", required=True)
    ap.add_argument("--wrf", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(f"{a.dati}/radar.pkl", "rb") as f:
        radar = pickle.load(f)
    wrf_prev, ponte_fn = None, None
    if a.wrf:
        try:
            import wrf as wrf_mod
            wrf_prev = wrf_mod.PrevisioneWRF(a.wrf)
            lon, lat = radar["griglia"].lonlat()
            ponte_fn = lambda w: wrf_mod.Ponte(w, lon, lat)  # noqa: E731
            print(f"WRF: corsa {wrf_prev.corsa:%Y-%m-%d %H} UTC, da {wrf_prev.centri[0]:%d %H:%M} a {wrf_prev.centri[-1]:%d %H:%M} UTC")
        except Exception as e:  # noqa: BLE001
            print(f"ATTENZIONE: WRF non leggibile ({e}), solo radar", file=sys.stderr)
    ris = calcola(radar, wrf_prev, ponte_fn)
    with open(a.out, "wb") as f:
        pickle.dump(ris, f)
    import json
    print(json.dumps(ris["info"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
