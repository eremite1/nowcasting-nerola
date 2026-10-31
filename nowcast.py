#!/usr/bin/env python3
"""Previsione a brevissimo termine (0-6 ore) della pioggia: radar della Protezione Civile + modello WRF di Nerola.

  nowcast.py --dati CARTELLA_SCARICO --wrf d01_tutto.npz --pagella pagella.json --out nowcast.pkl

Come funziona (tutto calcolato, nessuna regola "a occhio"; dettagli in fusione.py):

 1. PIU' MOVIMENTI. Il campo di pioggia viene separato in due scale: le CELLE (sotto circa 20 km) e il SISTEMA
    (fronte, banda di pioggia). Le celle si muovono come misurato tra gli ultimi scatti a 10 minuti (flusso ottico
    DIS, Kroeger e altri 2016); il sistema come misurato su intervalli di 60 minuti su campi molto sfumati, dove le
    celle non si vedono. Esempio: celle che corrono da sud a nord dentro un fronte che avanza da nord-ovest.
 2. VITA MEDIA, separata per celle e sistema: le celle si attenuano prima (come nel metodo STEPS, Bowler e altri 2006).
 3. WRF CORRETTO PUNTO PER PUNTO. Il WRF di adesso viene confrontato con il radar di adesso: si calcola zona per zona
    di quanto ha sbagliato posizione (un vettore per ogni punto, "field alignment") e intensita' (quantile mapping);
    la stessa correzione si applica a tutte le ore successive del WRF, che cosi' riparte da dove la pioggia c'e'
    davvero ma conserva cio' che il modello sa (nuove celle, fronti, intensificazioni, cambi di direzione).
 4. IL MOVIMENTO DEL SISTEMA, con il passare dei minuti, passa a quello del WRF corretto (che sa se il fronte curvera'
    o rallentera').
 5. FUSIONE con pesi ottimali r/(1-r^2) tra estrapolazione e WRF corretto. Le abilita' r vengono dalle verifiche dei
    giorni scorsi (verifica.py, componente per componente) quando ci sono abbastanza casi di pioggia; altrimenti dalla
    vita media misurata adesso e dall'accordo del WRF di adesso. Alla fine la distribuzione delle intensita' viene
    ripristinata (probability matching) per non avere una pioggia "annacquata".
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
SIGMA_SISTEMA = 20      # km: oltre questa scala e' "sistema" (fronte, banda), sotto sono "celle"
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


def _wrf_totale():
    """Prova "WRF totale" (file modalita.json, voce wrf_totale): previsione rielaborata solo con il WRF."""
    try:
        import json as _j
        import os as _o
        return bool(_j.load(open(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "modalita.json"))).get("wrf_totale"))
    except Exception:  # noqa: BLE001
        return False


WRF_TOTALE = _wrf_totale()


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

def _db15(r):
    """Decibel spostati di 15 (0 = non piove), per i confronti tra campi."""
    return in_db(r) + 15.0


def _corr_scala(prev, oss, valido, sigma=0):
    a = ndimage.gaussian_filter(prev, sigma) if sigma else prev
    b = ndimage.gaussian_filter(oss, sigma) if sigma else oss
    return correlazione(a, b, valido)


def _tau(r, minuti, lo, hi, default):
    if r is None or not np.isfinite(r):
        return default
    if r <= 0.05:
        return lo
    return float(np.clip(-minuti / math.log(min(r, 0.995)), lo, hi))


def ponte_riempi(a):
    """Fuori dall'area del WRF si usa il valore del punto piu' vicino dentro l'area."""
    noto = np.isfinite(a)
    if noto.all() or not noto.any():
        return np.nan_to_num(a)
    idx = ndimage.distance_transform_edt(~noto, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def calcola(radar, wrf_prev=None, ponte_fn=None, pagella=None):
    import fusione
    tempi = [tempo(t) for t in radar["tempi"]]
    t0 = tempi[-1]
    per_tempo = dict(zip(tempi, [c for c in radar["campi"]]))
    seq = [per_tempo[t0 - timedelta(minutes=PASSO_MIN * k)] for k in range(3, -1, -1) if t0 - timedelta(minutes=PASSO_MIN * k) in per_tempo]
    if len(seq) < 2:
        raise RuntimeError("servono almeno 2 scatti radar consecutivi a 10 minuti")
    info = {"t0": t0.strftime("%Y-%m-%dT%H:%MZ"), "scatti_per_il_movimento": len(seq), "metodo": "due scale + WRF corretto punto per punto"}
    radar0 = seq[-1]
    noto0 = np.isfinite(radar0)
    passi = ORE * 60 // PASSO_MIN

    # ---- 1. movimenti: celle (piccola scala) e sistema (grande scala)
    u_c, v_c, im = movimento(seq)
    # il sistema (fronte, banda) si misura su intervalli di 60 minuti e su campi molto sfumati (circa 20 km), dove le
    # singole celle non si vedono piu': cosi' non si confonde il movimento delle celle con quello del fronte
    coppie = []
    for j in range(3):
        a_t, b_t = t0 - timedelta(minutes=60 + 10 * j), t0 - timedelta(minutes=10 * j)
        if a_t in per_tempo and b_t in per_tempo:
            coppie.append((per_tempo[a_t], per_tempo[b_t]))
    u_s, v_s, media_s = None, None, None
    if coppie:
        U = np.zeros(radar0.shape); V = np.zeros(radar0.shape); P = np.zeros(radar0.shape)
        for a, b in coppie:
            ta = ndimage.gaussian_filter(np.where(np.isfinite(a), in_db(a) + 15, 0.0), SIGMA_SISTEMA)
            tb = ndimage.gaussian_filter(np.where(np.isfinite(b), in_db(b) + 15, 0.0), SIGMA_SISTEMA)
            peso = ((ta > 2) | (tb > 2)).astype("float64")
            if peso.sum() < 500:
                continue
            uu, vv = fusione.flusso(ta, tb, 0.0, 35.0)
            U += uu * peso / 6.0
            V += vv * peso / 6.0
            P += peso
        if P.sum() > 500:
            media_s = (U.sum() / P.sum(), V.sum() / P.sum())
            u_s = fusione.liscia_pesata(np.divide(U, P, out=np.zeros_like(U), where=P > 0), (P > 0).astype("float64"), 50, media_s[0])
            v_s = fusione.liscia_pesata(np.divide(V, P, out=np.zeros_like(V), where=P > 0), (P > 0).astype("float64"), 50, media_s[1])
    # controllo: il movimento d'insieme su 60 minuti si confonde quando i temporali nascono e muoiono sul posto (sembra
    # che "la pioggia si sposti" dove sta crescendo). Se non va d'accordo con il movimento misurato ogni 10 minuti
    # (direzione diversa di oltre 60 gradi o velocita' piu' che doppia) non lo si usa.
    if media_s is not None:
        pz = np.nan_to_num(radar0) >= SOGLIA
        mc = (float(u_c[pz].mean()), float(v_c[pz].mean())) if pz.sum() > 50 else (float(u_c.mean()), float(v_c.mean()))
        vs_, vc_ = math.hypot(*media_s), math.hypot(*mc)
        cosang = (media_s[0] * mc[0] + media_s[1] * mc[1]) / max(vs_ * vc_, 1e-9)
        if vc_ > 0.3 and (cosang < 0.5 or vs_ > 2 * vc_ + 0.5):
            info["movimento_sistema_scartato"] = True
            u_s, v_s, media_s = None, None, None
    if u_s is None:
        u_s, v_s = ndimage.gaussian_filter(u_c, 20), ndimage.gaussian_filter(v_c, 20)
    info["movimento"] = im
    if media_s is not None:
        kmh = math.hypot(*media_s) * 6.0
        info["movimento_sistema"] = {"velocita_kmh": round(kmh, 1),
                                     "direzione": round((math.degrees(math.atan2(-media_s[0], media_s[1])) + 360) % 360) if kmh > 1 else None}

    # ---- 2. scomposizione in scale (decibel): grande scala L = media su circa 20 km, piccola scala S = resto
    X = [np.where(np.isfinite(c), in_db(c), -15.0) for c in seq]
    L = [ndimage.gaussian_filter(x, SIGMA_SISTEMA) for x in X]
    Sm = [x - l for x, l in zip(X, L)]
    tau_tot, r30 = vita_media(seq, u_c, v_c) if len(seq) == 4 else (TAU_DEFAULT, None)
    tau_c, tau_l = 40.0, 240.0
    if len(seq) == 4:
        pS, vS = fusione.trasporta(Sm[0], 3, lambda k: (u_c, v_c), noto0)
        pL, vL = fusione.trasporta(L[0], 3, lambda k: (u_s, v_s), noto0)
        mS = vS[-1] & noto0 & ((X[-1] > -10) | (X[0] > -10))
        tau_c = _tau(_corr_scala(pS[-1], Sm[-1], mS), 30, 10, 180, 40.0)
        tau_l = _tau(_corr_scala(pL[-1], L[-1], vL[-1] & noto0 & (L[-1] > -14)), 30, 30, 720, 240.0)
    info["vita_media_min"] = round(tau_tot)
    info["vita_celle_min"] = round(tau_c)
    info["vita_sistema_min"] = round(tau_l)
    info["correlazione_30min"] = None if r30 is None or not np.isfinite(r30) else round(r30, 3)

    # ---- 3. WRF corretto punto per punto
    wrf_info = {"usato": False}
    wrf_corr, wrf_evo, corr_spost, ponte, tab_evo = {}, {}, None, None, None
    c_prior = C_DEFAULT
    if wrf_prev is not None and ponte_fn is not None:
        ponte = ponte_fn(wrf_prev)
        w0 = wrf_prev.campo(t0)
        if w0 is None:
            wrf_info = {"usato": False, "motivo": f"la corsa WRF {wrf_prev.corsa:%Y-%m-%d %H} UTC non copre l'ora attuale"}
        else:
            wrf0 = ponte.porta(w0)
            spost, info_all = fusione.allinea(radar0, wrf0, _db15, 0.0, 35.0, sigma_campo=4, sigma_spost=25, max_spost=60.0,
                                              soglia=SOGLIA)
            if spost is None:
                # ripiego: un solo spostamento per tutta la mappa (correlazione di fase), se migliora
                sp, fattore, c_g, note = correggi_wrf(radar0, wrf0)
                du = np.full(radar0.shape, sp[0])
                dv = np.full(radar0.shape, sp[1])
                info_all["ripiego_spostamento_unico_km"] = [round(sp[0], 1), round(-sp[1], 1)]
            else:
                du, dv = spost
            wrf0a = fusione.deforma(wrf0, du, dv)
            m = noto0 & np.isfinite(wrf0a)
            tab = None
            if (np.nan_to_num(radar0) >= SOGLIA)[m].mean() > 0.01 and (np.nan_to_num(wrf0a) >= SOGLIA)[m].mean() > 0.01:
                tab = fusione.tabella_quantili(radar0, wrf0a, m)
            prima, dopo = info_all.get("correlazione_prima"), info_all.get("correlazione_dopo")
            vals = [v for v in (prima, dopo) if v is not None]
            c_prior = float(np.clip(np.mean(vals) if vals else C_DEFAULT, 0.1, 0.7))
            for k in range(0, passi + 1):
                t = t0 + timedelta(minutes=k * PASSO_MIN)
                wc = wrf_prev.campo(t)
                if wc is None:
                    continue
                minuti = k * PASSO_MIN
                campo = fusione.deforma(ponte.porta(wc), du, dv, fattore=math.exp(-minuti / 360.0))
                wrf_corr[k] = fusione.applica_tabella(campo, tab, alfa=math.exp(-minuti / 240.0))
                # per l'evoluzione la correzione resta costante: cosi' la variazione nel tempo e' solo quella del modello
                wrf_evo[k] = fusione.applica_tabella(fusione.deforma(ponte.porta(wc), du, dv), tab)
            corr_spost = (du, dv)
            tab_evo = tab
            wrf_info = {"usato": True, "corsa": wrf_prev.corsa.strftime("%Y-%m-%d %H UTC"), "abilita": round(c_prior, 2),
                        "correzione_intensita": tab is not None, **info_all}
    info["wrf"] = wrf_info

    # ---- 4. movimento del WRF corretto (ogni ora), per guidare il sistema con l'avanzare delle ore
    mov_wrf = {}
    if len(wrf_corr) > 6:
        for k in range(0, passi, 6):
            if k in wrf_corr and k + 6 in wrf_corr:
                a, b = wrf_corr[k], wrf_corr[k + 6]
                if (np.nan_to_num(a) >= SOGLIA).sum() > 300 and (np.nan_to_num(b) >= SOGLIA).sum() > 300:
                    ta = ndimage.gaussian_filter(np.nan_to_num(_db15(a)), 8)
                    tb = ndimage.gaussian_filter(np.nan_to_num(_db15(b)), 8)
                    uw, vw = fusione.flusso(ta, tb, 0.0, 35.0)
                    peso = ((ta > 2) | (tb > 2)).astype("float64")
                    mov_wrf[k] = (fusione.liscia_pesata(uw, peso, 40) / 6.0, fusione.liscia_pesata(vw, peso, 40) / 6.0)
    info["movimento_wrf_ore"] = len(mov_wrf)

    def mov_sistema(k):
        minuti = k * PASSO_MIN
        w = math.exp(-minuti / 120.0)
        ora_k = (k - 1) // 6 * 6
        if ora_k in mov_wrf:
            uw, vw = mov_wrf[ora_k]
            return w * u_s + (1 - w) * uw, w * v_s + (1 - w) * vw
        return u_s, v_s

    # ---- 5. estrapolazione a due scale e ricomposizione
    L_est, val = fusione.trasporta(L[-1], passi, mov_sistema, noto0)
    S_est, _ = fusione.trasporta(Sm[-1], passi, lambda k: (u_c, v_c), noto0)
    x0 = np.sort(X[-1][noto0])
    estr = []
    for k in range(passi):
        minuti = (k + 1) * PASSO_MIN
        x = L_est[k] + S_est[k] * math.exp(-minuti / tau_c)
        # probability matching: la distribuzione delle intensita' resta quella osservata adesso (persistenza lagrangiana)
        vv = val[k]
        if vv.sum() > 100 and x0.size > 100:
            ordine = np.argsort(x[vv], kind="stable")
            bersaglio = np.interp(np.linspace(0, x0.size - 1, ordine.size), np.arange(x0.size), x0)
            nuovo = np.empty(ordine.size)
            nuovo[ordine] = bersaglio
            x = x.copy()
            x[vv] = nuovo
        r = np.where(x > 10 * math.log10(SOGLIA), 10 ** (x / 10.0), 0.0)
        estr.append(np.where(vv, r, np.nan).astype("float32"))

    # ---- 6. EVOLUZIONE GUIDATA DAL WRF (metodo principale, vedi evoluzione.py): si parte dal radar di adesso e lo si fa
    #         evolvere seguendo il WRF corretto (vento che spinge i temporali + crescita/calo/nascita della pioggia)
    import evoluzione
    secco = 10 * math.log10(SOGLIA) - 1.0
    db = lambda r: np.where(np.nan_to_num(r) >= SOGLIA, 10 * np.log10(np.maximum(np.nan_to_num(r), SOGLIA)), secco)  # noqa: E731
    # I campi del WRF sono medie orarie: tra un'ora e l'altra non si mescolano (una cella che si sposta "si dimezzerebbe"
    # a meta' strada) ma si fanno scorrere lungo il loro movimento (interpolazione con compensazione del moto).
    campi_db, mov_orari = {}, []
    if wrf_evo and corr_spost is not None:
        orari = []
        for c in wrf_prev.centri:
            if t0 - timedelta(hours=1) <= c <= t0 + timedelta(minutes=passi * PASSO_MIN) + timedelta(hours=1):
                campo = fusione.applica_tabella(fusione.deforma(ponte.porta(wrf_prev.campo(c)), *corr_spost), tab_evo)
                orari.append((c, np.where(np.isfinite(campo), db(campo), np.nan)))   # NaN = fuori dal WRF
        flussi = []
        for (ca, a), (cb, b) in zip(orari[:-1], orari[1:]):
            sa = ndimage.gaussian_filter(np.nan_to_num(a, nan=secco), 4)
            sb = ndimage.gaussian_filter(np.nan_to_num(b, nan=secco), 4)
            fu, fv = fusione.flusso(sa, sb, secco, 25.0, patch=21)
            pz = ((sa > secco + 3) | (sb > secco + 3)).astype("float64")
            fu, fv = fusione.liscia_pesata(fu, pz, 25), fusione.liscia_pesata(fv, pz, 25)
            flussi.append((fu, fv))
            # movimento per passo di 10 minuti: dove c'e' pioggia nel WRF quello misurato, altrove (sfumando) il vento
            # in quota del WRF o, se manca, il movimento medio
            peso = ((sa > secco + 3) | (sb > secco + 3)).astype("float64")
            tm = ca + (cb - ca) / 2
            vg = wrf_prev.vento_guida(tm)
            if vg is not None:
                gu, gv = evoluzione.vento_in_pixel(ponte_riempi(ponte.porta(vg[0])), ponte_riempi(ponte.porta(vg[1])), 1.0, 60)
            else:
                gu = np.full(sa.shape, (fu * peso).sum() / max(peso.sum(), 1.0))
                gv = np.full(sa.shape, (fv * peso).sum() / max(peso.sum(), 1.0))
            num_u = ndimage.gaussian_filter(fu * peso, 10); num_v = ndimage.gaussian_filter(fv * peso, 10)
            den = ndimage.gaussian_filter(peso, 10)
            a_ = np.clip(den / 0.3, 0, 1)
            mu = a_ * np.divide(num_u, np.maximum(den, 1e-6)) + (1 - a_) * gu
            mv = a_ * np.divide(num_v, np.maximum(den, 1e-6)) + (1 - a_) * gv
            dt_passi = (cb - ca).total_seconds() / (PASSO_MIN * 60)
            mov_orari.append((ca, cb, (mu / dt_passi).astype("float32"), (mv / dt_passi).astype("float32")))
        for k in range(0, passi + 1):
            t = t0 + timedelta(minutes=k * PASSO_MIN)
            for j in range(len(orari) - 1):
                (ca, a), (cb, b) = orari[j], orari[j + 1]
                if ca <= t <= cb:
                    f = (t - ca).total_seconds() / (cb - ca).total_seconds()
                    u, v = flussi[j]
                    pa = fusione.deforma(a, u, v, f)
                    pb = fusione.deforma(b, u, v, -(1 - f))
                    campi_db[k] = np.where(np.isfinite(pa) & np.isfinite(pb), (1 - f) * pa + f * pb,
                                           np.where(np.isfinite(pa), pa, pb))
                    break
    vento_info = "movimento misurato sul radar (il WRF non ha ancora il vento in quota)"

    # Il movimento DEVE essere lo stesso con cui il WRF sposta la sua pioggia: se la pioggia vera si muovesse con un'altra
    # velocita', la variazione presa dal WRF "nel punto di partenza" la farebbe sparire e ricomparire dove sta la pioggia
    # del WRF (un salto all'indietro). Quindi: movimento delle zone di pioggia del WRF tra un'ora e l'altra (lo stesso
    # usato per far scorrere il WRF tra le ore); dove il WRF non ha pioggia, il vento 850-500 hPa del WRF.
    def mov_wrf_passo(k):
        nonlocal vento_info
        t = t0 + timedelta(minutes=k * PASSO_MIN)
        vw = wrf_prev.vento_guida(t) if (wrf_prev is not None and ponte is not None) else None
        if vw is not None:
            vento_info = "vento del WRF tra 850 e 500 hPa (campo liscio)"
            gu, gv = evoluzione.vento_in_pixel(ponte_riempi(ponte.porta(vw[0])), ponte_riempi(ponte.porta(vw[1])), 1.0, PASSO_MIN)
            return ndimage.gaussian_filter(gu, 30), ndimage.gaussian_filter(gv, 30)
        return None

    # movimento osservato del SISTEMA (gia' lisciato su 50 km), ancora piu' liscio: nessuna deformazione locale
    # movimento misurato ogni 10 minuti sul radar, lisciato su circa 20 km: le forme si spostano intere, senza stirarsi.
    # (Verificato su dati veri del 30/10: a +30 minuti CSI 0.56 contro 0.31 della persistenza.)
    mov_oss_liscio = (ndimage.gaussian_filter(u_c, 20), ndimage.gaussian_filter(v_c, 20))
    vento_info = "movimento misurato sul radar ogni 10 minuti (lisciato)"
    # quanto pesa la variazione del WRF: accordo tra WRF corretto e radar adesso (correlazione dopo la correzione)
    acc = wrf_info.get("correlazione_dopo") if wrf_info.get("usato") else None
    peso_t = float(np.clip((acc or 0.0), 0.0, 0.9))
    info["peso_evoluzione_wrf"] = round(peso_t, 2)
    if WRF_TOTALE and wrf_evo and mov_orari:
        # PROVA "WRF TOTALE": l'ultimo scatto del radar viene rielaborato seguendo SOLO il WRF: si muove come si muove la
        # pioggia del WRF (zona per zona, anche in direzioni diverse) e cresce/cala/nasce/muore come nel WRF, a peso pieno.
        def mov_orari_passo(k):
            t = t0 + timedelta(minutes=k * PASSO_MIN - PASSO_MIN / 2)
            for ta, tb, mu, mv in mov_orari:
                if ta <= t <= tb:
                    return mu, mv
            return None
        vento_info = "WRF totale: movimento e evoluzione solo dal WRF"
        info["peso_evoluzione_wrf"] = 1.0
        info["modalita"] = "wrf_totale"
        evo, evo_val, evo_info = evoluzione.evolvi(db(radar0), noto0, campi_db, mov_oss_liscio, mov_orari_passo,
                                                   passi, PASSO_MIN, sigma=5.0, minimo=secco, massimo=23.0, tau_mov=1e-6,
                                                   peso_tendenza=1.0, max_tendenza=30.0)
    else:
        evo, evo_val, evo_info = evoluzione.evolvi(db(radar0), noto0, campi_db, mov_oss_liscio, mov_wrf_passo if wrf_evo else None,
                                                   passi, PASSO_MIN, sigma=12.0, minimo=secco, massimo=23.0, tau_mov=60.0,
                                                   peso_tendenza=peso_t, max_tendenza=12.0)
    evol = [np.where(v, np.where(x > secco + 0.5, 10 ** (x / 10.0), 0.0), np.nan).astype("float32") for x, v in zip(evo, evo_val)]
    info["evoluzione"] = {**evo_info, "movimento": vento_info}

    # ---- 7. combinazione finale: l'evoluzione guidata dal WRF e' la previsione. Solo se le verifiche dei giorni scorsi
    #         dimostrano che a una certa scadenza la semplice estrapolazione del radar fa meglio, la si mescola (pesi
    #         ottimali calcolati dall'abilita' misurata), cosi' la precisione la decidono i dati.
    fonte_pesi = "solo evoluzione guidata dal WRF"
    previsioni, pesi, comp = [np.where(noto0, np.nan_to_num(radar0), np.nan).astype("float32")], [1.0], {}
    for k in range(1, passi + 1):
        minuti = k * PASSO_MIN
        ev = evol[k - 1]
        es = estr[k - 1]
        r_v = fusione.abilita_misurata(pagella, minuti, "evoluzione")
        r_e = fusione.abilita_misurata(pagella, minuti, "estrapolazione")
        w = 0.0       # peso dell'estrapolazione
        if wrf_evo and r_v is not None and r_e is not None and r_e > r_v + 0.05:
            w = fusione.pesi_ottimali(r_e, r_v)
            fonte_pesi = "evoluzione WRF + estrapolazione dove le verifiche la danno migliore"
        if not wrf_evo:
            w = 1.0
            fonte_pesi = "solo estrapolazione (WRF non disponibile per quest'ora)"
        campo = ev if w == 0.0 else fondi(np.nan_to_num(ev), np.isfinite(ev), np.nan_to_num(es), 1 - w)
        previsioni.append(np.asarray(campo, dtype="float32"))
        pesi.append(round(1 - w, 3))
        if minuti in (30, 60, 120, 180):
            wrfc = wrf_corr.get(k)
            comp[minuti] = {"estrapolazione": es, "evoluzione": ev, "wrf": None if wrfc is None else wrfc.astype("float32")}
    info["peso_evoluzione"] = {f"+{k * PASSO_MIN}": pesi[k] for k in (1, 3, 6, 12, 18, 24, 30, 36) if k < len(pesi)}
    info["pesi"] = fonte_pesi
    return {"campi": np.stack(previsioni), "tempi": [(t0 + timedelta(minutes=k * PASSO_MIN)).strftime("%Y-%m-%dT%H:%MZ")
                                                    for k in range(passi + 1)],
            "pesi": pesi, "griglia": radar["griglia"], "info": info, "componenti": comp,
            "u": u_c.astype("float32"), "v": v_c.astype("float32")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dati", required=True)
    ap.add_argument("--wrf", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pagella", default="", help="pagella.json del giro precedente (per i pesi imparati)")
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
    pagella = None
    if a.pagella:
        try:
            import json as _j
            pagella = _j.loads(open(a.pagella).read())
        except Exception:  # noqa: BLE001
            pagella = None
    ris = calcola(radar, wrf_prev, ponte_fn, pagella)
    with open(a.out, "wb") as f:
        pickle.dump(ris, f)
    import json
    print(json.dumps(ris["info"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
