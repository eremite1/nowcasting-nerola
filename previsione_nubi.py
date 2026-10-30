"""Previsione delle nubi 0-6 ore: stesso metodo della pioggia (nowcast.py), applicato alla nuvolosita' del satellite.

 1. MOVIMENTO: dagli ultimi 4 scatti (ogni 10 minuti) si calcola come si spostano le nubi (flusso ottico DIS).
 2. ESTRAPOLAZIONE: l'ultimo scatto viene trasportato avanti lungo il movimento (schema semi-lagrangiano).
 3. VITA MEDIA: quanto e' affidabile l'estrapolazione oggi (scatto di 30 minuti fa trasportato fino ad adesso e
    confrontato con quello vero).
 4. WRF CORRETTO: la nuvolosita' del WRF (dalla radiazione infrarossa al suolo, vedi wrf.py) viene confrontata con
    quella vista adesso dal satellite e corretta punto per punto (posizione e quantita', vedi fusione.py); la stessa
    correzione si applica alle ore successive, cosi' anche le nubi nuove previste dal WRF compaiono nel posto giusto.
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
    c, w_info, correzione = 0.25, {"usato": False}, None
    if wrf_prev is not None:
        try:
            import wrf as wrf_mod
            ponte = wrf_mod.Ponte(wrf_prev, lon, lat)
            cw0 = wrf_prev.nuvolosita(t_ultimo)
            if cw0 is not None:
                import fusione
                cw0 = ponte.porta(cw0)
                # correzione punto per punto del WRF sulle nubi viste adesso dal satellite (posizione e quantita')
                spost, info_all = fusione.allinea(alfa0, cw0, lambda x: np.nan_to_num(x), 0.0, 1.0, sigma_campo=3,
                                                  sigma_spost=20, max_spost=60.0 / passo_km, soglia=0.2)
                du, dv = spost if spost is not None else (np.zeros(alfa0.shape), np.zeros(alfa0.shape))
                cwa = fusione.deforma(cw0, du, dv)
                m = np.isfinite(cwa) & np.isfinite(alfa0)
                tab = fusione.tabella_quantili(np.nan_to_num(alfa0), cwa, m, limite=4.0) if m.sum() > 1000 else None
                r0 = correlazione(ndimage.gaussian_filter(np.nan_to_num(alfa0), 9), np.nan_to_num(cw0), np.isfinite(cw0) & np.isfinite(alfa0))
                r1 = correlazione(ndimage.gaussian_filter(np.nan_to_num(alfa0), 9), np.nan_to_num(cwa), m)
                vals = [v for v in (r0, r1) if np.isfinite(v)]
                c = float(np.clip(np.mean(vals) if vals else 0.05, 0.05, 0.7))
                correzione = (du, dv, tab)
                w_info = {"usato": True, "abilita": round(c, 2), "nuvolosita_wrf": round(float(np.nanmean(cw0)), 2),
                          "nuvolosita_satellite": round(float(np.nanmean(alfa0)), 2),
                          "spostamento_medio_km": None if spost is None else round(info_all.get("spostamento_medio_px", 0) * passo_km, 1),
                          "correlazione_prima": None if not np.isfinite(r0) else round(r0, 3),
                          "correlazione_dopo": None if not np.isfinite(r1) else round(r1, 3)}
        except Exception as e:  # noqa: BLE001
            w_info = {"usato": False, "errore": str(e)[:200]}
            ponte = None
    info["wrf"] = w_info
    out = {}
    # EVOLUZIONE GUIDATA DAL WRF (vedi evoluzione.py): le nubi viste adesso vengono spostate dal vento in quota del WRF
    # e lungo il percorso aumentano, diminuiscono, nascono o si dissolvono come nel WRF corretto.
    if w_info.get("usato") and correzione is not None:
        import evoluzione
        import fusione
        du, dv, tab = correzione
        orari = []
        for tm in wrf_prev.tempi:
            if t_ultimo - timedelta(hours=1) <= tm <= max(tempi_voluti) + timedelta(hours=1):
                c_ = wrf_prev.nuvolosita(tm)
                orari.append((tm, np.clip(fusione.applica_tabella(fusione.deforma(ponte.porta(c_), du, dv), tab), 0, 1)))
        flussi = []
        for (ta, a), (tb, b) in zip(orari[:-1], orari[1:]):
            fu_, fv_ = fusione.flusso(ndimage.gaussian_filter(np.nan_to_num(a), 3), ndimage.gaussian_filter(np.nan_to_num(b), 3),
                                      0.0, 1.0, patch=21)
            flussi.append((ndimage.gaussian_filter(fu_, 25), ndimage.gaussian_filter(fv_, 25)))
        def campo_wrf(t):
            for j_ in range(len(orari) - 1):
                (ta, a), (tb, b) = orari[j_], orari[j_ + 1]
                if ta <= t <= tb:
                    f = (t - ta).total_seconds() / (tb - ta).total_seconds()
                    u_, v_ = flussi[j_]
                    pa = fusione.deforma(np.nan_to_num(a), u_, v_, f)
                    pb = fusione.deforma(np.nan_to_num(b), u_, v_, -(1 - f))
                    return np.where(np.isfinite(pa) & np.isfinite(pb), (1 - f) * pa + f * pb, np.where(np.isfinite(pa), pa, pb))
            return None
        campi_w = {}
        for k in range(0, passi + 1):
            cw = campo_wrf(t_ultimo + timedelta(minutes=k * PASSO))
            if cw is not None:
                campi_w[k] = cw

        # le nubi vere si spostano con lo stesso movimento con cui il WRF sposta le sue (altrimenti la variazione presa
        # dal WRF le farebbe sparire e ricomparire altrove); dove il WRF non ha nubi, il vento 850-500 hPa del WRF
        riempi = lambda a: np.where(np.isfinite(a), a, np.nanmean(a) if np.isfinite(a).any() else 0.0)  # noqa: E731
        mov_orari = []
        for j_, ((ta, a), (tb, b)) in enumerate(zip(orari[:-1], orari[1:])):
            fu, fv = flussi[j_]
            peso = ((np.nan_to_num(a) > 0.15) | (np.nan_to_num(b) > 0.15)).astype("float64")
            vg = wrf_prev.vento_guida(ta + (tb - ta) / 2)
            passi_ora = (tb - ta).total_seconds() / (PASSO * 60)
            if vg is not None:
                gu, gv = evoluzione.vento_in_pixel(riempi(ponte.porta(vg[0])), riempi(ponte.porta(vg[1])), passo_km, PASSO)
                gu, gv = gu * passi_ora, gv * passi_ora
            else:
                gu = np.full(a.shape, (fu * peso).sum() / max(peso.sum(), 1.0))
                gv = np.full(a.shape, (fv * peso).sum() / max(peso.sum(), 1.0))
            den = ndimage.gaussian_filter(peso, 8)
            al = np.clip(den / 0.3, 0, 1)
            mu = al * ndimage.gaussian_filter(fu * peso, 8) / np.maximum(den, 1e-6) + (1 - al) * gu
            mv = al * ndimage.gaussian_filter(fv * peso, 8) / np.maximum(den, 1e-6) + (1 - al) * gv
            mov_orari.append((ta, tb, mu / passi_ora, mv / passi_ora))

        def mov_wrf(k):
            # solo il vento in quota del WRF, lisciato: le nubi si spostano come un insieme (niente strisce)
            vg = wrf_prev.vento_guida(t_ultimo + timedelta(minutes=k * PASSO))
            if vg is None:
                return None
            gu, gv = evoluzione.vento_in_pixel(riempi(ponte.porta(vg[0])), riempi(ponte.porta(vg[1])), passo_km, PASSO)
            return ndimage.gaussian_filter(gu, 30), ndimage.gaussian_filter(gv, 30)

        mov_liscio = (ndimage.gaussian_filter(u, 40), ndimage.gaussian_filter(v, 40))
        peso_t = float(np.clip(w_info.get("correlazione_dopo") or 0.0, 0.0, 0.9))
        info["peso_evoluzione_wrf"] = round(peso_t, 2)
        evo, val, ev_info = evoluzione.evolvi(np.nan_to_num(alfa0), np.isfinite(alfa0), campi_w, mov_liscio, mov_wrf, passi, PASSO,
                                              sigma=8.0, minimo=0.0, massimo=1.0, tau_mov=60.0, peso_tendenza=peso_t,
                                              max_tendenza=0.5)
        info["evoluzione"] = ev_info
        for t in tempi_voluti:
            k = int(round((t - t_ultimo).total_seconds() / 60 / PASSO))
            if k <= 0:
                continue
            a = evo[k - 1]
            lm = np.where(valido[k - 1], l_est[k - 1], 0.85).astype("float64")
            out[t] = (np.clip(a, 0, 1), np.clip(lm, 0, 1))
        return out, info
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
            if cw is not None:
                import fusione
                du, dv, tab = correzione
                minuti = k * PASSO
                cw = fusione.deforma(ponte.porta(cw), du, dv, fattore=math.exp(-minuti / 360.0))
                cw = np.clip(fusione.applica_tabella(cw, tab, alfa=math.exp(-minuti / 240.0)), 0, 1)
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
