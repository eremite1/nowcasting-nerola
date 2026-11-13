#!/usr/bin/env python3
"""GARA DEI METODI: tanti modi diversi di prevedere le prossime 6 ore, messi a confronto con quello che il radar e il
satellite vedono davvero dopo. Vince chi ci azzecca di piu'.

Ad ogni giro (ogni 10 minuti):
  1. ARCHIVIO: per ogni metodo si salvano le previsioni a +30 minuti, +1, +2, +3, +4, +5 e +6 ore (solo sul Lazio, in
     forma compatta). L'archivio sta nella cartella $GARA, che passa da un avvio al successivo tramite la cache di GitHub
     (non va nel sito: sarebbe troppo pesante).
  2. CONFRONTO: le previsioni la cui ora e' ormai arrivata vengono sovrapposte all'immagine vera del radar (pioggia) o
     del satellite (nubi) e si calcola il punteggio.
  3. CLASSIFICA: le somme dei punteggi stanno in gara_storico.json (giorno per giorno, piccolo) e la classifica degli
     ultimi 7 e 14 giorni in gara.json, che la pagina mostra.

IL PUNTEGGIO (da 0 a 100%):
  - "voto" = Fractions Skill Score (FSS, Roberts e Lean 2008) su quadrati di circa 10 km: in ogni quadrato si guarda
    QUANTA parte e' coperta da pioggia (almeno 1 mm/h) o da nubi, nella previsione e nella realta'. 100% = identiche,
    0% = niente in comune. Rispetto al confronto punto per punto premia anche chi sbaglia di poco la posizione, come
    farebbe una persona che guarda le due immagini. E' il metodo usato dai servizi meteo per i modelli ad alta
    risoluzione e per il nowcasting.
  - "azzeccati" = indice di successo (CSI) punto per punto: su 100 punti dove pioveva nella previsione o nella realta',
    quanti erano giusti (per la pioggia ad almeno 1 e ad almeno 5 mm/h).
Un caso conta solo se c'era pioggia (o nubi) abbastanza, nella previsione o nella realta': i giorni sereni non
gonfiano i voti.
"""
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage

SCADENZE = (30, 60, 120, 180, 240, 300, 360)
LAZIO = (11.30, 14.25, 41.05, 43.20)
GIORNI_STORICO = 30

# chiave -> (nome breve, spiegazione). L'ordine e' quello della tabella sulla pagina.
METODI = {
    "pioggia": {
        "pubblicata": ("Quella sul radar", "la previsione che si vede sulla mappa radar: a ogni scadenza usa il metodo che sta vincendo"),
        "persistenza": ("Tutto fermo", "la pioggia resta dov'è adesso (il termine di paragone più banale)"),
        "traslazione": ("Solo spostamento", "tutta la pioggia scivola insieme, alla velocità media misurata sul radar"),
        "estrapolazione": ("Spostamento a due scale", "celle e fronte si muovono ognuno come misurato sul radar, le celle si attenuano"),
        "vento_wrf": ("Spinta dal vento del WRF", "la pioggia di adesso viene spostata dal vento in quota del WRF, senza crescere o calare"),
        "evoluzione": ("Radar che evolve col WRF", "la pioggia di adesso si sposta e cresce o cala come nel WRF, quanto il WRF ci sta prendendo"),
        "wrf_totale": ("Radar guidato tutto dal WRF", "la pioggia di adesso si sposta e cambia esattamente come nel WRF"),
        "wrf_corretto": ("Solo WRF corretto", "la pioggia del WRF, rimessa nel posto giusto confrontandola con il radar di adesso"),
        "wrf_grezzo": ("Solo WRF", "la pioggia del WRF così com'è, senza correzioni"),
        "media": ("Metà e metà", "50% spostamento del radar e 50% WRF corretto, a tutte le ore"),
        "sfumata_1h": ("Dal radar al WRF in 1 ora", "all'inizio quasi tutto radar, dopo un'ora già più WRF che radar"),
        "sfumata_3h": ("Dal radar al WRF in 3 ore", "come sopra ma il passaggio al WRF è più lento"),
        "pesi_ottimali": ("Pesi calcolati oggi", "il peso tra radar e WRF dipende da quanto ciascuno ci sta prendendo oggi"),
        "evol_liscia": ("Evoluzione senza dettagli imprevedibili", "come «Radar che evolve col WRF», ma più passa il tempo più sfuma i dettagli piccoli, con la stessa quantità di pioggia"),
        "evol_liscia_forte": ("Evoluzione molto sfumata", "come sopra, sfumando di più"),
        "totale_liscia": ("Guidato dal WRF, sfumato", "«Radar guidato tutto dal WRF» con la sfumatura dei dettagli piccoli"),
        "per_scadenza": ("Il migliore a ogni scadenza", "a ogni scadenza si usa il metodo che sta vincendo a quella scadenza"),
    },
    "nubi": {
        "pubblicata": ("Quella pubblicata", "la previsione delle nubi che si vede sulla pagina"),
        "persistenza": ("Tutto fermo", "le nubi restano dove sono adesso"),
        "traslazione": ("Solo spostamento", "le nubi si spostano come misurato sul satellite, senza cambiare"),
        "wrf_grezzo": ("Solo WRF", "la nuvolosità del WRF così com'è"),
        "media": ("Metà e metà", "50% spostamento del satellite e 50% WRF"),
        "sfumata_1h": ("Dal satellite al WRF in 1 ora", "all'inizio quasi tutto satellite, poi sempre più WRF"),
        "sfumata_3h": ("Dal satellite al WRF in 3 ore", "come sopra ma il passaggio al WRF è più lento"),
    },
}
# soglie e scala del confronto
REGOLE = {
    "pioggia": {"soglia": 1.0, "soglia_forte": 5.0, "finestra_km": 10.0, "minimo_area": 0.003},
    "nubi": {"soglia": 0.5, "soglia_forte": None, "finestra_km": 10.0, "minimo_area": 0.02},
}


def tempo(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%MZ")


# ---------------------------------------------------------------------------------------- forma compatta (1 byte)

def comprimi(campo, tipo):
    """pioggia: 0 = non piove, 1-254 = decibel da -10 a +40 a passi di 0,2 dB; nubi: 0-250 = copertura 0-1.
    255 = nessun dato."""
    a = np.asarray(campo, dtype="float64")
    noto = np.isfinite(a)
    a = np.nan_to_num(a)
    if tipo == "pioggia":
        q = np.where(a >= 0.1, np.clip(np.round((10 * np.log10(np.maximum(a, 0.1)) + 10) * 5) + 1, 1, 254), 0)
    else:
        q = np.clip(np.round(a * 250), 0, 250)
    q = np.where(noto, q, 255)
    return q.astype(np.uint8)


def espandi(q, tipo):
    q = np.asarray(q)
    if tipo == "pioggia":
        v = np.where(q > 0, 10 ** (((q.astype("float64") - 1) / 5 - 10) / 10), 0.0)
    else:
        v = q.astype("float64") / 250.0
    return np.where(q == 255, np.nan, v)


# ---------------------------------------------------------------------------------------------- punteggi

def fss(prev, oss, soglia, finestra):
    """Fractions Skill Score: 1 - MSE(frazioni) / (media(Pf^2) + media(Po^2)), su finestre quadrate di 'finestra' pixel."""
    m = np.isfinite(prev) & np.isfinite(oss)
    if m.sum() < 1000:
        return None
    pf = ((np.nan_to_num(prev) >= soglia) & m).astype("float64")
    po = ((np.nan_to_num(oss) >= soglia) & m).astype("float64")
    mm = m.astype("float64")
    n = ndimage.uniform_filter(mm, finestra, mode="constant")
    ok = n > 0.5
    ff = np.divide(ndimage.uniform_filter(pf, finestra, mode="constant"), n, out=np.zeros_like(n), where=ok)
    fo = np.divide(ndimage.uniform_filter(po, finestra, mode="constant"), n, out=np.zeros_like(n), where=ok)
    ff, fo = ff[ok & m], fo[ok & m]
    rif = np.mean(ff ** 2) + np.mean(fo ** 2)
    if rif <= 0:
        return None
    return float(1 - np.mean((ff - fo) ** 2) / rif)


def csi(prev, oss, soglia):
    m = np.isfinite(prev) & np.isfinite(oss)
    p = np.nan_to_num(prev)[m] >= soglia
    o = np.nan_to_num(oss)[m] >= soglia
    tot = int((p | o).sum())
    if tot < 30:
        return None
    return float((p & o).sum() / tot)


def punteggi(prev, oss, tipo, passo_km):
    """Dizionario dei punteggi di un confronto, o None se non c'era abbastanza pioggia/nubi per giudicare."""
    R = REGOLE[tipo]
    m = np.isfinite(prev) & np.isfinite(oss)
    if m.sum() < 1000:
        return None
    area_o = float((np.nan_to_num(oss)[m] >= R["soglia"]).mean())
    area_p = float((np.nan_to_num(prev)[m] >= R["soglia"]).mean())
    if max(area_o, area_p) < R["minimo_area"]:
        return None
    finestra = max(3, int(round(R["finestra_km"] / passo_km)) | 1)
    out = {"fss": fss(prev, oss, R["soglia"], finestra), "csi": csi(prev, oss, R["soglia"]),
           "area_o": area_o, "area_p": area_p}
    if R["soglia_forte"]:
        out["csi_forte"] = csi(prev, oss, R["soglia_forte"])
    return out if out["fss"] is not None else None


# ---------------------------------------------------------------------------------------------- archivio

def cartella():
    c = Path(os.environ.get("GARA", "/tmp/gara"))
    (c / "archivio").mkdir(parents=True, exist_ok=True)
    return c


def archivia(tipo, t0, oss0, metodi, griglia_info, ritaglio=None):
    """t0: istante di partenza (datetime). oss0: osservato adesso (per la persistenza).
    metodi: {nome: {minuti: campo}} gia' sulla griglia di confronto. ritaglio: (r0, r1, c0, c1) o None."""
    sel = (lambda a: a) if ritaglio is None else (lambda a: np.asarray(a)[ritaglio[0]:ritaglio[1], ritaglio[2]:ritaglio[3]])
    dati = {"oss0": comprimi(sel(oss0), tipo)}
    for nome, per_min in metodi.items():
        for minuti, campo in per_min.items():
            if campo is not None and minuti in SCADENZE:
                dati[f"{nome}__{minuti}"] = comprimi(sel(campo), tipo)
    p = cartella() / "archivio" / f"{tipo}_{t0:%Y%m%d_%H%M}.npz"
    np.savez_compressed(p, t0=iso(t0), info=json.dumps(griglia_info), **dati)
    return p


def _storico(precedente):
    """Somme giornaliere dei punteggi: {tipo: {metodo: {minuti: {giorno: [n, fss, csi, n_csi, csi_forte, n_forte]}}}}."""
    # quello della cache (passa da un avvio all'altro) o quello pubblicato: il piu' recente dei due
    scelti = []
    for c in (cartella() / "gara_storico.json", Path(precedente) / "gara_storico.json" if precedente else None):
        if c and c.exists():
            try:
                d = json.loads(c.read_text())
                scelti.append((d.get("_aggiornato", ""), d))
            except Exception:  # noqa: BLE001
                pass
    return max(scelti, key=lambda x: x[0])[1] if scelti else {}


def verifica(tipo, osservati, griglia_info, adesso, precedente=""):
    """Confronta le previsioni archiviate con le osservazioni. osservati: {datetime: campo} sulla stessa griglia
    (gia' ritagliata come l'archivio). Aggiorna lo storico e ritorna quanti confronti nuovi."""
    st = _storico(precedente)
    fatti = st.setdefault("_fatti", {}).setdefault(tipo, [])
    fatti_set = set(fatti)
    passo_km = float(griglia_info.get("passo_km", 1.0))
    nuovi = 0
    for p in sorted((cartella() / "archivio").glob(f"{tipo}_*.npz")):
        try:
            z = np.load(p)
            t0 = tempo(str(z["t0"]))
        except Exception:  # noqa: BLE001
            p.unlink(missing_ok=True)
            continue
        if adesso - t0 > timedelta(minutes=max(SCADENZE) + 60):
            p.unlink(missing_ok=True)             # ormai tutte le scadenze sono passate
            continue
        oss0 = espandi(z["oss0"], tipo)
        for minuti in SCADENZE:
            chiave = f"{iso(t0)}+{minuti}"
            valida = t0 + timedelta(minutes=minuti)
            if chiave in fatti_set or valida not in osservati:
                continue
            oss = np.asarray(osservati[valida], dtype="float64")
            if oss.shape != oss0.shape:
                continue
            giorno = valida.strftime("%Y-%m-%d")
            campi = {"persistenza": oss0}
            for k in z.files:
                if k.endswith(f"__{minuti}"):
                    campi[k.split("__")[0]] = espandi(z[k], tipo)
            for nome, prev in campi.items():
                s = punteggi(prev, oss, tipo, passo_km)
                if s is None:
                    continue
                v = st.setdefault(tipo, {}).setdefault(nome, {}).setdefault(str(minuti), {}).setdefault(giorno, [0, 0.0, 0.0, 0, 0.0, 0])
                v[0] += 1
                v[1] += s["fss"]
                if s.get("csi") is not None:
                    v[2] += s["csi"]
                    v[3] += 1
                if s.get("csi_forte") is not None:
                    v[4] += s["csi_forte"]
                    v[5] += 1
            fatti.append(chiave)
            fatti_set.add(chiave)
            nuovi += 1
    # pulizia: giorni vecchi e confronti gia' fatti di cui non serve piu' ricordarsi
    limite = (adesso - timedelta(days=GIORNI_STORICO)).strftime("%Y-%m-%d")
    for t, metodi in st.items():
        if t.startswith("_") or not isinstance(metodi, dict):
            continue
        for m in metodi.values():
            for mm in m.values():
                for g in [g for g in mm if g < limite]:
                    del mm[g]
    lim_f = iso(adesso - timedelta(hours=9))
    st["_fatti"][tipo] = [f for f in fatti if f[:16] >= lim_f]
    st["_aggiornato"] = iso(adesso)
    (cartella() / "gara_storico.json").write_text(json.dumps(st, separators=(",", ":")))
    return nuovi, st


def classifica(st, adesso, giorni):
    """{tipo: {"metodi": [...], ...}} con i voti medi per scadenza negli ultimi 'giorni' giorni."""
    da = (adesso - timedelta(days=giorni)).strftime("%Y-%m-%d")
    out = {}
    for tipo, metodi in st.items():
        if tipo.startswith("_") or not isinstance(metodi, dict):
            continue
        righe = []
        for nome, per_min in metodi.items():
            voce = {"chiave": nome, "nome": METODI.get(tipo, {}).get(nome, (nome, ""))[0],
                    "spiegazione": METODI.get(tipo, {}).get(nome, ("", ""))[1], "scadenze": {}}
            tot_f, tot_n = 0.0, 0
            for minuti in SCADENZE:
                gg = [v for g, v in per_min.get(str(minuti), {}).items() if g >= da]
                n = sum(v[0] for v in gg)
                if n == 0:
                    continue
                f = sum(v[1] for v in gg) / n
                nc = sum(v[3] for v in gg)
                nf = sum(v[5] for v in gg)
                voce["scadenze"][str(minuti)] = {
                    "voto": round(100 * f), "casi": n,
                    "azzeccati": round(100 * sum(v[2] for v in gg) / nc) if nc else None,
                    "azzeccati_forte": round(100 * sum(v[4] for v in gg) / nf) if nf else None}
                tot_f += f * n
                tot_n += n
            if tot_n:
                voce["voto_medio"] = round(100 * tot_f / tot_n, 1)
                voce["casi"] = tot_n
                righe.append(voce)
        righe.sort(key=lambda r: -r["voto_medio"])
        # il migliore di ogni scadenza
        migliori = {}
        for minuti in SCADENZE:
            cand = [(r["scadenze"][str(minuti)]["voto"], r["chiave"]) for r in righe
                    if str(minuti) in r["scadenze"] and r["chiave"] != "pubblicata"]
            if cand:
                migliori[str(minuti)] = max(cand)[1]
        out[tipo] = {"metodi": righe, "migliori": migliori}
    return out


def scrivi_classifica(st, adesso, cartella_out, scelta=None):
    ris = {"aggiornata": iso(adesso), "scadenze": list(SCADENZE),
           "regole": {"pioggia": "pioggia di almeno 1 mm/h, quadrati di 10 km", "nubi": "cielo coperto per piu' di meta', quadrati di 10 km"},
           "periodi": {str(g): classifica(st, adesso, g) for g in (7, 14)}}
    if scelta:
        ris["scelta"] = scelta
    Path(cartella_out, "gara.json").write_text(json.dumps(ris, ensure_ascii=False, separators=(",", ":")))
    Path(cartella_out, "gara_storico.json").write_text(json.dumps(st, separators=(",", ":")))
    return ris


# --------------------------------------------------------------------------------- scelta del metodo vincente

CASI_MINIMI = 60          # confronti con pioggia (somma di tutte le scadenze) prima di fidarsi della classifica
MARGINE = 2.0             # punti di voto in piu' necessari per cambiare metodo


def migliore(precedente, tipo="pioggia", predefinito="evoluzione"):
    """Metodo da pubblicare: quello con il voto medio piu' alto negli ultimi 14 giorni, se ha abbastanza casi e batte
    il metodo predefinito di almeno MARGINE punti. Ritorna (chiave, motivo)."""
    try:
        g = json.loads(Path(precedente, "gara.json").read_text())
        righe = g["periodi"]["14"][tipo]["metodi"]
    except Exception:  # noqa: BLE001
        return predefinito, "classifica non ancora disponibile"
    validi = [r for r in righe if r["chiave"] not in ("pubblicata", "persistenza") and r.get("casi", 0) >= CASI_MINIMI
              and len(r["scadenze"]) == len(SCADENZE)]
    if not validi:
        return predefinito, f"servono almeno {CASI_MINIMI} confronti con pioggia per ogni metodo"
    vinc = max(validi, key=lambda r: r["voto_medio"])
    base = next((r for r in validi if r["chiave"] == predefinito), None)
    if base is None or vinc["chiave"] == predefinito:
        return vinc["chiave"], f"voto medio {vinc['voto_medio']}% su {vinc['casi']} confronti"
    if vinc["voto_medio"] >= base["voto_medio"] + MARGINE:
        return vinc["chiave"], f"{vinc['voto_medio']}% contro {base['voto_medio']}% del metodo predefinito"
    return predefinito, f"nessun metodo batte il predefinito di almeno {MARGINE:g} punti"


def migliori_per_scadenza(precedente, tipo="pioggia", predefinito="evoluzione"):
    """Per ogni scadenza, il metodo con il voto piu' alto negli ultimi 14 giorni (con abbastanza casi). Ritorna ({minuti: metodo}, motivo) oppure None."""
    try:
        g = json.loads(Path(precedente, "gara.json").read_text())
        righe = g["periodi"]["14"][tipo]["metodi"]
    except Exception:  # noqa: BLE001
        return None
    base = next((r for r in righe if r["chiave"] == predefinito), None)
    if base is None:
        return None
    mappa, cambi = {}, []
    for sc in SCADENZE:
        s = str(sc)
        vb = (base["scadenze"].get(s) or {}).get("voto")
        migliore, voto = predefinito, vb if vb is not None else -1
        for r in righe:
            if r["chiave"] in ("pubblicata", "persistenza", "per_scadenza", predefinito):
                continue
            x = r["scadenze"].get(s) or {}
            if x.get("casi", 0) >= CASI_MINIMI // len(SCADENZE) and x.get("voto") is not None and x["voto"] > voto:
                migliore, voto = r["chiave"], x["voto"]
        mappa[sc] = migliore
        if migliore != predefinito:
            cambi.append(f"+{sc} min: {METODI[tipo].get(migliore, (migliore,))[0]} {voto}% (contro {vb}%)")
    if not cambi:
        return None
    return mappa, "il migliore a ogni scadenza: " + "; ".join(cambi)


# --------------------------------------------------------------------------------------------- nubi (satellite)

def nubi_giro(mezza, seq, t_ult, prev_nubi, wrf_prev, lon, lat, passo_km, precedente=""):
    """Gara delle nubi, chiamata da satellite.py. mezza: {istante: (copertura 0-1, luminosita')} sul Lazio a meta'
    definizione; seq: istanti usati per il movimento (ogni 10 minuti, l'ultimo e' t_ult); prev_nubi: previsione
    pubblicata {istante: (copertura, luminosita')}. Il confronto si fa a un quarto di definizione (circa 1 km)."""
    import cv2
    import nowcast
    import previsione_nubi

    def giu(a):
        a = np.asarray(a, dtype="float32")
        noto = np.isfinite(a).astype("float32")
        h, w = a.shape
        v = cv2.resize(np.nan_to_num(a), (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        n = cv2.resize(noto, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        return np.where(n > 0.5, v / np.maximum(n, 1e-6), np.nan)

    alfa0 = np.asarray(mezza[t_ult][0], dtype="float64")
    passi = max(SCADENZE) // 10
    u, v, _ = previsione_nubi.movimento([mezza[t] for t in seq])
    est, val = nowcast.estrapola(np.where(np.isfinite(alfa0), alfa0, np.nan), u, v, passi)
    ponte = None
    if wrf_prev is not None:
        import wrf as wrf_mod
        ponte = wrf_mod.Ponte(wrf_prev, lon, lat)

    def cw(minuti):
        if ponte is None:
            return None
        c = wrf_prev.nuvolosita(t_ult + timedelta(minutes=minuti))
        return None if c is None else np.clip(ponte.porta(c), 0, 1)

    def misto(a, c, w):
        if c is None:
            return None
        return np.where(np.isfinite(a), w * np.nan_to_num(a) + (1 - w) * np.nan_to_num(c), c)

    metodi = {}
    for minuti in SCADENZE:
        k = minuti // 10
        a = np.where(val[k - 1], est[k - 1], np.nan)
        c = cw(minuti)
        tp = t_ult + timedelta(minutes=minuti)
        voci = {"traslazione": a, "wrf_grezzo": c, "media": misto(a, c, 0.5),
                "sfumata_1h": misto(a, c, math.exp(-minuti / 60.0)), "sfumata_3h": misto(a, c, math.exp(-minuti / 180.0)),
                "pubblicata": prev_nubi[tp][0] if tp in prev_nubi else None}
        for nome, campo in voci.items():
            metodi.setdefault(nome, {})[minuti] = None if campo is None else giu(campo)
    info = {"passo_km": passo_km * 2}
    archivia("nubi", t_ult, giu(alfa0), metodi, info)
    osservati = {t: giu(al) for t, (al, _) in mezza.items()}
    nuovi, _ = verifica("nubi", osservati, info, t_ult, precedente)
    return nuovi


__all__ = ["archivia", "verifica", "scrivi_classifica", "migliore", "comprimi", "espandi", "METODI", "SCADENZE", "math"]
