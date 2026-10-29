"""Tipo di precipitazione (pioggia, mista, neve, gelicidio, granuli di ghiaccio) e grandine.

METODO PRINCIPALE: COLONNA D'ARIA (colonna.py, calcolata dal WRF a tutte le quote)
  bulbo umido sotto 0 = l'acqua gela; tra 0 e 2 = la neve si mantiene; sopra 2 = la neve fonde.
  Scendendo dalla nube al suolo si somma quanta "occasione di fondere" trova la neve (area di fusione: strati con
  bulbo umido sopra 2 gradi, per il loro spessore) e, sotto lo strato caldo, quanta occasione ha la goccia di ri-gelare
  (area di rigelo: strati sotto 0). Soglie del metodo di Bourgouin (2000):
    poca fusione                          -> NEVE (se al suolo il bulbo umido non supera 2 gradi)
    fusione media                         -> MISTA
    fusione completa, poi strato freddo:  spesso -> GRANULI DI GHIACCIO, sottile con suolo sotto 0 -> GELICIDIO
    fusione completa                      -> PIOGGIA
  Quindi: strato caldo in quota e gelo solo vicino al suolo NON danno mai neve.
  Il WRF viene corretto vicino al suolo con la temperatura osservata adesso dalle stazioni della Protezione Civile
  (prodotto TEMP) e la stessa correzione si porta nelle ore di previsione.

RIPIEGO (se il WRF non ha ancora gli indici della colonna): solo il bulbo umido al suolo (neve sotto 1 grado, mista
tra 1 e 2, pioggia sopra 2).

GRANDINE: dove il radar stima una probabilita' di grandine (POH) di almeno il 50% e nello stesso punto vede pioggia
intensa (almeno 10 mm/h); solo osservata, non prevista.
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy import ndimage

PIOGGIA, MISTA, NEVE, GELICIDIO, GRANULI = 0, 1, 2, 3, 4
SOGLIA_NEVE, SOGLIA_MISTA = 1.0, 2.0     # solo per il ripiego (bulbo umido al suolo)
STRATO_SUOLO = 250.0                     # metri di colonna corretti con la temperatura osservata
SOGLIA_GRANDINE = 50.0


def bulbo_umido(t, ur):
    """Temperatura di bulbo umido (gradi C) da temperatura (gradi C) e umidita' relativa (%), Stull (2011)."""
    t = np.asarray(t, dtype="float64")
    ur = np.clip(np.asarray(ur, dtype="float64"), 5, 100)
    return (t * np.arctan(0.151977 * np.sqrt(ur + 8.313659)) + np.arctan(t + ur) - np.arctan(ur - 1.676331)
            + 0.00391838 * ur ** 1.5 * np.arctan(0.023101 * ur) - 4.686035)


def umidita_relativa(t_k, q, p_pa):
    """Umidita' relativa (%) da temperatura (K), rapporto di mescolanza (kg/kg) e pressione (Pa)."""
    e = q * (p_pa / 100.0) / (0.622 + q)
    tc = t_k - 273.15
    es = 6.112 * np.exp(17.67 * tc / (tc + 243.5))
    return np.clip(100.0 * e / es, 1, 100)


def _riempi(a):
    """Fuori dall'area del WRF si usa il valore del punto piu' vicino dentro l'area (niente scalini sul bordo)."""
    noto = np.isfinite(a)
    if noto.all() or not noto.any():
        return a
    idx = ndimage.distance_transform_edt(~noto, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def classifica(tw):
    out = np.full(np.shape(tw), PIOGGIA, dtype=np.uint8)
    out[tw <= SOGLIA_MISTA] = MISTA
    out[tw <= SOGLIA_NEVE] = NEVE
    return out


def _in_celsius(a):
    med = float(np.nanmedian(a)) if np.isfinite(a).any() else 10.0
    return a - 273.15 if med > 150 else a


class Tipo:
    """Calcola il tipo di precipitazione sulla griglia del radar per ogni istante (osservato e previsto)."""

    def __init__(self, griglia, temperatura=None, wrf_prev=None, t0=None):
        self.griglia = griglia
        self.lon, self.lat = griglia.lonlat()
        self.info = {}
        self.wrf = wrf_prev
        self.t0 = t0
        self.ponte = None
        if wrf_prev is not None and all(getattr(wrf_prev, k, None) is not None for k in ("_t2", "_q2", "_ps")):
            import wrf as wrf_mod
            self.ponte = wrf_mod.Ponte(wrf_prev, self.lon, self.lat)
        # temperatura osservata (TEMP della Protezione Civile), portata sulla griglia del radar
        self.t_oss = None
        if temperatura is not None and len(temperatura.get("campi", [])):
            import disegna
            c, r = temperatura["griglia"].indici(self.lon, self.lat)
            mappa = (np.asarray(c, dtype="float32"), np.asarray(r, dtype="float32"))
            t = _in_celsius(np.asarray(temperatura["campi"][-1], dtype="float64"))
            # il file mette 0 esatto dove non ha il dato (mare, zone senza stazioni): se gli zeri esatti sono tanti,
            # sono "nessun dato" e non zero gradi
            if np.isfinite(t).any() and (t == 0).sum() > 0.05 * np.isfinite(t).sum():
                t = np.where(t == 0, np.nan, t)
                self.info["zeri_tolti"] = True
            t = disegna.ricampiona(t, mappa)
            if np.isfinite(t).mean() > 0.2:
                # i buchi (mare, zone senza stazioni) si riempiono con il valore dei dintorni
                noto = np.isfinite(t)
                idx = ndimage.distance_transform_edt(~noto, return_distances=False, return_indices=True)
                self.t_oss = t[tuple(idx)]
                self.info["temperatura_osservata"] = temperatura["tempi"][-1]
                self.info["temperatura_min_max"] = [round(float(np.nanmin(t)), 1), round(float(np.nanmax(t)), 1)]
        self.correzione = None
        if self.t_oss is not None and self.ponte is not None and t0 is not None:
            tw = self._wrf(t0)
            if tw is not None:
                self.correzione = self.t_oss - tw[0]
        self.info["fonte"] = ("temperatura osservata + umidita' WRF" if self.t_oss is not None and self.ponte is not None
                              else "temperatura osservata (umidita' 90%)" if self.t_oss is not None
                              else "WRF" if self.ponte is not None else "non disponibile (tutto pioggia)")

    def _wrf(self, t):
        """(temperatura C, umidita' %) del WRF all'istante t sulla griglia del radar."""
        w = self.wrf
        if self.ponte is None or not (w.tempi[0] <= t <= w.tempi[-1]):
            return None
        s = np.array([(c - w.tempi[0]).total_seconds() for c in w.tempi])
        x = (t - w.tempi[0]).total_seconds()
        i = int(np.clip(np.searchsorted(s, x) - 1, 0, len(s) - 2))
        f = (x - s[i]) / (s[i + 1] - s[i])
        T = w._t2[i] * (1 - f) + w._t2[i + 1] * f
        q = w._q2[i] * (1 - f) + w._q2[i + 1] * f
        p = w._ps[i] * (1 - f) + w._ps[i + 1] * f
        ur = umidita_relativa(T, q, p)
        return _riempi(self.ponte.porta(T - 273.15)), _riempi(self.ponte.porta(ur))

    def _colonna(self, t):
        """Indici della colonna del WRF all'istante t sulla griglia del radar (dizionario) o None."""
        w = self.wrf
        if self.ponte is None or getattr(w, "_col", None) is None or not (w.tempi[0] <= t <= w.tempi[-1]):
            return None
        s = np.array([(c - w.tempi[0]).total_seconds() for c in w.tempi])
        x = (t - w.tempi[0]).total_seconds()
        i = int(np.clip(np.searchsorted(s, x) - 1, 0, len(s) - 2))
        f = (x - s[i]) / (s[i + 1] - s[i])
        return {k: _riempi(self.ponte.porta(v[i] * (1 - f) + v[i + 1] * f)) for k, v in w._col.items()}

    def campo(self, t):
        """Tipo (0 pioggia, 1 mista, 2 neve, 3 gelicidio, 4 granuli) all'istante t, o None se non si sa."""
        col = self._colonna(t)
        if col is not None:
            import colonna
            self.info["metodo"] = "colonna d'aria del WRF (bulbo umido a tutte le quote), corretta al suolo con le stazioni"
            tw_wrf_suolo = col["TW2"]
            tws = tw_wrf_suolo
            if self.t_oss is not None:
                # bulbo umido osservato al suolo (temperatura delle stazioni + umidita' del WRF)
                if not hasattr(self, "_delta"):
                    col0 = self._colonna(self.t0)
                    s0 = self._wrf(self.t0)
                    tw_oss = bulbo_umido(self.t_oss, s0[1] if s0 is not None else 90.0)
                    self._delta = tw_oss - col0["TW2"] if col0 is not None else 0.0
                tws = tw_wrf_suolo + self._delta
            afus = col["AFUS"] + (np.maximum(tws - 2, 0) - np.maximum(tw_wrf_suolo - 2, 0)) * STRATO_SUOLO
            arig = col["ARIG"]
            caldo_quota = (arig > 0) | (col["AFUS"] >= colonna.A_PIOGGIA)
            arig = np.where(caldo_quota, np.maximum(arig + (np.maximum(-tws, 0) - np.maximum(-tw_wrf_suolo, 0)) * STRATO_SUOLO, 0), arig)
            return colonna.classifica(np.maximum(afus, 0), arig, tws)
        self.info["metodo"] = "solo bulbo umido al suolo (il WRF non ha ancora gli indici della colonna d'aria)"
        tw_wrf = self._wrf(t)
        if self.t_oss is not None:
            if t <= self.t0 or tw_wrf is None:
                ur = tw_wrf[1] if tw_wrf is not None else 90.0
                T = self.t_oss
            else:
                T = tw_wrf[0] + (self.correzione if self.correzione is not None else 0.0)
                ur = tw_wrf[1]
        elif tw_wrf is not None:
            T, ur = tw_wrf
        else:
            return None
        tw = bulbo_umido(np.nan_to_num(T, nan=10.0), np.nan_to_num(ur, nan=90.0))
        return classifica(tw)


def grandine(dati_poh, radar=None, soglia=SOGLIA_GRANDINE, pioggia_minima=10.0):
    """Posizioni delle zone con probabilita' di grandine >= soglia: [{'ora','lat','lon','poh'}].
    Una zona viene accettata solo se nello stesso istante il radar vede li' pioggia intensa (almeno 10 mm/h):
    senza un temporale non puo' esserci grandine (cosi' si scartano gli echi falsi)."""
    out = []
    if not dati_poh:
        return out
    g = dati_poh["griglia"]
    radar_t = dict(zip(radar["tempi"], radar["campi"])) if radar else {}
    for t, campo in zip(dati_poh["tempi"], dati_poh["campi"]):
        a = np.asarray(campo, dtype="float64")
        if np.nanmax(a) <= 1.0 + 1e-6 and np.nanmax(a) > 0:
            a = a * 100.0          # da 0-1 a percentuale
        m = np.nan_to_num(a) >= soglia
        if not m.any():
            continue
        et, n = ndimage.label(m)
        for k in range(1, n + 1):
            zona = et == k
            if zona.sum() < 2:
                continue
            r, c = ndimage.center_of_mass(zona)
            lon, lat = g.proj.indietro(g.x0 + c * g.dx, g.y0 - r * g.dy)
            if radar:
                rc = radar_t.get(t)
                if rc is None:
                    continue
                rg = radar["griglia"]
                cc, rr = rg.indici(np.array([lon]), np.array([lat]))
                cc, rr = int(round(float(cc[0]))), int(round(float(rr[0])))
                finestra = np.nan_to_num(rc[max(0, rr - 5):rr + 6, max(0, cc - 5):cc + 6])
                if finestra.size == 0 or finestra.max() < pioggia_minima:
                    continue
            out.append({"ora": t, "lat": round(float(lat), 4), "lon": round(float(lon), 4), "poh": int(round(float(np.nanmax(a[zona])))),
                        "km2": int(zona.sum() * g.dx * g.dy / 1e6)})
    return out


__all__ = ["Tipo", "grandine", "bulbo_umido", "timedelta", "datetime", "timezone", "math"]
