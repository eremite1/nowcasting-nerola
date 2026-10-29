"""Legge la previsione del modello WRF di Nerola (progetto wrf-nerola) e la porta sulla griglia del radar.

Il WRF pubblica, nel ramo "risultati" del suo repository, il file d01_tutto.npz (dominio a 3 km, un istante ogni ora)
con la pioggia accumulata dall'inizio: RAINC (pioggia convettiva) + RAINNC (pioggia "a grande scala").
Da qui ricaviamo l'intensita' media di ogni ora (mm/h) e la interpoliamo nel tempo e nello spazio.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay


def _tempo(s):
    return datetime.strptime(str(s), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)


class PrevisioneWRF:
    def __init__(self, percorso):
        z = np.load(percorso, allow_pickle=False)
        self.lat = np.asarray(z["lat"], dtype="float64")
        self.lon = np.asarray(z["lon"], dtype="float64")
        self.tempi = [_tempo(t) for t in z["tempi"]]
        acc = np.asarray(z["RAINC"], dtype="float64") + np.asarray(z["RAINNC"], dtype="float64")
        # pioggia di ogni ora (mm in un'ora = mm/h medi), attribuita a meta' ora
        tassi = np.clip(np.diff(acc, axis=0), 0, None)
        self.centri = [self.tempi[i] + (self.tempi[i + 1] - self.tempi[i]) / 2 for i in range(len(self.tempi) - 1)]
        durate = np.array([(self.tempi[i + 1] - self.tempi[i]).total_seconds() / 3600 for i in range(len(self.tempi) - 1)])
        self.tassi = tassi / durate[:, None, None]
        self.corsa = self.tempi[0]
        self.nubi = self._nuvolosita(z) if all(k in z.files for k in ("GLW", "T2", "Q2", "PSFC")) else None
        # temperatura, umidita' e pressione al suolo (per il tipo di precipitazione, vedi tipo.py)
        self._t2 = np.asarray(z["T2"], dtype="float32") if "T2" in z.files else None
        self._q2 = np.asarray(z["Q2"], dtype="float32") if "Q2" in z.files else None
        self._ps = np.asarray(z["PSFC"], dtype="float32") if "PSFC" in z.files else None
        self.u10 = np.asarray(z["U10"], dtype="float64") if "U10" in z.files else None
        self.v10 = np.asarray(z["V10"], dtype="float64") if "V10" in z.files else None

    @staticmethod
    def _nuvolosita(z):
        """Nuvolosita' (0-1) del WRF ricavata dalla radiazione infrarossa che arriva al suolo (GLW).
        Con il cielo sereno l'aria irraggia poco: emissivita' di Brutsaert (1975), eps = 1,24 (e/T)^(1/7), con e la
        pressione del vapore in hPa e T in kelvin. Una nube irraggia come un corpo nero alla temperatura dell'aria.
        Quindi GLW = (1-c) eps sigma T^4 + c sigma T^4  ->  c = (GLW - eps sigma T^4) / (sigma T^4 (1 - eps)).
        (Metodo simile a Crawford e Duchon 1999.) Valori per ogni istante del modello (non medie orarie)."""
        sigma = 5.670374e-8
        T = np.asarray(z["T2"], dtype="float64")
        q = np.asarray(z["Q2"], dtype="float64")
        p = np.asarray(z["PSFC"], dtype="float64") / 100.0
        e = q * p / (0.622 + 0.378 * q)
        eps = np.clip(1.24 * (np.maximum(e, 0.01) / T) ** (1 / 7), 0.4, 0.99)
        nero = sigma * T ** 4
        c = (np.asarray(z["GLW"], dtype="float64") - eps * nero) / (nero * (1 - eps))
        return np.clip(c, 0, 1)

    def nuvolosita(self, t):
        """Nuvolosita' prevista dal WRF all'istante t (interpolazione lineare tra gli istanti del modello), None se fuori."""
        if self.nubi is None or not (self.tempi[0] <= t <= self.tempi[-1]):
            return None
        s = np.array([(c - self.tempi[0]).total_seconds() for c in self.tempi])
        x = (t - self.tempi[0]).total_seconds()
        i = int(np.clip(np.searchsorted(s, x) - 1, 0, len(s) - 2))
        w = (x - s[i]) / (s[i + 1] - s[i])
        return self.nubi[i] * (1 - w) + self.nubi[i + 1] * w

    def copre(self, t):
        return self.centri[0] <= t <= self.centri[-1]

    def campo(self, t):
        """Intensita' di pioggia (mm/h) prevista dal WRF all'istante t (interpolazione lineare tra le ore), None se fuori."""
        if not self.copre(t):
            return None
        s = np.array([(c - self.centri[0]).total_seconds() for c in self.centri])
        x = (t - self.centri[0]).total_seconds()
        i = int(np.clip(np.searchsorted(s, x) - 1, 0, len(s) - 2))
        w = (x - s[i]) / (s[i + 1] - s[i])
        return self.tassi[i] * (1 - w) + self.tassi[i + 1] * w


class Ponte:
    """Interpolazione dalla griglia del WRF (3 km) ai punti della griglia del radar (1 km). Fuori dal WRF: NaN."""

    def __init__(self, wrf, lon_radar, lat_radar):
        punti = np.column_stack([wrf.lon.ravel(), wrf.lat.ravel()])
        self.tri = Delaunay(punti)
        self.forma = lon_radar.shape
        self.dest = np.column_stack([lon_radar.ravel(), lat_radar.ravel()])
        # il bordo del WRF ha la "zona di rilassamento" (5 punti) dove il modello e' forzato dai dati esterni: la escludiamo
        interno = np.zeros(wrf.lat.shape)
        interno[5:-5, 5:-5] = 1.0
        self.dentro = LinearNDInterpolator(self.tri, interno.ravel(), fill_value=0.0)(self.dest).reshape(self.forma) > 0.999

    def porta(self, campo):
        v = LinearNDInterpolator(self.tri, np.asarray(campo).ravel(), fill_value=np.nan)(self.dest).reshape(self.forma)
        v[~self.dentro] = np.nan
        return v


def ora(t):
    return t.strftime("%Y-%m-%d %H:%M UTC")


__all__ = ["PrevisioneWRF", "Ponte", "ora", "timedelta"]
