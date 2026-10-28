"""Lettura dei file GeoTIFF della Protezione Civile e conversione tra coordinate della griglia e latitudine/longitudine.

Il mosaico radar nazionale e' su una griglia di 1 km in proiezione Trasversa di Mercatore (WGS84, origine 42 N 12,5 E).
Le formule della proiezione sono scritte qui (Snyder, "Map Projections - A Working Manual", USGS 1987), cosi' non
dipendiamo da librerie esterne; se nel file c'e' un'altra proiezione si usa pyproj (installato sui computer di GitHub).
"""
import io

import numpy as np

A_WGS84 = 6378137.0
F_WGS84 = 1 / 298.257223563


class TrasversaMercatore:
    """Proiezione Trasversa di Mercatore sull'ellissoide WGS84 (formule di Snyder, precise al millimetro entro pochi gradi)."""

    def __init__(self, lon0, lat0=0.0, k0=1.0, fe=0.0, fn=0.0, a=A_WGS84, f=F_WGS84):
        self.lon0, self.k0, self.fe, self.fn, self.a = np.radians(lon0), k0, fe, fn, a
        self.e2 = f * (2 - f)
        self.ep2 = self.e2 / (1 - self.e2)
        self.m0 = self._m(np.radians(lat0))

    def _m(self, phi):
        e2, a = self.e2, self.a
        e4, e6 = e2 * e2, e2 * e2 * e2
        return a * ((1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * phi)
                    + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * phi) - (35 * e6 / 3072) * np.sin(6 * phi))

    def avanti(self, lon, lat):
        """(lon, lat) in gradi -> (x, y) in metri."""
        phi = np.radians(np.asarray(lat, dtype="float64"))
        lam = np.radians(np.asarray(lon, dtype="float64"))
        e2, ep2, a, k0 = self.e2, self.ep2, self.a, self.k0
        n = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
        t = np.tan(phi) ** 2
        c = ep2 * np.cos(phi) ** 2
        aa = (lam - self.lon0) * np.cos(phi)
        m = self._m(phi)
        x = k0 * n * (aa + (1 - t + c) * aa ** 3 / 6 + (5 - 18 * t + t * t + 72 * c - 58 * ep2) * aa ** 5 / 120) + self.fe
        y = k0 * (m - self.m0 + n * np.tan(phi) * (aa ** 2 / 2 + (5 - t + 9 * c + 4 * c * c) * aa ** 4 / 24
                                                  + (61 - 58 * t + t * t + 600 * c - 330 * ep2) * aa ** 6 / 720)) + self.fn
        return x, y

    def indietro(self, x, y):
        """(x, y) in metri -> (lon, lat) in gradi."""
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        e2, ep2, a, k0 = self.e2, self.ep2, self.a, self.k0
        e4, e6 = e2 * e2, e2 * e2 * e2
        m = self.m0 + (y - self.fn) / k0
        mu = m / (a * (1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256))
        e1 = (1 - np.sqrt(1 - e2)) / (1 + np.sqrt(1 - e2))
        phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu) + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
                + (151 * e1 ** 3 / 96) * np.sin(6 * mu) + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
        c1 = ep2 * np.cos(phi1) ** 2
        t1 = np.tan(phi1) ** 2
        n1 = a / np.sqrt(1 - e2 * np.sin(phi1) ** 2)
        r1 = a * (1 - e2) / (1 - e2 * np.sin(phi1) ** 2) ** 1.5
        d = (x - self.fe) / (n1 * k0)
        phi = phi1 - (n1 * np.tan(phi1) / r1) * (d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ep2) * d ** 4 / 24
                                                 + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * ep2 - 3 * c1 ** 2) * d ** 6 / 720)
        lam = self.lon0 + (d - (1 + 2 * t1 + c1) * d ** 3 / 6
                           + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ep2 + 24 * t1 ** 2) * d ** 5 / 120) / np.cos(phi1)
        return np.degrees(lam), np.degrees(phi)


class Geografica:
    """Griglia gia' in longitudine/latitudine."""

    def avanti(self, lon, lat):
        return np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")

    def indietro(self, x, y):
        return np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")


class MercatoreWeb:
    """EPSG:3857 (sfera)."""
    R = 6378137.0

    def avanti(self, lon, lat):
        lon = np.asarray(lon, dtype="float64")
        lat = np.asarray(lat, dtype="float64")
        return np.radians(lon) * self.R, self.R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))

    def indietro(self, x, y):
        return np.degrees(np.asarray(x) / self.R), np.degrees(2 * np.arctan(np.exp(np.asarray(y) / self.R)) - np.pi / 2)


class ConPyproj:
    def __init__(self, epsg):
        self.epsg = epsg
        self._prepara()

    def _prepara(self):
        from pyproj import Transformer
        self._a = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self._i = Transformer.from_crs(self.epsg, 4326, always_xy=True)

    def __getstate__(self):
        return {"epsg": self.epsg}

    def __setstate__(self, stato):
        self.epsg = stato["epsg"]
        self._prepara()

    def avanti(self, lon, lat):
        return self._a.transform(lon, lat)

    def indietro(self, x, y):
        return self._i.transform(x, y)


class Griglia:
    """Griglia regolare: il centro della colonna c e della riga r sta in x = x0 + c*dx, y = y0 - r*dy (righe dall'alto)."""

    def __init__(self, nx, ny, x0, y0, dx, dy, proiezione, descrizione=""):
        self.nx, self.ny, self.x0, self.y0, self.dx, self.dy = int(nx), int(ny), float(x0), float(y0), float(dx), float(dy)
        self.proj = proiezione
        self.descrizione = descrizione

    def ritaglio(self, r0, r1, c0, c1):
        return Griglia(c1 - c0, r1 - r0, self.x0 + c0 * self.dx, self.y0 - r0 * self.dy, self.dx, self.dy, self.proj, self.descrizione)

    def xy(self):
        x = self.x0 + np.arange(self.nx) * self.dx
        y = self.y0 - np.arange(self.ny) * self.dy
        return np.meshgrid(x, y)

    def lonlat(self):
        x, y = self.xy()
        return self.proj.indietro(x, y)

    def indici(self, lon, lat):
        """Colonna e riga (con la virgola) dei punti (lon, lat)."""
        x, y = self.proj.avanti(lon, lat)
        return (np.asarray(x) - self.x0) / self.dx, (self.y0 - np.asarray(y)) / self.dy

    def riquadro_indici(self, lon0, lon1, lat0, lat1, margine=2):
        """Righe e colonne che coprono un riquadro lon/lat."""
        lo = np.linspace(lon0, lon1, 30)
        la = np.linspace(lat0, lat1, 30)
        LO, LA = np.meshgrid(lo, la)
        c, r = self.indici(LO, LA)
        c0 = max(0, int(np.floor(c.min())) - margine)
        c1 = min(self.nx, int(np.ceil(c.max())) + margine + 1)
        r0 = max(0, int(np.floor(r.min())) - margine)
        r1 = min(self.ny, int(np.ceil(r.max())) + margine + 1)
        return r0, r1, c0, c1

    def come_dict(self):
        return {"nx": self.nx, "ny": self.ny, "x0": self.x0, "y0": self.y0, "dx": self.dx, "dy": self.dy,
                "descrizione": self.descrizione}


# ------------------------------------------------------------------ lettura GeoTIFF

def _chiavi_geotiff(page):
    """Legge le chiavi GeoTIFF (GeoKeyDirectory) direttamente dai tag del file."""
    tags = page.tags
    out = {}
    if 34735 not in tags:
        return out
    d = [int(v) for v in tags[34735].value]
    doppi = list(tags[34736].value) if 34736 in tags else []
    testo = tags[34737].value if 34737 in tags else ""
    for i in range(1, d[3] + 1):
        chiave, dove, quanti, valore = d[4 * i:4 * i + 4]
        if dove == 0:
            out[chiave] = valore
        elif dove == 34736:
            out[chiave] = doppi[valore] if quanti == 1 else doppi[valore:valore + quanti]
        elif dove == 34737:
            out[chiave] = testo[valore:valore + quanti].rstrip("|")
    return out


def _proiezione(chiavi):
    tipo = chiavi.get(1024)  # GTModelTypeGeoKey: 1 proiettata, 2 geografica
    if tipo == 2:
        return Geografica(), "geografica (lon/lat)"
    epsg = chiavi.get(3072)  # ProjectedCSTypeGeoKey
    if epsg and epsg != 32767:
        if 32601 <= epsg <= 32660 or 32701 <= epsg <= 32760:
            zona = epsg % 100
            return TrasversaMercatore(-183 + 6 * zona, 0, 0.9996, 500000.0, 0.0 if epsg < 32700 else 1e7), f"UTM EPSG:{epsg}"
        if epsg in (3857, 900913):
            return MercatoreWeb(), "Mercatore web EPSG:3857"
        return ConPyproj(epsg), f"EPSG:{epsg} (pyproj)"
    trasf = chiavi.get(3075)  # ProjCoordTransGeoKey: 1 = Trasversa di Mercatore
    if trasf == 1:
        lon0 = chiavi.get(3088, chiavi.get(3080, 0.0))
        lat0 = chiavi.get(3089, chiavi.get(3081, 0.0))
        k0 = chiavi.get(3092, 1.0)
        fe = chiavi.get(3082, chiavi.get(3084, 0.0))
        fn = chiavi.get(3083, chiavi.get(3085, 0.0))
        return TrasversaMercatore(lon0, lat0, k0, fe, fn), f"Trasversa di Mercatore lon0={lon0} lat0={lat0} k0={k0}"
    if trasf == 7:  # Mercatore
        return MercatoreWeb(), "Mercatore"
    raise ValueError(f"proiezione del file non riconosciuta: {chiavi}")


def leggi_geotiff(dati):
    """Da bytes di un GeoTIFF ritorna (matrice float32 con NaN dove manca il dato, Griglia, info)."""
    import tifffile
    with tifffile.TiffFile(io.BytesIO(dati)) as tif:
        page = tif.pages[0]
        a = page.asarray()
        if a.ndim == 3:  # piu' bande: si tiene la prima
            a = a[..., 0] if a.shape[-1] <= 4 else a[0]
        a = a.astype("float32")
        tags = page.tags
        chiavi = _chiavi_geotiff(page)
        nodata = None
        if 42113 in tags:
            try:
                nodata = float(str(tags[42113].value).strip().strip("\x00"))
            except ValueError:
                nodata = None
        meta_gdal = str(tags[42112].value) if 42112 in tags else ""
        ny, nx = a.shape
        if 34264 in tags:  # ModelTransformationTag
            m = list(tags[34264].value)
            sx, sy, tx, ty = m[0], -m[5], m[3], m[7]
        else:
            sx, sy = list(tags[33550].value)[:2]
            tp = list(tags[33922].value)
            tx = tp[3] - tp[0] * sx
            ty = tp[4] + tp[1] * sy
    punto = chiavi.get(1025) == 2  # RasterTypeGeoKey: 2 = il valore e' nel punto (non nell'area)
    x0 = tx if punto else tx + sx / 2
    y0 = ty if punto else ty - sy / 2
    proj, descr = _proiezione(chiavi)
    griglia = Griglia(nx, ny, x0, y0, sx, sy, proj, descr)
    if nodata is not None:
        a[a == np.float32(nodata)] = np.nan
    a[~np.isfinite(a)] = np.nan
    info = {"proiezione": descr, "nodata": nodata, "forma": [ny, nx], "passo": [sx, sy], "metadati": meta_gdal[:600],
            "minimo": float(np.nanmin(a)) if np.isfinite(a).any() else None,
            "massimo": float(np.nanmax(a)) if np.isfinite(a).any() else None,
            "validi": float(np.isfinite(a).mean())}
    return a, griglia, info


def scrivi_geotiff_tm(matrice, x_sinistra, y_alto, passo, lon0=12.5, lat0=42.0, nodata=-9999.0):
    """Scrive un GeoTIFF in Trasversa di Mercatore (serve solo per le prove con dati simulati)."""
    import tifffile
    chiavi = [1, 1, 0, 7,
              1024, 0, 1, 1,          # proiettata
              1025, 0, 1, 1,          # pixel = area
              2048, 0, 1, 4326,
              3072, 0, 1, 32767,      # proiezione definita dall'utente
              3075, 0, 1, 1,          # Trasversa di Mercatore
              3088, 34736, 1, 0,
              3089, 34736, 1, 1]
    doppi = [float(lon0), float(lat0)]
    m = np.where(np.isfinite(matrice), matrice, nodata).astype("float32")
    extratags = [(33550, "d", 3, (passo, passo, 0.0), False),
                 (33922, "d", 6, (0.0, 0.0, 0.0, x_sinistra, y_alto, 0.0), False),
                 (34735, "H", len(chiavi), chiavi, False),
                 (34736, "d", len(doppi), doppi, False),
                 (42113, "s", 0, f"{nodata:g}", False)]
    buf = io.BytesIO()
    tifffile.imwrite(buf, m, extratags=extratags, compression="zlib")
    return buf.getvalue()


def _verifica():
    tm = TrasversaMercatore(12.5, 42.0)
    for lon, lat in ((12.77, 42.13), (9.0, 37.0), (18.5, 46.5)):
        x, y = tm.avanti(lon, lat)
        lo, la = tm.indietro(x, y)
        assert abs(lo - lon) < 1e-5 and abs(la - lat) < 1e-5, (lon, lat, lo, la)


if __name__ == "__main__":
    _verifica()
    tm = TrasversaMercatore(12.5, 42.0)
    print("Nerola nella griglia radar (km):", [round(v / 1000, 2) for v in tm.avanti(12.77, 42.13)])
    print("prova andata e ritorno: ok")
