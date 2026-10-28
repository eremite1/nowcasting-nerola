"""Scarica i prodotti della piattaforma Radar-DPC della Protezione Civile (dati in libero accesso).

Documentazione: https://dpc-radar.readthedocs.io/it/radar_v2/api.html
  GET  /findLastProductByType?type=SRI          -> ultimo istante disponibile (millisecondi dal 1970, UTC)
  POST /downloadProduct {productType, productDate} -> indirizzo temporaneo del file GeoTIFF

Prodotti usati:
  SRI     intensita' di pioggia al suolo (mm/h), ogni 5 minuti
  IR_108  satellite, canale infrarosso a 10,8 micron (nubi e loro temperatura)
"""
import json
import time
import urllib.error
import urllib.request

BASI = ["https://radar-api.protezionecivile.it", "https://radar-api.protezionecivile.it/wide/product"]
INTESTAZIONI = {
    "User-Agent": "nowcasting-nerola (www.meteonerola.it)",
    "Origin": "https://radar.protezionecivile.it",
    "Referer": "https://radar.protezionecivile.it/",
    "Accept": "application/json",
}


class NonTrovato(Exception):
    pass


def _richiesta(url, corpo=None, tentativi=3, timeout=60):
    ultimo = None
    for n in range(tentativi):
        try:
            dati = json.dumps(corpo).encode() if corpo is not None else None
            h = dict(INTESTAZIONI)
            if dati is not None:
                h["Content-Type"] = "application/json"
            rich = urllib.request.Request(url, data=dati, headers=h, method="POST" if dati is not None else "GET")
            with urllib.request.urlopen(rich, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise NonTrovato(url) from e
            ultimo = f"HTTP {e.code}: {e.read()[:200]!r}"
        except Exception as e:  # noqa: BLE001
            ultimo = repr(e)
        time.sleep(3 * (n + 1))
    raise RuntimeError(f"{url}: {ultimo}")


_base_buona = None


def _con_basi(funzione):
    """Prova i due indirizzi conosciuti dell'API (il nuovo e quello vecchio) e ricorda quello che funziona."""
    global _base_buona
    elenco = [_base_buona] if _base_buona else BASI
    errori = []
    for b in elenco:
        try:
            r = funzione(b)
            _base_buona = b
            return r
        except NonTrovato:
            if _base_buona:
                raise
            errori.append(f"{b}: non trovato")
        except Exception as e:  # noqa: BLE001
            errori.append(f"{b}: {e}")
    if all("non trovato" in e for e in errori):
        raise NonTrovato("; ".join(errori))
    raise RuntimeError("; ".join(errori))


def ultimo(tipo):
    """Ritorna (istante in millisecondi, periodo in minuti) dell'ultimo prodotto disponibile."""
    def f(base):
        r = json.loads(_richiesta(f"{base}/findLastProductByType?type={tipo}"))
        p = r["lastProducts"][0]
        periodo = p.get("period", "PT5M")
        minuti = 5
        if periodo.startswith("PT") and periodo.endswith("M"):
            minuti = int(periodo[2:-1])
        elif periodo.startswith("PT") and periodo.endswith("H"):
            minuti = 60 * int(periodo[2:-1])
        return int(p["time"]), minuti
    return _con_basi(f)


def scarica(tipo, istante_ms):
    """Bytes del GeoTIFF del prodotto `tipo` all'istante dato. Solleva NonTrovato se quel file non esiste."""
    def f(base):
        r = json.loads(_richiesta(f"{base}/downloadProduct", {"productType": tipo, "productDate": int(istante_ms)}))
        return r["url"]
    url = _con_basi(f)
    rich = urllib.request.Request(url, headers={"User-Agent": INTESTAZIONI["User-Agent"]})
    for n in range(3):
        try:
            with urllib.request.urlopen(rich, timeout=120) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if n == 2:
                raise
            time.sleep(3)
