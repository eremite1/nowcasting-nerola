#!/usr/bin/env python3
"""Dati live della stazione meteo di Nerola (Davis Vantage Vue, WeeWX): www.meteonerola.it/vue/raspberry.xml

  stazione.py --out CARTELLA_NUOVA

Scrive stazione.json con temperatura, umidita', punto di rugiada, pressione e tendenza, vento, raffica, pioggia di
oggi, dell'ultima ora e intensita' attuale, piu' massime e minime del giorno. Se il file non e' raggiungibile o e' vecchio
di oltre 30 minuti, i dati vengono segnati come "non aggiornati" e la pagina non li mostra come attuali.
"""
import argparse
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

URL = "https://www.meteonerola.it/vue/raspberry.xml"
MESI = {"gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6, "lug": 7, "ago": 8, "set": 9, "ott": 10, "nov": 11, "dic": 12}


def numero(testo):
    if testo is None:
        return None
    m = re.search(r"-?\d+(?:[.,]\d+)?", testo)
    return float(m.group(0).replace(",", ".")) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"fonte": "Stazione meteo di Nerola (Davis Vantage Vue)", "aggiornata": False}
    try:
        rich = urllib.request.Request(URL, headers={"User-Agent": "nowcasting-nerola"})
        with urllib.request.urlopen(rich, timeout=30) as r:
            radice = ET.fromstring(r.read())
        v = {e.tag: (e.text or "").strip() for e in radice}
        for chiave, tag in (("temperatura", "temp"), ("temp_max", "tempmax"), ("temp_min", "tempmin"), ("umidita", "ur"),
                            ("rugiada", "dewp"), ("pressione", "press"), ("tendenza_pressione", "presstend"), ("vento", "vento"),
                            ("raffica", "raffica"), ("raffica_max_oggi", "rafficamaxday"), ("pioggia_oggi", "pioggiaday"),
                            ("pioggia_ultima_ora", "pioggialasth"), ("pioggia_24h", "pioggialast24h"), ("intensita", "rainrate"),
                            ("vento_gradi", "ventodirgradi")):
            out[chiave] = numero(v.get(tag))
        out["vento_direzione"] = v.get("ventodir")
        out["ora_max"], out["ora_min"] = v.get("tempmaxhour"), v.get("tempminhour")
        m = re.match(r"(\d+)-(\w+)-(\d{4})\s+(\d+):(\d+)", v.get("dataora", ""))
        if m:
            t = datetime(int(m.group(3)), MESI.get(m.group(2).lower()[:3], 1), int(m.group(1)), int(m.group(4)), int(m.group(5)),
                         tzinfo=ZoneInfo("Europe/Rome"))
            out["ora"] = t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
            out["aggiornata"] = (datetime.now(timezone.utc) - t).total_seconds() < 1800
        print("stazione:", {k: out[k] for k in ("ora", "temperatura", "umidita", "intensita", "pioggia_oggi", "aggiornata") if k in out})
    except Exception as e:  # noqa: BLE001
        out["errore"] = str(e)[:200]
        print("ATTENZIONE: stazione non raggiungibile:", e, file=sys.stderr)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "stazione.json").write_text(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
