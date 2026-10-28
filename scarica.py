#!/usr/bin/env python3
"""Scarica gli ultimi scatti del radar (pioggia, prodotto SRI) e del satellite (infrarosso, IR_108) della Protezione Civile,
li ritaglia sull'Italia centrale e li salva in file piccoli per i passi successivi.

  scarica.py --out CARTELLA [--ore-radar 2] [--ore-satellite 3]

Scrive:
  radar.pkl      pioggia (mm/h) ogni 10 minuti, ultime 2 ore, piu' la griglia
  satellite.pkl  infrarosso, ultime 3 ore (se disponibile)
  diagnosi.json  cosa e' stato trovato (utile per controllare se qualcosa non va)
"""
import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import fonti
import geo

# Riquadro del radar tenuto per il calcolo (lon_min, lon_max, lat_min, lat_max): molto piu' largo del Lazio, perche'
# in 6 ore la pioggia puo' arrivare da 300 km e oltre (mare Tirreno, Toscana, Campania, Adriatico).
RIQUADRO_RADAR = (8.3, 17.2, 38.8, 45.4)
# Riquadro del satellite (serve solo da vedere)
RIQUADRO_SAT = (10.4, 15.2, 40.3, 43.9)


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def serie(tipo, ore, passo_min, riquadro, diagnosi):
    t0, periodo = fonti.ultimo(tipo)
    passo = max(passo_min, periodo)
    istanti = [t0 - k * passo * 60000 for k in range(int(ore * 60 / passo) + 1)][::-1]
    print(f"{tipo}: ultimo {iso(t0)}, periodo {periodo} min, ne chiedo {len(istanti)} ogni {passo} min")

    def uno(ms):
        try:
            return ms, fonti.scarica(tipo, ms)
        except fonti.NonTrovato:
            return ms, None
        except Exception as e:  # noqa: BLE001
            print(f"  {tipo} {iso(ms)}: errore {e}", file=sys.stderr)
            return ms, None

    with ThreadPoolExecutor(4) as pool:
        risultati = list(pool.map(uno, istanti))
    campi, tempi, griglia, ritaglio = [], [], None, None
    for ms, dati in risultati:
        if dati is None:
            print(f"  {tipo} {iso(ms)}: mancante")
            continue
        try:
            a, g, info = geo.leggi_geotiff(dati)
        except Exception as e:  # noqa: BLE001
            print(f"  {tipo} {iso(ms)}: file non leggibile ({e})", file=sys.stderr)
            diagnosi.setdefault("errori", []).append(f"{tipo} {iso(ms)}: {e}")
            continue
        if tipo == "SRI":
            a[a < 0] = np.nan      # il radar scrive -9999 dove non vede (fuori copertura): e' "nessun dato", non "zero"
        if griglia is None:
            griglia = g
            ritaglio = g.riquadro_indici(*riquadro)
            diagnosi[tipo] = {"primo_file": info, "periodo_min": periodo, "ritaglio": list(ritaglio)}
            print(f"  griglia: {info['proiezione']}, {info['forma']}, passo {info['passo']}, nodata {info['nodata']}")
        elif (g.nx, g.ny, g.x0, g.y0) != (griglia.nx, griglia.ny, griglia.x0, griglia.y0):
            print(f"  {tipo} {iso(ms)}: griglia diversa dalle altre, scartato", file=sys.stderr)
            continue
        r0, r1, c0, c1 = ritaglio
        campi.append(a[r0:r1, c0:c1])
        tempi.append(iso(ms))
    if not campi:
        raise RuntimeError(f"nessun file {tipo} scaricato")
    r0, r1, c0, c1 = ritaglio
    g = griglia.ritaglio(r0, r1, c0, c1)
    pila = np.stack(campi).astype("float32")
    diagnosi[tipo].update({"scaricati": len(campi), "chiesti": len(istanti), "ultimo": tempi[-1],
                           "valori": [float(np.nanmin(pila)), float(np.nanpercentile(pila, 50)), float(np.nanmax(pila))]
                           if np.isfinite(pila).any() else None})
    print(f"  {tipo}: {len(campi)} su {len(istanti)}, ritaglio {pila.shape[1:]}, valori {diagnosi[tipo]['valori']}")
    return {"campi": pila, "tempi": tempi, "griglia": g, "passo_min": passo}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ore-radar", type=float, default=2.0)
    ap.add_argument("--ore-satellite", type=float, default=3.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    diagnosi = {"ora_scarico": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    inizio = time.time()
    try:
        radar = serie("SRI", a.ore_radar, 10, RIQUADRO_RADAR, diagnosi)
        with open(out / "radar.pkl", "wb") as f:
            pickle.dump(radar, f)
    except Exception as e:  # noqa: BLE001
        diagnosi["errore_radar"] = str(e)
        (out / "diagnosi.json").write_text(json.dumps(diagnosi, indent=1, default=str))
        sys.exit(f"ERRORE radar: {e}")
    try:
        sat = serie("IR_108", a.ore_satellite, 15, RIQUADRO_SAT, diagnosi)
        with open(out / "satellite.pkl", "wb") as f:
            pickle.dump(sat, f)
    except Exception as e:  # noqa: BLE001
        diagnosi["errore_satellite"] = str(e)
        print(f"ATTENZIONE satellite non disponibile: {e}", file=sys.stderr)
    diagnosi["secondi"] = round(time.time() - inizio, 1)
    (out / "diagnosi.json").write_text(json.dumps(diagnosi, indent=1, default=str))
    print(f"fatto in {diagnosi['secondi']} s")


if __name__ == "__main__":
    main()
