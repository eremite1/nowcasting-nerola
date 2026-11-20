#!/usr/bin/env python3
"""Scarica gli ultimi scatti del radar (pioggia, prodotto SRI) e del satellite (infrarosso, IR_108) della Protezione Civile,
li ritaglia sull'Italia centrale e li salva in file piccoli per i passi successivi.

  scarica.py --out CARTELLA [--ore-radar 2] [--ore-satellite 3]

Scrive:
  radar.pkl      pioggia (mm/h) ogni 5 minuti, ultime 2 ore, piu' la griglia
  satellite.pkl  infrarosso della Protezione Civile (gradi C), ogni 5 minuti, ultime 2,5 ore: non viene disegnato,
                 serve solo a tarare in temperatura le immagini infrarosse ad alta definizione di EUMETSAT (satellite.py)
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
# Pioggia caduta (accumulo in mm) misurata dal radar nelle ultime 1, 3, 6, 12 e 24 ore (prodotti Radar-DPC)
ACCUMULI = ("SRT1", "CUM3", "CUM6", "CUM12", "CUM24")
# Altri prodotti del radar della Protezione Civile (solo l'ultimo scatto): riflettivita' massima sulla verticale (VMI) e
# riflettivita' a quota costante 2 e 5 km (CAPPI), in dBZ
PRODOTTI_DBZ = ("VMI", "CAPPI2", "CAPPI5")
# Riquadro del satellite (serve per la taratura delle temperature)
RIQUADRO_SAT = (10.4, 15.2, 40.3, 43.9)
# Tutta Italia (solo per le immagini osservate e gli accumuli della zona "Italia"), a 2 km: si ritaglia dagli stessi file
RIQUADRO_ITALIA = (6.3, 19.0, 36.3, 47.3)


def dimezza(a):
    """Da 1 a 2 km: per ogni quadratino di 2x2 pixel il valore piu' alto (i rovesci intensi non spariscono)."""
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    b = a[:h, :w].reshape(h // 2, 2, w // 2, 2)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmax(np.nanmax(b, axis=3), axis=1)


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def serie(tipo, ore, passo_min, riquadro, diagnosi, italia=None):
    """italia: dizionario da riempire con lo stesso prodotto ritagliato su tutta Italia a 2 km (stessi file scaricati)."""
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
    campi_it, rit_it = [], None
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
        if tipo in ("SRI", "POH") or tipo in ACCUMULI:
            a[a < 0] = np.nan      # valori negativi (-9999) = "nessun dato", non "zero"
        if tipo == "TEMP":
            a[(a < -60) | (a > 330)] = np.nan      # il radar scrive -9999 dove non vede (fuori copertura): e' "nessun dato", non "zero"
        if griglia is None:
            griglia = g
            ritaglio = g.riquadro_indici(*riquadro)
            if italia is not None:
                rit_it = g.riquadro_indici(*RIQUADRO_ITALIA)
            diagnosi[tipo] = {"primo_file": info, "periodo_min": periodo, "ritaglio": list(ritaglio)}
            print(f"  griglia: {info['proiezione']}, {info['forma']}, passo {info['passo']}, nodata {info['nodata']}")
        elif (g.nx, g.ny, g.x0, g.y0) != (griglia.nx, griglia.ny, griglia.x0, griglia.y0):
            print(f"  {tipo} {iso(ms)}: griglia diversa dalle altre, scartato", file=sys.stderr)
            continue
        r0, r1, c0, c1 = ritaglio
        campi.append(a[r0:r1, c0:c1])
        tempi.append(iso(ms))
        if rit_it is not None:
            i0, i1, j0, j1 = rit_it
            campi_it.append(dimezza(a[i0:i1, j0:j1]).astype("float32"))
    if not campi:
        raise RuntimeError(f"nessun file {tipo} scaricato")
    r0, r1, c0, c1 = ritaglio
    g = griglia.ritaglio(r0, r1, c0, c1)
    pila = np.stack(campi).astype("float32")
    diagnosi[tipo].update({"scaricati": len(campi), "chiesti": len(istanti), "ultimo": tempi[-1],
                           "valori": [float(np.nanmin(pila)), float(np.nanpercentile(pila, 50)), float(np.nanmax(pila))]
                           if np.isfinite(pila).any() else None})
    print(f"  {tipo}: {len(campi)} su {len(istanti)}, ritaglio {pila.shape[1:]}, valori {diagnosi[tipo]['valori']}")
    if rit_it is not None and campi_it:
        i0, i1, j0, j1 = rit_it
        gi = griglia
        h, w = campi_it[0].shape
        italia.update({"campi": np.stack(campi_it), "tempi": list(tempi), "passo_min": passo,
                       "griglia": type(gi)(w, h, gi.x0 + (j0 + 0.5) * gi.dx, gi.y0 - (i0 + 0.5) * gi.dy, 2 * gi.dx, 2 * gi.dy,
                                           gi.proj, gi.descrizione)})
    return {"campi": pila, "tempi": tempi, "griglia": g, "passo_min": passo}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ore-radar", type=float, default=2.0)
    ap.add_argument("--ore-satellite", type=float, default=2.5)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    diagnosi = {"ora_scarico": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    inizio = time.time()
    try:
        radar_it = {}
        radar = serie("SRI", a.ore_radar, 5, RIQUADRO_RADAR, diagnosi, italia=radar_it)
        with open(out / "radar.pkl", "wb") as f:
            pickle.dump(radar, f)
        if radar_it:
            with open(out / "radar_italia.pkl", "wb") as f:
                pickle.dump(radar_it, f)
    except Exception as e:  # noqa: BLE001
        diagnosi["errore_radar"] = str(e)
        (out / "diagnosi.json").write_text(json.dumps(diagnosi, indent=1, default=str))
        sys.exit(f"ERRORE radar: {e}")
    try:
        sat = serie("IR_108", a.ore_satellite, 5, RIQUADRO_SAT, diagnosi)
        with open(out / "satellite.pkl", "wb") as f:
            pickle.dump(sat, f)
    except Exception as e:  # noqa: BLE001
        diagnosi["errore_satellite"] = str(e)
        print(f"ATTENZIONE satellite non disponibile: {e}", file=sys.stderr)
    # probabilita' di grandine (POH) dell'ultima ora e temperatura al suolo (TEMP): servono per il tipo di precipitazione
    for tipo, ore, passo, nome in (("POH", 1.0, 10, "grandine.pkl"), ("TEMP", 0.0, 60, "temperatura.pkl")):
        try:
            d = serie(tipo, ore, passo, RIQUADRO_SAT, diagnosi)
            with open(out / nome, "wb") as f:
                pickle.dump(d, f)
        except Exception as e:  # noqa: BLE001
            diagnosi[f"errore_{tipo}"] = str(e)
            print(f"ATTENZIONE {tipo} non disponibile: {e}", file=sys.stderr)
    # pioggia caduta nelle ultime 1-24 ore (un solo campo per prodotto, il piu' recente); se manca, pazienza
    accumuli = {}
    for tipo in ACCUMULI:
        try:
            d_it = {}
            d = serie(tipo, 0.0, 60, RIQUADRO_SAT, diagnosi, italia=d_it)
            if d_it:
                d["italia"] = d_it
            if np.nanmax(d["campi"]) > 2000:          # valori assurdi: unita' diverse da quelle attese
                raise RuntimeError(f"valori fuori scala (massimo {np.nanmax(d['campi']):.0f})")
            accumuli[tipo] = d
        except Exception as e:  # noqa: BLE001
            diagnosi[f"errore_{tipo}"] = str(e)[:200]
            print(f"ATTENZIONE {tipo} non disponibile: {e}", file=sys.stderr)
    if accumuli:
        with open(out / "accumuli.pkl", "wb") as f:
            pickle.dump(accumuli, f)
    # altri prodotti del radar (riflettivita'): l'ultimo scatto, se c'e'
    prodotti = {}
    for tipo in PRODOTTI_DBZ:
        nomi = (tipo.replace("CAPPI", "CAPPI_"), tipo) if tipo.startswith("CAPPI") else (tipo,)
        for nome in nomi:
            try:
                d_it = {}
                d = serie(nome, 0.0, 5, RIQUADRO_SAT, diagnosi, italia=d_it)
                for dd in [d] + ([d_it] if d_it else []):
                    dd["campi"] = np.where(dd["campi"] < -30, np.nan, dd["campi"]).astype("float32")   # -9999 = niente dato
                if d_it:
                    d["italia"] = d_it
                prodotti[tipo] = d
                break
            except Exception as e:  # noqa: BLE001
                diagnosi[f"errore_{nome}"] = str(e)[:200]
                print(f"ATTENZIONE {nome} non disponibile: {e}", file=sys.stderr)
    if prodotti:
        with open(out / "prodotti.pkl", "wb") as f:
            pickle.dump(prodotti, f)
    diagnosi["secondi"] = round(time.time() - inizio, 1)
    (out / "diagnosi.json").write_text(json.dumps(diagnosi, indent=1, default=str))
    print(f"fatto in {diagnosi['secondi']} s")


if __name__ == "__main__":
    main()
