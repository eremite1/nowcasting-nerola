#!/usr/bin/env python3
"""Gara dei metodi, parte della pioggia (vedi gara.py): archivia le previsioni di adesso di tutti i metodi, confronta
quelle vecchie con il radar vero e scrive la classifica (gara.json) nella cartella pubblicata.

  gara_giro.py --dati CARTELLA_SCARICO --nowcast nowcast.pkl --precedente CARTELLA_PUBBLICATA --out CARTELLA_NUOVA

Le nubi vengono archiviate e confrontate da satellite.py (stessa cartella $GARA, stesso storico): qui si scrive solo
la classifica finale con entrambe.
"""
import argparse
import json
import pickle
from datetime import datetime, timezone

import numpy as np

import gara


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dati", required=True)
    ap.add_argument("--nowcast", required=True)
    ap.add_argument("--precedente", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.nowcast, "rb") as f:
        nc = pickle.load(f)
    with open(f"{a.dati}/radar.pkl", "rb") as f:
        radar = pickle.load(f)
    g = nc["griglia"]
    t0 = gara.tempo(nc["tempi"][0])
    G = nc.get("gara") or {}
    r0, r1, c0, c1 = G.get("ritaglio") or g.riquadro_indici(*gara.LAZIO)
    info = {"x0": g.x0, "y0": g.y0, "ritaglio": [int(r0), int(r1), int(c0), int(c1)], "passo_km": abs(g.dx) / 1000.0}
    if G.get("campi"):
        gara.archivia("pioggia", t0, G["oss0"], G["campi"], info)
    osservati = {gara.tempo(t): np.asarray(c, dtype="float32")[r0:r1, c0:c1] for t, c in zip(radar["tempi"], radar["campi"])}
    nuovi, st = gara.verifica("pioggia", osservati, info, t0, a.precedente)
    scelta = (nc.get("info") or {}).get("metodo_pubblicato")
    ris = gara.scrivi_classifica(st, t0, a.out, scelta)
    sette = ris["periodi"]["7"].get("pioggia", {}).get("metodi", [])
    print(f"gara pioggia: {nuovi} confronti nuovi, metodi archiviati {len(G.get('campi') or {})}, "
          f"in testa (7 giorni): {', '.join('%s %s%%' % (r['nome'], r['voto_medio']) for r in sette[:3]) or 'nessuno ancora'}")
    print("pubblicato:", json.dumps(scelta, ensure_ascii=False))


if __name__ == "__main__":
    print("gara:", datetime.now(timezone.utc).strftime("%H:%M"))
    main()
