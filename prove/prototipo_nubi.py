#!/usr/bin/env python3
"""Prototipo del satellite "nubi su mappa vera" per alcuni istanti scelti (serve solo per vedere come viene).

  prove/prototipo_nubi.py --sfondo CARTELLA_SFONDO --taratura satellite_taratura.json --out CARTELLA AAAA-MM-GGTHH:MMZ ...
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import disegna  # noqa: E402
import nubi  # noqa: E402
import satellite as sat  # noqa: E402
import sfondo as sfondo_mod  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfondo", required=True)
    ap.add_argument("--taratura", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("tempi", nargs="+")
    a = ap.parse_args()
    out = Path(a.out)
    cache = out / "cache"
    tab = json.loads(Path(a.taratura).read_text())["tabella"]
    sf = sfondo_mod.carica(a.sfondo)
    dem = sfondo_mod.carica_dem(a.sfondo)
    for chiave, cfg in disegna.AREE.items():
        z = disegna.Zona(chiave, cfg)
        q = nubi.dem_su_zona(dem, z)
        pk = nubi.passo_km(z)
        mare = q <= 0.5
        base = nubi.mappa_naturale(q, mare, pk)
        Image.fromarray((base * 255).astype(np.uint8)).save(out / f"{chiave}_base.jpg", quality=90)
        disegna.confini(z, sf, out / f"{chiave}_confini.png", massimo=45 if z.dettaglio else 20, scuro=False)
        conf = Image.open(out / f"{chiave}_confini.png").convert("RGBA")
        mx = ((z.lon - sat.BBOX[0]) / (sat.BBOX[1] - sat.BBOX[0]) * sat.LARGHEZZA - 0.5).astype("float32")
        my = ((sat.BBOX[3] - z.lat) / (sat.BBOX[3] - sat.BBOX[2]) * sat.ALTEZZA - 0.5).astype("float32")
        for s in a.tempi:
            t = datetime.strptime(s, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
            for nome in ("ir", "vis"):
                (cache / nome).mkdir(parents=True, exist_ok=True)
                if not (cache / nome / f"{t:%Y%m%d_%H%M}.png").exists():
                    sat.scarica_uno(nome, t, cache / nome)
            g = sat.leggi(cache, "ir", t)
            if g is None:
                print("manca ir", s)
                continue
            T = sat.a_temp(disegna.ricampiona(sat.sfuma(g, sat.STRATI["ir"][2]), (mx, my)), tab)
            cosz = sat.coseno_zenitale(t, z.lon, z.lat)
            gio = sat.giorno(cosz)
            v = sat.leggi(cache, "vis", t)
            R = sat.riflettanza(disegna.ricampiona(sat.sfuma(v, sat.STRATI["vis"][2]), (mx, my)), cosz) if v is not None else None
            alfa, lum = nubi.copertura(T, R, gio, pk)
            rgb = nubi.componi(base, alfa, lum, T)
            im = Image.fromarray((rgb * 255).astype(np.uint8)).convert("RGBA")
            im.alpha_composite(conf)
            im.convert("RGB").save(out / f"{chiave}_{t:%Y%m%d_%H%M}.jpg", quality=88)
            print(chiave, s, "nuvolosita' media", round(float(alfa.mean()), 2))


if __name__ == "__main__":
    main()
