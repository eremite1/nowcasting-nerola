#!/usr/bin/env python3
"""Prova di tutta la catena SENZA internet: crea scatti radar finti (celle di pioggia che si muovono a velocita' nota),
una previsione WRF finta sulla vera griglia del WRF di Nerola, e controlla che il nowcasting ritrovi il movimento giusto.

  prove/prova_sintetica.py --wrf-vero d01_tutto.npz --out CARTELLA
"""
import argparse
import os
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import geo  # noqa: E402
import scarica  # noqa: E402

VEL = (3.0, -1.0)   # pixel (km) ogni 10 minuti: verso est e verso nord -> 18 km/h circa, da ovest-sud-ovest


def celle(nx, ny, centri, t_min):
    yy, xx = np.mgrid[0:ny, 0:nx].astype("float64")
    r = np.zeros((ny, nx))
    for cx, cy, picco, raggio in centri:
        x = cx + VEL[0] * t_min / 10
        y = cy + VEL[1] * t_min / 10
        d2 = ((xx - x) ** 2 + (yy - y) ** 2) / raggio ** 2
        r += picco * np.exp(-d2) * (1 + 0.3 * np.sin((xx - x) / 3.0) * np.cos((yy - y) / 4.0))
    r[r < 0.1] = 0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrf-vero", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tm = geo.TrasversaMercatore(12.5, 42.0)
    g = geo.Griglia(1200, 1400, -599500, 649500, 1000, 1000, tm, "prova")
    r0, r1, c0, c1 = g.riquadro_indici(*scarica.RIQUADRO_RADAR)
    gc = g.ritaglio(r0, r1, c0, c1)
    # celle a ovest di Roma, sul mare, che stanno arrivando
    cc, rr = gc.indici(np.array([12.0, 11.6, 12.5]), np.array([41.8, 42.3, 41.5]))
    centri = [(cc[0], rr[0], 25.0, 12), (cc[1], rr[1], 6.0, 30), (cc[2], rr[2], 12.0, 8)]
    t0 = datetime(2026, 10, 28, 9, 0, tzinfo=timezone.utc)
    tempi, campi = [], []
    for k in range(12, -1, -1):
        t = t0 - timedelta(minutes=10 * k)
        c = celle(gc.nx, gc.ny, centri, -10 * k).astype("float32")
        c[:40, :] = np.nan    # una fascia senza copertura radar, per provare i dati mancanti
        tempi.append(t.strftime("%Y-%m-%dT%H:%MZ"))
        campi.append(c)
    radar = {"campi": np.stack(campi), "tempi": tempi, "griglia": gc, "passo_min": 10}
    with open(out / "radar.pkl", "wb") as f:
        pickle.dump(radar, f)
    # satellite finto: temperature (K) con nubi fredde sulle celle
    gs_r = g.riquadro_indici(*scarica.RIQUADRO_SAT)
    gs = g.ritaglio(*gs_r)
    sat = []
    st = []
    for k in range(12, -1, -1):
        cs = celle(gs.nx, gs.ny, [(x - (gs.x0 - gc.x0) / 1000, y + (gs.y0 - gc.y0) / 1000, p, r * 1.6) for x, y, p, r in centri], -15 * k)
        sat.append((290 - 3.5 * cs).astype("float32"))
        st.append((t0 - timedelta(minutes=15 * k)).strftime("%Y-%m-%dT%H:%MZ"))
    with open(out / "satellite.pkl", "wb") as f:
        pickle.dump({"campi": np.stack(sat), "tempi": st, "griglia": gs, "passo_min": 15}, f)
    # WRF finto sulla vera griglia: stessa pioggia ma spostata di 15 km a sud e con meta' intensita'
    z = dict(np.load(a.wrf_vero))
    lat, lon = z["lat"], z["lon"]
    corsa = datetime(2026, 10, 28, 0, 0, tzinfo=timezone.utc)
    nt = 31
    acc = np.zeros((nt,) + lat.shape)
    tot = np.zeros(lat.shape)
    for h in range(1, nt):
        tc = corsa + timedelta(hours=h - 0.5)
        minuti = (tc - t0).total_seconds() / 60
        cc_, rr_ = gc.indici(lon, lat)
        lat_s = lat + 15 / 111.0     # il campo viene valutato 15 km piu' a nord -> la pioggia del WRF sta 15 km a sud
        cc_, rr_ = gc.indici(lon, lat_s)
        x = np.zeros(lat.shape)
        for cx, cy, picco, raggio in centri:
            xx = cx + VEL[0] * minuti / 10
            yy = cy + VEL[1] * minuti / 10
            x += 0.5 * picco * np.exp(-((cc_ - xx) ** 2 + (rr_ - yy) ** 2) / raggio ** 2)
        tot = tot + x
        acc[h] = tot
    z["RAINNC"] = acc.astype("float32")
    z["RAINC"] = np.zeros_like(acc, dtype="float32")
    z["tempi"] = np.array([(corsa + timedelta(hours=h)).strftime("%Y-%m-%d_%H:%M:%S") for h in range(nt)])
    for k in list(z):
        if k not in ("lat", "lon", "hgt", "landmask", "RAINNC", "RAINC", "tempi"):
            del z[k]
    np.savez_compressed(out / "wrf_finto.npz", **z)
    print("creati: radar.pkl, satellite.pkl, wrf_finto.npz in", out)
    print(f"movimento vero: {np.hypot(*VEL) * 6:.1f} km/h")


if __name__ == "__main__":
    main()
