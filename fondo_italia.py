#!/usr/bin/env python3
"""Fondo fisso della zona "Italia" del radar (si lancia una volta sola, con il workflow fondo-italia.yml).

Scrive in statici/italia/: mappa.jpg (colori naturali), mappa_grigia.jpg (fondo neutro per il radar),
confini.png e confini_bianchi.png (confini, coste, regioni e citta', trasparenti).
Stessa geometria della zona "italia" di disegna.py: longitudine e latitudine lineari sui pixel.
Confini e coste: Natural Earth (pubblico dominio).
"""
import math
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disegna  # noqa: E402

CITTA = [("Roma", 41.90, 12.50, True), ("Milano", 45.46, 9.19, True), ("Napoli", 40.85, 14.27, True), ("Torino", 45.07, 7.69, True),
         ("Palermo", 38.12, 13.36, True), ("Bari", 41.12, 16.87, True), ("Firenze", 43.77, 11.25, True), ("Bologna", 44.49, 11.34, True),
         ("Venezia", 45.44, 12.33, True), ("Genova", 44.41, 8.93, True), ("Cagliari", 39.22, 9.12, True), ("Catania", 37.50, 15.09, False),
         ("Pescara", 42.46, 14.21, False), ("Ancona", 43.62, 13.52, False), ("Perugia", 43.11, 12.39, False), ("Trieste", 45.65, 13.78, False),
         ("Bolzano", 46.50, 11.35, False), ("Trento", 46.07, 11.12, False), ("Reggio Calabria", 38.11, 15.65, False), ("Potenza", 40.64, 15.80, False),
         ("Campobasso", 41.56, 14.66, False), ("L'Aquila", 42.35, 13.40, False), ("Aosta", 45.74, 7.32, False), ("Catanzaro", 38.91, 16.59, False),
         ("Sassari", 40.73, 8.56, False), ("Lecce", 40.35, 18.17, False), ("Foggia", 41.46, 15.54, False), ("Verona", 45.44, 10.99, False)]


def figura(z):
    dpi = 100
    fig = plt.figure(figsize=(z.W / dpi, z.H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_extent([z.lon0, z.lon1, z.lat0, z.lat1], crs=ccrs.PlateCarree())
    ax.set_aspect("auto")
    ax.axis("off")
    return fig, ax


def main():
    z = disegna.Zona("italia", disegna.AREE["italia"])
    out = Path(__file__).resolve().parent / "statici" / "italia"
    out.mkdir(parents=True, exist_ok=True)
    terra = cf.NaturalEarthFeature("physical", "land", "10m")
    laghi = cf.NaturalEarthFeature("physical", "lakes", "10m")
    for nome, col_terra, col_mare, col_lago in (("mappa.jpg", "#cdd8b6", "#9fc4df", "#9fc4df"),
                                                ("mappa_grigia.jpg", "#858585", "#666b75", "#666b75")):
        fig, ax = figura(z)
        fig.patch.set_facecolor(col_mare)
        ax.set_facecolor(col_mare)
        ax.add_feature(terra, facecolor=col_terra, edgecolor="none")
        ax.add_feature(laghi, facecolor=col_lago, edgecolor="none")
        fig.savefig(out / nome, dpi=100, facecolor=col_mare, pil_kwargs={"quality": 90})
        plt.close(fig)
    regioni = cf.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines", "10m")
    stati = cf.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "10m")
    costa = cf.NaturalEarthFeature("physical", "coastline", "10m")
    for nome, linea, testo, alone in (("confini.png", "#1a1a1a", "#111111", "#ffffff"), ("confini_bianchi.png", "#f2f2f2", "#ffffff", "#000000")):
        fig, ax = figura(z)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.add_feature(regioni, facecolor="none", edgecolor=linea, linewidth=0.5, alpha=0.6)
        ax.add_feature(stati, facecolor="none", edgecolor=linea, linewidth=1.1, alpha=0.9)
        ax.add_feature(costa, facecolor="none", edgecolor=linea, linewidth=0.8, alpha=0.9)
        for n, la, lo, grande in CITTA:
            ax.plot(lo, la, "o", ms=3.5 if grande else 2.5, color=testo, mec=alone, mew=0.8, transform=ccrs.PlateCarree())
            t = ax.text(lo + 0.08, la + 0.05, n, fontsize=10 if grande else 8, fontweight="bold" if grande else "normal",
                        color=testo, transform=ccrs.PlateCarree())
            t.set_path_effects([pe.withStroke(linewidth=2.5, foreground=alone, alpha=0.85)])
        fig.savefig(out / nome, dpi=100, transparent=True)
        plt.close(fig)
    print("fondo Italia:", z.W, "x", z.H, "px in", out)


if __name__ == "__main__":
    main()
