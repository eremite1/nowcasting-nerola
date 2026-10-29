#!/usr/bin/env python3
"""Pagella del nowcasting: confronta le previsioni fatte nelle ore precedenti con quello che il radar ha visto davvero.

  verifica.py --dati CARTELLA_SCARICO --nowcast nowcast.pkl --precedente CARTELLA_RISULTATI_VECCHI --out CARTELLA_PUBBLICA

Ad ogni giro:
  - salva in archivio/ la previsione di adesso a +30, +60, +120 e +180 minuti (solo sul Lazio, file piccoli);
  - riprende le previsioni delle ore scorse la cui ora e' ormai arrivata e le confronta con il radar;
  - aggiunge le righe a storico.csv e scrive pagella.json (medie degli ultimi 7 e 30 giorni).

Il punteggio e' l'"indice di successo" (CSI, critical success index): su 100 punti dove pioveva nella previsione o nella
realta', quanti erano giusti. 100 = perfetto, 0 = tutto sbagliato. Per avere un termine di paragone lo stesso punteggio
e' calcolato per la "persistenza", cioe' la previsione piu' banale possibile: "la pioggia resta dov'e'". Il nowcasting
serve se fa meglio della persistenza.
"""
import argparse
import csv
import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

SCADENZE = (30, 60, 120, 180)
SOGLIE = (1.0, 5.0)                 # mm/h: pioggia moderata e forte
LAZIO = (11.30, 14.25, 41.05, 43.20)
ORE_ARCHIVIO = 3.6
GIORNI_STORICO = 60


def tempo(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)


def csi(prev, oss, soglia):
    m = np.isfinite(prev) & np.isfinite(oss)
    p = prev[m] >= soglia
    o = oss[m] >= soglia
    colpiti = int((p & o).sum())
    falsi = int((p & ~o).sum())
    mancati = int((~p & o).sum())
    tot = colpiti + falsi + mancati
    return (colpiti / tot if tot >= 20 else None), int(o.sum())


def correla(prev, oss):
    """Correlazione tra previsione e osservazione (decibel, sfumati su circa 6 km), solo se piove abbastanza."""
    from scipy import ndimage
    m = np.isfinite(prev) & np.isfinite(oss)
    bag = ((np.nan_to_num(prev) >= 0.2) | (np.nan_to_num(oss) >= 0.2)) & m
    if bag.sum() < 400:
        return None
    db = lambda r: np.where(np.nan_to_num(r) >= 0.1, 10 * np.log10(np.maximum(np.nan_to_num(r), 0.1)), -15.0) + 15  # noqa: E731
    a = ndimage.gaussian_filter(db(prev), 6)
    b = ndimage.gaussian_filter(db(oss), 6)
    zona = ndimage.binary_dilation(bag, iterations=10) & m
    if a[zona].std() < 1e-6 or b[zona].std() < 1e-6:
        return None
    return float(np.corrcoef(a[zona], b[zona])[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dati", required=True)
    ap.add_argument("--nowcast", required=True)
    ap.add_argument("--precedente", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "archivio").mkdir(parents=True, exist_ok=True)
    with open(f"{a.dati}/radar.pkl", "rb") as f:
        radar = pickle.load(f)
    with open(a.nowcast, "rb") as f:
        nc = pickle.load(f)
    g = nc["griglia"]
    r0, r1, c0, c1 = g.riquadro_indici(*LAZIO)
    t0 = tempo(nc["tempi"][0])
    adesso = t0

    # 1) archivio: previsione di adesso (e osservazione di adesso, per la persistenza)
    campi = {f"p{s}": nc["campi"][s // 10][r0:r1, c0:c1].astype("float16") for s in SCADENZE}
    # le due componenti separate (estrapolazione del radar e WRF corretto), per imparare quanto pesarle
    for s, comp in (nc.get("componenti") or {}).items():
        if comp.get("estrapolazione") is not None:
            campi[f"e{s}"] = np.nan_to_num(comp["estrapolazione"][r0:r1, c0:c1]).astype("float16")
        if comp.get("wrf") is not None:
            campi[f"w{s}"] = np.nan_to_num(comp["wrf"][r0:r1, c0:c1], nan=0.0).astype("float16")
    campi["oss"] = nc["campi"][0][r0:r1, c0:c1].astype("float16")
    nome = f"{t0:%Y%m%d_%H%M}.npz"
    np.savez_compressed(out / "archivio" / nome, t0=nc["tempi"][0], x0=g.x0, y0=g.y0, riquadro=np.array([r0, r1, c0, c1]), **campi)

    # 2) previsioni vecchie da verificare
    oss_per_tempo = {tempo(t): c[r0:r1, c0:c1] for t, c in zip(radar["tempi"], radar["campi"])}
    precedente = Path(a.precedente) if a.precedente else None
    righe_vecchie = []
    if precedente and (precedente / "storico.csv").exists():
        with open(precedente / "storico.csv", newline="") as f:
            righe_vecchie = list(csv.DictReader(f))
    gia = {(r["partenza"], r["minuti"]) for r in righe_vecchie}
    nuove = []
    if precedente and (precedente / "archivio").is_dir():
        for p in sorted((precedente / "archivio").glob("*.npz")):
            try:
                z = np.load(p)
                tp = tempo(str(z["t0"]))
            except Exception:  # noqa: BLE001
                continue
            if adesso - tp > timedelta(hours=ORE_ARCHIVIO):
                continue
            if tp < adesso and not (out / "archivio" / p.name).exists():
                # lo ricopio nel nuovo archivio (servira' ancora per le scadenze piu' lunghe)
                (out / "archivio" / p.name).write_bytes(p.read_bytes())
            if abs(float(z["x0"]) - g.x0) > 1 or abs(float(z["y0"]) - g.y0) > 1 or list(z["riquadro"]) != [r0, r1, c0, c1]:
                continue
            for s in SCADENZE:
                chiave = (str(z["t0"]), str(s))
                valida = tp + timedelta(minutes=s)
                if chiave in gia or valida not in oss_per_tempo:
                    continue
                oss = oss_per_tempo[valida].astype("float64")
                prev = z[f"p{s}"].astype("float64")
                pers = z["oss"].astype("float64")
                riga = {"partenza": str(z["t0"]), "minuti": str(s), "valida": valida.strftime("%Y-%m-%dT%H:%MZ")}
                # correlazione (campi in decibel, sfumati su circa 6 km) di ciascuna componente con la pioggia vera
                for nome, chiave in (("fusione", f"p{s}"), ("estrapolazione", f"e{s}"), ("wrf", f"w{s}")):
                    if chiave in z.files:
                        c_ = correla(z[chiave].astype("float64"), oss)
                        riga[f"corr_{nome}"] = "" if c_ is None else f"{c_:.3f}"
                for sg in SOGLIE:
                    c_p, n_o = csi(prev, oss, sg)
                    c_s, _ = csi(pers, oss, sg)
                    riga[f"csi{sg:g}_nowcast"] = "" if c_p is None else f"{c_p:.3f}"
                    riga[f"csi{sg:g}_persistenza"] = "" if c_s is None else f"{c_s:.3f}"
                    riga[f"pixel_pioggia{sg:g}"] = str(n_o)
                nuove.append(riga)
                gia.add(chiave)
    limite = adesso - timedelta(days=GIORNI_STORICO)
    righe = [r for r in righe_vecchie if tempo(r["valida"]) >= limite] + nuove
    campi_csv = ["partenza", "minuti", "valida"] + [f"{k}{sg:g}{suf}" for sg in SOGLIE
                                                    for k, suf in (("csi", "_nowcast"), ("csi", "_persistenza"), ("pixel_pioggia", ""))] \
        + ["corr_fusione", "corr_estrapolazione", "corr_wrf"]
    with open(out / "storico.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campi_csv, extrasaction="ignore")
        w.writeheader()
        for r in sorted(righe, key=lambda r: (r["valida"], int(r["minuti"]))):
            w.writerow(r)

    # 3) pagella: medie degli ultimi 7 e 30 giorni, solo dove la media ha senso
    pagella = {"aggiornata": adesso.strftime("%Y-%m-%dT%H:%MZ"), "nuovi_confronti": len(nuove), "periodi": {}}
    for giorni in (7, 30):
        da = adesso - timedelta(days=giorni)
        per = {}
        for s in SCADENZE:
            sel = [r for r in righe if int(r["minuti"]) == s and tempo(r["valida"]) >= da]
            voce = {}
            for sg in SOGLIE:
                coppie = [(float(r[f"csi{sg:g}_nowcast"]), float(r[f"csi{sg:g}_persistenza"])) for r in sel
                          if r.get(f"csi{sg:g}_nowcast") and r.get(f"csi{sg:g}_persistenza")]
                if coppie:
                    voce[f"{sg:g}"] = {"nowcast": round(100 * float(np.mean([c[0] for c in coppie]))),
                                       "persistenza": round(100 * float(np.mean([c[1] for c in coppie]))), "casi": len(coppie)}
            per[str(s)] = voce
        pagella["periodi"][str(giorni)] = per
    # abilita' delle componenti (ultimi 14 giorni): serve a nowcast.py per scegliere i pesi
    da = adesso - timedelta(days=14)
    abil = {}
    for s in SCADENZE:
        sel = [r for r in righe if int(r["minuti"]) == s and tempo(r["valida"]) >= da
               and r.get("corr_estrapolazione") and r.get("corr_wrf")]
        if sel:
            abil[str(s)] = {"estrapolazione": round(float(np.mean([float(r["corr_estrapolazione"]) for r in sel])), 3),
                            "wrf": round(float(np.mean([float(r["corr_wrf"]) for r in sel])), 3),
                            "fusione": round(float(np.mean([float(r["corr_fusione"]) for r in sel if r.get("corr_fusione")] or [0])), 3),
                            "casi": len(sel)}
    pagella["abilita_componenti"] = abil
    (out / "pagella.json").write_text(json.dumps(pagella, ensure_ascii=False, indent=1))
    print(f"confronti nuovi: {len(nuove)}, righe nello storico: {len(righe)}, file in archivio: "
          f"{len(list((out / 'archivio').glob('*.npz')))}")


if __name__ == "__main__":
    main()
