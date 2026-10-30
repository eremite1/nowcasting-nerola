#!/usr/bin/env python3
"""Allerte automatiche per Nerola: pioggia in arrivo, inizio pioggia, neve, temporale vicino, grandine.

  notifiche.py --nuovo CARTELLA_NUOVA --precedente CARTELLA_PUBBLICATA

Canali (tutti gratuiti):
  - ntfy (https://ntfy.sh): app gratuita per telefono, basta iscriversi all'argomento. Nessuna registrazione.
    Argomento: variabile NTFY_TOPIC (predefinito "meteonerola-allerte").
  - Telegram: un canale pubblico; serve un "bot" (variabili TELEGRAM_TOKEN e TELEGRAM_CHAT, segreti del repository).
Per non disturbare: ogni tipo di allerta viene mandato al massimo una volta ogni 3 ore, e "pioggia in arrivo" solo
quando la pioggia e' prevista entro 60 minuti. Lo stato (cosa e' gia' stato mandato e quando) e' in notifiche.json.
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

PAUSA_ORE = 3
PAGINA = "https://eremite1.github.io/nowcasting-nerola/"


def manda_ntfy(titolo, testo, priorita, tag):
    topic = os.environ.get("NTFY_TOPIC", "meteonerola-allerte")
    if not topic:
        return False
    q = urllib.parse.urlencode({"title": titolo, "priority": priorita, "tags": tag, "click": PAGINA})
    rich = urllib.request.Request(f"https://ntfy.sh/{urllib.parse.quote(topic)}?{q}", data=testo.encode("utf-8"), method="POST")
    try:
        with urllib.request.urlopen(rich, timeout=20) as r:
            return r.status == 200
    except Exception as e:  # noqa: BLE001
        print("ntfy non riuscito:", e, file=sys.stderr)
        return False


def manda_telegram(titolo, testo):
    token, chat = os.environ.get("TELEGRAM_TOKEN", ""), os.environ.get("TELEGRAM_CHAT", "")
    if not token or not chat:
        return False
    dati = urllib.parse.urlencode({"chat_id": chat, "text": f"{titolo}\n{testo}\n{PAGINA}", "disable_web_page_preview": "true"}).encode()
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=dati, timeout=20) as r:
            return r.status == 200
    except Exception as e:  # noqa: BLE001
        print("Telegram non riuscito:", e, file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nuovo", required=True)
    ap.add_argument("--precedente", default="")
    ap.add_argument("--prova", action="store_true", help="non manda niente, scrive solo cosa manderebbe")
    a = ap.parse_args()
    nuovo = Path(a.nuovo)
    try:
        n = json.loads((nuovo / "satellite.json").read_text())["nerola"]
    except Exception as e:  # noqa: BLE001
        print("nessun riassunto di Nerola:", e)
        return
    stato = {}
    if a.precedente and (Path(a.precedente) / "notifiche.json").exists():
        try:
            stato = json.loads((Path(a.precedente) / "notifiche.json").read_text())
        except Exception:  # noqa: BLE001
            stato = {}
    adesso = datetime.now(timezone.utc)
    inviate = stato.get("inviate", {})

    def puo(tipo):
        t = inviate.get(tipo)
        return t is None or adesso - datetime.fromisoformat(t) >= timedelta(hours=PAUSA_ORE)

    eventi = []
    liv = n.get("livello_pioggia")
    prima = stato.get("livello_pioggia")
    neve = "nev" in (n.get("pioggia") or "").lower()
    if liv == "presto" and prima not in ("presto", "ora") and puo("arrivo"):
        eventi.append(("arrivo", "❄️ Neve in arrivo a Nerola" if neve else "🌧️ Pioggia in arrivo a Nerola", n["pioggia"], 4, "umbrella"))
    if liv == "ora" and prima != "ora" and puo("inizio"):
        eventi.append(("inizio", "❄️ Nevica a Nerola" if neve else "🌧️ Piove a Nerola", n["pioggia"], 3, "cloud_with_rain"))
    if n.get("livello_fulmini") == "vicino" and puo("fulmini"):
        eventi.append(("fulmini", "⚡ Temporale vicino a Nerola", n["fulmini"], 4, "zap"))
    if n.get("grandine") and puo("grandine"):
        eventi.append(("grandine", "🧊 Possibile grandine vicino a Nerola", n["grandine"], 5, "warning"))
    for tipo, titolo, testo, prio, tag in eventi:
        print("ALLERTA:", titolo, "-", testo)
        if not a.prova:
            ok1 = manda_ntfy(titolo, testo, prio, tag)
            ok2 = manda_telegram(titolo, testo)
            if ok1 or ok2:
                inviate[tipo] = adesso.isoformat()
    (nuovo / "notifiche.json").write_text(json.dumps({"livello_pioggia": liv, "inviate": inviate,
                                                      "ntfy": os.environ.get("NTFY_TOPIC", "meteonerola-allerte"),
                                                      "telegram": bool(os.environ.get("TELEGRAM_TOKEN")),
                                                      "telegram_link": os.environ.get("TELEGRAM_LINK", "")}, ensure_ascii=False))
    if not eventi:
        print("nessuna allerta da mandare")


if __name__ == "__main__":
    main()
