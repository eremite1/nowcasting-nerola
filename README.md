# nowcasting-nerola

Radar, satellite e **previsione della pioggia a 0-6 ore** sul Lazio e sulla Sabina, con mappe proprie, aggiornati da soli
ogni 10 minuti sui computer di GitHub. Fa parte di [meteonerola.it](https://www.meteonerola.it) insieme a
[wrf-nerola](https://github.com/eremite1/wrf-nerola) (modello WRF) e [meteo-mappe](https://github.com/eremite1/meteo-mappe).

## Cosa fa

1. **Scarica** dalla Protezione Civile (piattaforma Radar-DPC, dati liberi) gli ultimi scatti del radar nazionale
   (intensità di pioggia, ogni 5 minuti, ultime 2 ore) e dal servizio EUMETView di EUMETSAT le immagini del satellite
   Meteosat di terza generazione (MTG): visibile a 500 m, infrarosso a 1 km, colori reali, fulmini, ogni 10 minuti.
**Metodo principale: evoluzione guidata dal WRF** (`evoluzione.py`): si parte dall'ultimo scatto di radar e satellite e
lo si fa evolvere per 6 ore seguendo il WRF corretto: il vento in quota del WRF (850-500 hPa) sposta pioggia e nubi, e
lungo il percorso crescono, calano, nascono o si esauriscono come nel WRF. La semplice estrapolazione resta come
confronto: entra nella previsione solo dove le verifiche dimostrano che a quella scadenza fa meglio.
**Allerte** (`notifiche.py`): pioggia in arrivo, inizio pioggia, temporale vicino, grandine, su ntfy (gratis) e Telegram.

2. **Più movimenti**: la pioggia viene separata in celle (sotto circa 20 km) e sistema (fronte, banda). Le celle si
   muovono come misurato ogni 10 minuti, il sistema come misurato su 60 minuti; con le ore il sistema segue il
   movimento previsto dal WRF corretto (`nowcast.py`, `fusione.py`).
3. **Vita media separata** per celle e sistema: le celle si attenuano prima.
4. **WRF corretto punto per punto**: confrontato con radar e satellite di adesso, si calcola per ogni zona l'errore di
   posizione e di intensità, e la correzione si proietta su tutte le 6 ore (pioggia e nubi).
5. **Pesi imparati dai dati**: estrapolazione e WRF corretto vengono verificati separatamente; quando ci sono
   abbastanza casi di pioggia, i pesi a ogni scadenza vengono dall'abilità misurata.
- **Tipo di precipitazione**: pioggia, mista, neve (temperatura di bulbo umido da temperatura osservata della
  Protezione Civile e umidità del WRF) e grandine (probabilità di grandine del radar) (`tipo.py`).
6. **Si dà il voto da solo**: ogni previsione viene confrontata con quello che il radar vede davvero 30, 60, 120 e
   180 minuti dopo, e messa a confronto con la previsione più banale ("la pioggia resta dov'è").
7. **Nubi su mappa vera**: le nubi vengono separate dal suolo (piu' chiare nel visibile, piu' fredde nell'infrarosso)
   e disegnate bianche, con la loro ombra, sopra una mappa a colori naturali con il rilievo (`nubi.py`).
8. **Previsione delle nubi 0-6 ore** con lo stesso metodo della pioggia; il WRF entra con la nuvolosita' ricavata dalla
   radiazione infrarossa al suolo (`previsione_nubi.py`, `wrf.py`).
9. **Nerola adesso**: pioggia in arrivo, fulmine piu' vicino, cielo, in frasi semplici; e una griglia per sapere la
   previsione toccando un punto qualsiasi della mappa (`riassunto.py`).
11. **Gara dei metodi** (`gara.py`, `gara_giro.py`): ogni 10 minuti le 6 ore vengono previste in 12 modi diversi
   (solo spostamento, spostamento a due scale, radar che evolve col WRF, solo WRF, miscele con pesi diversi...). Quando
   arriva l'ora prevista ogni previsione viene sovrapposta al radar vero (e al satellite per le nubi) e riceve un voto
   0-100% (Fractions Skill Score su quadrati di 10 km). La classifica e' in `gara.json`; con abbastanza confronti la
   previsione pubblicata diventa da sola quella del metodo vincente (`modalita.json`, voce `scelta_automatica`).
12. **Pioggia caduta** nelle ultime 1, 3, 6, 12 e 24 ore (prodotti SRT1 e CUM della Protezione Civile).
13. **Radar come pluviometri virtuali** (`radar_punti.py`): pioggia oraria in 192 punti sul Lazio, per confrontare con il
   radar le previsioni di tutti i modelli nella pagella delle Mappe Lazio (anche WRF Nerola, stessi punti).
10. **Pagina**: un'unica mappa con strati (nubi, pioggia, fulmini), animazione sfumata da -2 a +6 ore, zoom, copia e
   salva; per esperti visibile HD, colori reali, infrarosso e top nubi.

**Come gira:** un avvio lavora per circa 5 ore e 45 minuti facendo un giro ogni 10 minuti, poi fa ripartire se stesso
(gli avvii a orario di GitHub sono troppo irregolari per un servizio ogni 10 minuti).

I dettagli scientifici (metodi e riferimenti) sono scritti all'inizio del file `nowcast.py`.

## Dove sono i risultati

- Ramo **`risultati`**: la pagina (`index.html`), le immagini, `indice.json`, la pagella (`pagella.json`, `storico.csv`).
- Immagine pronta dell'ultimo radar: `lazio/ultimo.png` e `sabina/ultimo.png` nel ramo `risultati`.
- Ramo **`diagnosi`**: compare solo se un giro non è riuscito, con il registro degli errori.

## I file

| File | A cosa serve |
|---|---|
| `scarica.py` | scarica radar e satellite dalla Protezione Civile e li ritaglia sull'Italia centrale |
| `fonti.py` | come si "parla" con il servizio della Protezione Civile |
| `geo.py` | legge i file GeoTIFF e converte le coordinate (proiezione del radar ↔ latitudine/longitudine) |
| `nubi.py`, `previsione_nubi.py`, `riassunto.py` | nubi su mappa, previsione delle nubi, Nerola adesso e fulmini |
| `satellite.py` | satellite ad alta definizione (EUMETSAT MTG): scarico, taratura delle temperature, 5 prodotti |
| `giro.sh` | un giro completo (scarico, previsione, pagella, immagini) |
| `wrf.py` | legge l'ultima previsione di wrf-nerola e la porta sulla griglia del radar |
| `fusione.py` | correzione del WRF punto per punto, movimenti a più scale, pesi imparati |
| `evoluzione.py` | evoluzione di radar e satellite guidata dal WRF |
| `notifiche.py` | allerte sul telefono (ntfy, Telegram) |
| `colonna.py` | bulbo umido e indici della colonna d'aria (copia da wrf-nerola) |
| `tipo.py` | pioggia, mista, neve e grandine |
| `nowcast.py` | il cuore: movimento, estrapolazione, correzione del WRF, fusione |
| `verifica.py` | la pagella |
| `gara.py`, `gara_giro.py` | la gara dei metodi: archivio, confronto con radar e satellite, classifica |
| `radar_punti.py` | pioggia oraria del radar in punti fissi (e del WRF negli stessi punti) per le pagelle dei modelli |
| `disegna.py` | le immagini |
| `sfondo.py` | confini, fiumi, nomi dei paesi e rilievo (lo stesso di wrf-nerola) |
| `pagina/index.html` | la pagina web |
| `prove/prova_sintetica.py` | prova di tutto il sistema con pioggia finta, senza internet |
| `.github/workflows/nowcasting.yml` | l'automazione che fa girare tutto ogni 10 minuti |

## Fonti e licenze

- Radar: Dipartimento della Protezione Civile, piattaforma Radar-DPC, dati elaborati.
- Satellite: contains modified EUMETSAT Meteosat (MTG) data, servizio EUMETView (CC BY 4.0).
- Modello: WRF su dati ECMWF IFS open data (CC BY 4.0), dati modificati.
- Rilievo: Terrain Tiles (SRTM/GMTED). Confini: ISTAT (CC BY 3.0 IT), Natural Earth. Nomi: GeoNames (CC BY 4.0).

Previsione sperimentale: non usare per decisioni di sicurezza. Per le allerte valgono i bollettini ufficiali.
