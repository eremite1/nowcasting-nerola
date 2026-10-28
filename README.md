# nowcasting-nerola

Radar, satellite e **previsione della pioggia a 0-6 ore** sul Lazio e sulla Sabina, con mappe proprie, aggiornati da soli
ogni 10 minuti sui computer di GitHub. Fa parte di [meteonerola.it](https://www.meteonerola.it) insieme a
[wrf-nerola](https://github.com/eremite1/wrf-nerola) (modello WRF) e [meteo-mappe](https://github.com/eremite1/meteo-mappe).

## Cosa fa

1. **Scarica** dalla Protezione Civile (piattaforma Radar-DPC, dati liberi) gli ultimi scatti del radar nazionale
   (intensità di pioggia, ogni 5 minuti, ultime 2 ore) e dal servizio EUMETView di EUMETSAT le immagini del satellite
   Meteosat di terza generazione (MTG): visibile a 500 m, infrarosso a 1 km, colori reali, fulmini, ogni 10 minuti.
2. **Calcola come si muove la pioggia** confrontando gli ultimi 4 scatti (flusso ottico) e la **trasporta in avanti**
   fino a 6 ore.
3. **Misura quanto durano oggi le celle di pioggia**: temporali = vita breve, piogge estese = vita lunga.
4. **Corregge il WRF** di Nerola confrontandolo con il radar di adesso (posizione, intensità, affidabilità).
5. **Unisce** le due cose: nelle prime decine di minuti conta quasi solo il radar, poi il peso passa al WRF corretto.
   I pesi non sono fissi: sono calcolati ogni volta dall'affidabilità misurata di ciascuno.
6. **Si dà il voto da solo**: ogni previsione viene confrontata con quello che il radar vede davvero 30, 60, 120 e
   180 minuti dopo, e messa a confronto con la previsione più banale ("la pioggia resta dov'è").
7. **Disegna** le mappe (rilievo, confini, paesi, Nerola) e la pagina web con l'animazione. Satellite in 5 versioni:
   radar + satellite (stile "sandwich"), visibile HD, colori reali, infrarosso, top nubi (temperature tarate ogni volta
   con l'infrarosso della Protezione Civile).

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
| `satellite.py` | satellite ad alta definizione (EUMETSAT MTG): scarico, taratura delle temperature, 5 prodotti |
| `giro.sh` | un giro completo (scarico, previsione, pagella, immagini) |
| `wrf.py` | legge l'ultima previsione di wrf-nerola e la porta sulla griglia del radar |
| `nowcast.py` | il cuore: movimento, estrapolazione, correzione del WRF, fusione |
| `verifica.py` | la pagella |
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
