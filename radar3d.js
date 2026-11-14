// RADAR 3D di meteonerola: il volume di pioggia visto dal radar (10 quote, da 1 a 10 km) sopra il rilievo del Lazio.
// Dati preparati da radar3d.py: radar3d/info.json, radar3d/volume.bin, radar3d/terreno.bin. Grafica: three.js.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// stessa scala a gradini della mappa base (mm/h); dal radar i dBZ diventano mm/h con la formula di Marshall-Palmer
const SOGLIE = [0.1, 0.25, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 5, 7, 10, 14, 20, 28, 40, 60, 90, 130, 200];
const COLORI = ["#a5ecff", "#62d5f6", "#2fb0ef", "#1f78e2", "#1849c9", "#0d8f3a", "#14a845", "#3cc23a", "#86d92a", "#f0e600",
  "#ffc300", "#ff9200", "#ff5800", "#eb1c1c", "#bd0016", "#8a0012", "#ff4fd2", "#d01fc8", "#8e10b8"];
const PAESI = [["Nerola", 12.786, 42.163, true], ["Roma", 12.496, 41.903], ["Rieti", 12.862, 42.404], ["Viterbo", 12.107, 42.417],
  ["Latina", 12.904, 41.467], ["Frosinone", 13.350, 41.640], ["Terni", 12.646, 42.564], ["L'Aquila", 13.399, 42.350]];
const GG = ["Dom", "Lun", "Mar", "Mer", "Gio", "Ven", "Sab"], MESI = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"];

const $ = (id) => document.getElementById(id);
const box = $("v3d"), msg = $("v3d-msg");
const mmh = (dbz) => Math.pow(Math.pow(10, dbz / 10) / 200, 1 / 1.6);
const dbzDa = (v) => (v - 1) / 2 - 10;
function colore(dbz) { const r = mmh(dbz); let k = 0; while (k + 1 < SOGLIE.length && r >= SOGLIE[k + 1]) k++; return COLORI[k]; }

let info = null, vol = null, terr = null, attivo = true;
let esag = 3, soglia = 20;
let scena, cam, rend, ctr, gruppoPioggia = null, etichette = [];
// il suolo e' diviso in tessere: ognuna ha una versione leggera (sempre pronta) e una dettagliata, che si scarica e si
// mostra solo quando ti avvicini (zoom dinamico: piu' vicino = mappa e rilievo piu' fini)
let tessere = [], texGrossa = null;
const VICINO = 95;          // distanza (km sullo schermo 3D) sotto la quale una tessera passa al dettaglio

async function scarica() {
  const i = await (await fetch("radar3d/info.json?c=" + Date.now())).json();
  if (info && i.ora_ms === info.ora_ms) return false;
  const v = "?v=" + i.ora_ms;
  const [a, b] = await Promise.all([fetch("radar3d/volume.bin" + v), fetch("radar3d/terreno.bin" + v)]);
  if (!a.ok || !b.ok) throw new Error("dati del radar 3D non disponibili");
  vol = new Uint8Array(await a.arrayBuffer()); terr = new Uint16Array(await b.arrayBuffer()); info = i;
  const t = new Date(i.ora_ms), hh = (n) => String(n).padStart(2, "0");
  $("t3-ora").textContent = "ore " + hh(t.getHours()) + ":" + hh(t.getMinutes());
  $("t3-giorno").textContent = GG[t.getDay()] + " " + t.getDate() + " " + MESI[t.getMonth()];
  return true;
}

// coordinate: x verso est, z verso sud (km, centro della mappa = 0), y in alto (km x esagerazione)
const geo = () => ({ C: info.colonne, R: info.righe, W: info.larghezza_km, H: info.altezza_km, dx: info.larghezza_km / info.colonne, dz: info.altezza_km / info.righe });
function inXZ(lon, lat) {
  const [l0, l1, a0, a1] = info.riquadro, g = geo();
  return [(lon - l0) / (l1 - l0) * g.W - g.W / 2, (a1 - lat) / (a1 - a0) * g.H - g.H / 2];
}
function quotaIn(x, z) {
  const g = geo(), c = Math.min(g.C - 1, Math.max(0, Math.floor((x + g.W / 2) / g.dx))), r = Math.min(g.R - 1, Math.max(0, Math.floor((z + g.H / 2) / g.dz)));
  return terr[r * g.C + c] / 1000;
}

// mappa leggera con sopra i confini (mappa.jpg + confini.png fusi in un'unica immagine)
function caricaImg(src) { return new Promise((ok, no) => { const im = new Image(); im.onload = () => ok(im); im.onerror = no; im.src = src; }); }
async function preparaTexGrossa() {
  const v = "?v=" + info.ora_ms;
  const [m, c] = await Promise.all([caricaImg(info.mappa + v), caricaImg((info.confini || "lazio/confini.png") + v).catch(() => null)]);
  const cv = document.createElement("canvas"); cv.width = m.naturalWidth * 2; cv.height = m.naturalHeight * 2;
  const x = cv.getContext("2d"); x.drawImage(m, 0, 0, cv.width, cv.height); if (c) x.drawImage(c, 0, 0, cv.width, cv.height);
  texGrossa = new THREE.CanvasTexture(cv); texGrossa.colorSpace = THREE.SRGBColorSpace; texGrossa.anisotropy = 4;
}

// geometria di una griglia di quote: vertici da (x0,z0) a (x1,z1), quote in metri riga per riga
function griglia(nr, nc, quote, x0, z0, x1, z1, uv) {
  const n = nr * nc, pos = new Float32Array(n * 3), uvs = new Float32Array(n * 2), idx = [];
  for (let r = 0; r < nr; r++) for (let c = 0; c < nc; c++) {
    const i = r * nc + c;
    pos[3 * i] = x0 + (x1 - x0) * c / (nc - 1); pos[3 * i + 1] = quote(r, c) / 1000 * esag; pos[3 * i + 2] = z0 + (z1 - z0) * r / (nr - 1);
    const [u, w] = uv(r, c); uvs[2 * i] = u; uvs[2 * i + 1] = w;
  }
  for (let r = 0; r < nr - 1; r++) for (let c = 0; c < nc - 1; c++) {
    const a = r * nc + c, b = a + 1, d = a + nc, e = d + 1; idx.push(a, d, b, b, d, e);
  }
  // "gonna" lungo i bordi: una striscia che scende sotto il suolo, cosi' tra due tessere vicine (una leggera e una
  // dettagliata) non restano fessure da cui si vede il cielo
  const bordo = [];
  for (let c = 0; c < nc; c++) bordo.push(c);
  for (let r = 1; r < nr; r++) bordo.push(r * nc + nc - 1);
  for (let c = nc - 2; c >= 0; c--) bordo.push((nr - 1) * nc + c);
  for (let r = nr - 2; r >= 0; r--) bordo.push(r * nc);
  const nb = bordo.length, P = new Float32Array((n + nb) * 3), U = new Float32Array((n + nb) * 2);
  P.set(pos); U.set(uvs);
  bordo.forEach((v, k) => {
    P[3 * (n + k)] = pos[3 * v]; P[3 * (n + k) + 1] = pos[3 * v + 1] - 1.5 * esag; P[3 * (n + k) + 2] = pos[3 * v + 2];
    U[2 * (n + k)] = uvs[2 * v]; U[2 * (n + k) + 1] = uvs[2 * v + 1];
  });
  for (let k = 0; k < nb; k++) { const a = bordo[k], b = bordo[(k + 1) % nb], a2 = n + k, b2 = n + (k + 1) % nb; idx.push(a, a2, b, b, a2, b2); }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(P, 3)); geom.setAttribute("uv", new THREE.BufferAttribute(U, 2));
  geom.setIndex(idx); geom.computeVertexNormals(); return geom;
}

function costruisciTerreno() {
  const g = geo(), NT = (info.dettaglio && info.dettaglio.tessere) || 4;
  tessere.forEach((t) => { scena.remove(t.grossa); t.grossa.geometry.dispose(); if (t.fine) { scena.remove(t.fine); t.fine.geometry.dispose(); } });
  const vecchie = tessere; tessere = [];
  const mat = new THREE.MeshLambertMaterial({ map: texGrossa });
  for (let i = 0; i < NT; i++) for (let j = 0; j < NT; j++) {
    const r0 = Math.round(i * (g.R - 1) / NT), r1 = Math.round((i + 1) * (g.R - 1) / NT);
    const c0 = Math.round(j * (g.C - 1) / NT), c1 = Math.round((j + 1) * (g.C - 1) / NT);
    const X = (c) => -g.W / 2 + (c + 0.5) * g.dx, Z = (r) => -g.H / 2 + (r + 0.5) * g.dz;
    const geom = griglia(r1 - r0 + 1, c1 - c0 + 1, (r, c) => terr[(r0 + r) * g.C + c0 + c], X(c0), Z(r0), X(c1), Z(r1),
      (r, c) => [(c0 + c + 0.5) / g.C, 1 - (r0 + r + 0.5) / g.R]);
    const grossa = new THREE.Mesh(geom, mat); scena.add(grossa);
    const old = vecchie.find((t) => t.i === i && t.j === j);
    const t = { i, j, grossa, fine: null, dati: old ? old.dati : null, carica: false,
      centro: new THREE.Vector3((X(c0) + X(c1)) / 2, 0, (Z(r0) + Z(r1)) / 2), bordi: [-g.W / 2 + j * g.W / NT, -g.H / 2 + i * g.H / NT, -g.W / 2 + (j + 1) * g.W / NT, -g.H / 2 + (i + 1) * g.H / NT] };
    if (t.dati) costruisciFine(t);
    tessere.push(t);
  }
  livelloDettaglio();
}

function costruisciFine(t) {
  const d = t.dati, [x0, z0, x1, z1] = t.bordi;
  if (t.fine) { scena.remove(t.fine); t.fine.geometry.dispose(); }
  const geom = griglia(d.righe, d.colonne, (r, c) => d.quote[r * d.colonne + c], x0, z0, x1, z1, (r, c) => [c / (d.colonne - 1), 1 - r / (d.righe - 1)]);
  t.fine = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({ map: d.tex })); t.fine.visible = false; scena.add(t.fine);
}

async function caricaFine(t) {
  const det = info.dettaglio; if (!det || t.carica) return;
  t.carica = true;
  const e = det.elenco.find((x) => x.i === t.i && x.j === t.j); if (!e) return;
  try {
    const base = "radar3d/dettaglio/t" + t.i + "_" + t.j, v = "?d=" + det.versione;
    const [b, tex] = await Promise.all([fetch(base + ".bin" + v).then((r) => { if (!r.ok) throw new Error(); return r.arrayBuffer(); }),
      new THREE.TextureLoader().loadAsync(base + ".jpg" + v)]);
    tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
    t.dati = { righe: e.righe, colonne: e.colonne, quote: new Uint16Array(b), tex };
    costruisciFine(t); livelloDettaglio(); disegna();
  } catch (err) { /* la tessera dettagliata non c'e': resta quella leggera */ }
}

// per ogni tessera: dettagliata se la telecamera e' vicina, leggera altrimenti
function livelloDettaglio() {
  if (!cam) return;
  const p = new THREE.Vector3();
  tessere.forEach((t) => {
    p.copy(t.centro); p.y = 0;
    const vicina = cam.position.distanceTo(p) < VICINO * Math.max(1, esag / 3) + 20;
    if (vicina && !t.dati) caricaFine(t);
    const fine = vicina && !!t.fine;
    if (t.fine) t.fine.visible = fine;
    t.grossa.visible = !fine;
  });
}

function costruisciPioggia() {
  const g = geo(), N = g.C * g.R;
  if (gruppoPioggia) { scena.remove(gruppoPioggia); gruppoPioggia.geometry.dispose(); gruppoPioggia.material.dispose(); }
  const celle = [];
  let max = -99, alto = 0;
  for (let k = 0; k < info.quote_m.length; k++) {
    const q = info.quote_m[k] / 1000;
    for (let i = 0; i < N; i++) {
      const v = vol[k * N + i]; if (!v) continue;
      const d = dbzDa(v); if (d < soglia) continue;
      if (terr[i] / 1000 > q + 0.3) continue;              // sotto il rilievo: eco di montagna, non pioggia
      celle.push(k, i, d); if (d > max) max = d; if (q > alto) alto = q;
    }
  }
  const n = celle.length / 3;
  const geom = new THREE.BoxGeometry(g.dx * 0.96, 0.96 * esag, g.dz * 0.96);
  const mat = new THREE.MeshLambertMaterial();
  const m = new THREE.InstancedMesh(geom, mat, Math.max(1, n));
  const M = new THREE.Matrix4(), col = new THREE.Color();
  for (let j = 0; j < n; j++) {
    const k = celle[3 * j], i = celle[3 * j + 1], d = celle[3 * j + 2];
    const r = Math.floor(i / g.C), c = i % g.C;
    M.makeTranslation(-g.W / 2 + (c + 0.5) * g.dx, info.quote_m[k] / 1000 * esag, -g.H / 2 + (r + 0.5) * g.dz);
    m.setMatrixAt(j, M); m.setColorAt(j, col.set(colore(d)));
  }
  m.count = n; m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true;
  gruppoPioggia = m; scena.add(m);
  $("t3-etichetta").textContent = n ? "ECO FINO A " + alto + " KM · MAX " + Math.round(max) + " dBZ" : "NESSUNA PIOGGIA SOPRA LA SOGLIA";
}

function etichetta(testo, forte) {
  const c = document.createElement("canvas"), x = c.getContext("2d"), f = forte ? 44 : 34;
  x.font = "700 " + f + "px system-ui, sans-serif";
  c.width = Math.ceil(x.measureText(testo).width) + 24; c.height = f + 20;
  x.font = "700 " + f + "px system-ui, sans-serif"; x.textBaseline = "middle";
  x.lineWidth = 8; x.strokeStyle = "rgba(255,255,255,.95)"; x.strokeText(testo, 12, c.height / 2);
  x.fillStyle = forte ? "#c0111f" : "#1d2733"; x.fillText(testo, 12, c.height / 2);
  // grandezza fissa sullo schermo: le scritte non diventano enormi quando ti avvicini
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false, sizeAttenuation: false }));
  const h = forte ? 0.045 : 0.034; s.scale.set(h * c.width / c.height, h, 1); s.renderOrder = 10;
  return s;
}
function costruisciEtichette() {
  etichette.forEach((e) => scena.remove(e)); etichette = [];
  PAESI.forEach(([n, lo, la, forte]) => {
    const [x, z] = inXZ(lo, la), y = quotaIn(x, z) * esag;
    const s = etichetta(n, forte); s.center.set(0.5, 0); s.position.set(x, y + 0.6, z); scena.add(s); etichette.push(s);
    const punto = new THREE.Mesh(new THREE.SphereGeometry(0.35, 12, 8), new THREE.MeshBasicMaterial({ color: forte ? 0xc0111f : 0x1d2733 }));
    punto.position.set(x, y + 0.3, z); scena.add(punto); etichette.push(punto);
  });
}

function vistaIniziale() {
  const [x, z] = inXZ(12.786, 42.163);
  ctr.target.set(x, 2, z);
  cam.position.set(x - 70, 85, z + 120);
  ctr.update(); disegna();
}

function disegna() { if (rend && attivo) rend.render(scena, cam); }

function dimensiona() {
  const w = box.clientWidth, h = box.clientHeight; if (!w || !h) return;
  rend.setSize(w, h); cam.aspect = w / h; cam.updateProjectionMatrix(); disegna();
}

async function avvia() {
  try { await scarica(); } catch (e) { msg.textContent = "Il radar 3D non è ancora disponibile: riprova tra qualche minuto."; return; }
  scena = new THREE.Scene();
  scena.background = new THREE.Color(0xcfdbe8);
  scena.fog = new THREE.Fog(0xcfdbe8, 350, 900);
  scena.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.6));
  const sole = new THREE.DirectionalLight(0xffffff, 1.6); sole.position.set(-120, 200, 80); scena.add(sole);
  cam = new THREE.PerspectiveCamera(42, 1, 0.5, 3000);
  rend = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  rend.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  box.appendChild(rend.domElement); msg.remove();
  ctr = new OrbitControls(cam, rend.domElement);
  ctr.enableDamping = true; ctr.dampingFactor = 0.08; ctr.maxPolarAngle = Math.PI / 2.05; ctr.minDistance = 8; ctr.maxDistance = 600;
  ctr.addEventListener("change", () => { livelloDettaglio(); disegna(); });
  await preparaTexGrossa();
  costruisciTerreno(); costruisciPioggia(); costruisciEtichette(); vistaIniziale();
  new ResizeObserver(dimensiona).observe(box); dimensiona();
  rend.setAnimationLoop(() => { if (attivo && ctr.update()) disegna(); });
  // aggiornamento: ogni 5 minuti controlla se c'e' un radar nuovo
  setInterval(async () => {
    if (!attivo || document.hidden) return;
    try { if (await scarica()) { costruisciPioggia(); disegna(); } } catch (e) { /* riprova dopo */ }
  }, 300000);
}

$("t3-soglia").addEventListener("input", (e) => {
  soglia = +e.target.value; $("t3-soglia-v").textContent = soglia + " dBZ";
  $("t3-soglia-mm").textContent = "(circa " + String(Math.round(mmh(soglia) * 10) / 10).replace(".", ",") + " mm/h e oltre)";
  if (info) { costruisciPioggia(); disegna(); }
});
$("t3-esag").addEventListener("input", (e) => {
  esag = +e.target.value; $("t3-esag-v").textContent = esag === 1 ? "no (reali)" : esag + " volte";
  if (info) { costruisciTerreno(); costruisciPioggia(); costruisciEtichette(); disegna(); }
});
$("t3-centro").addEventListener("click", () => { if (info) vistaIniziale(); });
// legenda: stessi colori della mappa base, ma in dBZ (intensita' dell'eco, l'unita' del radar)
(function () {
  const b = $("t3-scala"); if (!b) return;
  const n = SOGLIE.length, pos = (k) => (100 * k / n).toFixed(2);
  const dbz = (r) => Math.round(10 * Math.log10(200 * Math.pow(r, 1.6)));
  b.style.background = "linear-gradient(90deg," + COLORI.map((c, k) => c + " " + pos(k) + "% " + pos(k + 1) + "%").join(",") + ")";
  b.innerHTML = SOGLIE.map((r, k) => k % 2 === 0 ? '<span style="left:' + pos(k) + '%">' + dbz(r) + "</span>" : "").join("");
})();
$("t3-soglia").dispatchEvent(new Event("input"));

window.mnRadar3D = { mostra() { attivo = true; if (rend) dimensiona(); }, pausa() { attivo = false; } };
avvia();
