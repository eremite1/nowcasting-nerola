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
let scena, cam, rend, ctr, gruppoPioggia = null, terreno = null, etichette = [];

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

function costruisciTerreno() {
  const g = geo();
  if (terreno) { scena.remove(terreno); terreno.geometry.dispose(); }
  const geom = new THREE.PlaneGeometry(g.W, g.H, g.C - 1, g.R - 1);
  geom.rotateX(-Math.PI / 2);
  const pos = geom.attributes.position;
  for (let i = 0; i < pos.count; i++) pos.setY(i, terr[i] / 1000 * esag);
  geom.computeVertexNormals();
  const tex = new THREE.TextureLoader().load(info.mappa + "?v=" + info.ora_ms, () => disegna());
  tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4;
  terreno = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({ map: tex }));
  scena.add(terreno);
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
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  const h = forte ? 5.5 : 4.2; s.scale.set(h * c.width / c.height, h, 1); s.renderOrder = 10;
  return s;
}
function costruisciEtichette() {
  etichette.forEach((e) => scena.remove(e)); etichette = [];
  PAESI.forEach(([n, lo, la, forte]) => {
    const [x, z] = inXZ(lo, la), y = quotaIn(x, z) * esag;
    const s = etichetta(n, forte); s.position.set(x, y + 4, z); scena.add(s); etichette.push(s);
    const asta = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.25, 4, 6), new THREE.MeshBasicMaterial({ color: forte ? 0xc0111f : 0x1d2733 }));
    asta.position.set(x, y + 2, z); scena.add(asta); etichette.push(asta);
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
  ctr.addEventListener("change", disegna);
  costruisciTerreno(); costruisciPioggia(); costruisciEtichette(); vistaIniziale();
  new ResizeObserver(dimensiona).observe(box); dimensiona();
  rend.setAnimationLoop(() => { if (attivo && ctr.update()) disegna(); });
  // aggiornamento: ogni 5 minuti controlla se c'e' un radar nuovo
  setInterval(async () => {
    if (!attivo || document.hidden) return;
    try { if (await scarica()) { costruisciTerreno(); costruisciPioggia(); costruisciEtichette(); disegna(); } } catch (e) { /* riprova dopo */ }
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
$("t3-soglia").dispatchEvent(new Event("input"));

window.mnRadar3D = { mostra() { attivo = true; if (rend) dimensiona(); }, pausa() { attivo = false; } };
avvia();
