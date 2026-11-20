// RADAR 3D di meteonerola: il volume di pioggia visto dal radar (10 quote CAPPI, da 1 a 10 km) sopra il rilievo del Lazio.
// Dati preparati da radar3d.py: radar3d/info.json, radar3d/volume.bin, radar3d/terreno.bin. Grafica: three.js.
// Il volume non e' fatto di cubetti: e' una "texture 3D" che la scheda grafica attraversa raggio per raggio (ray marching),
// cosi' la pioggia appare come una nube continua. Quattro modi: Volume, Olografico, Iso-superfici, Sezione verticale.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// scala colori continua per i dBZ (eco radar): blu debole, verde moderata, giallo-arancio forte, rosso molto forte, viola grandine
const SCALA = [[-10, "#0b2a55"], [5, "#2a5fb8"], [12, "#2a8fe0"], [18, "#22c3e6"], [24, "#19c37d"], [30, "#7fd72a"], [35, "#f1e11c"],
  [40, "#ffae1a"], [45, "#ff6a1a"], [50, "#e8172c"], [55, "#b0005a"], [60, "#d63cff"], [66, "#ffffff"], [70, "#ffffff"]];
const PAESI = [["Nerola", 12.786, 42.163, true], ["Roma", 12.496, 41.903], ["Rieti", 12.862, 42.404], ["Viterbo", 12.107, 42.417],
  ["Latina", 12.904, 41.467], ["Frosinone", 13.350, 41.640], ["Terni", 12.646, 42.564], ["L'Aquila", 13.399, 42.350]];
const GG = ["Dom", "Lun", "Mar", "Mer", "Gio", "Ven", "Sab"], MESI = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"];
const NOMI_MODO = ["VOLUME", "OLOGRAFICO", "ISO-SUPERFICI", "SEZIONE VERTICALE"];
const NERO_LAT = 42.163, NERO_LON = 12.786;

const $ = (id) => document.getElementById(id);
const box = $("v3d"), msg = $("v3d-msg");
const mmh = (dbz) => Math.pow(Math.pow(10, dbz / 10) / 200, 1 / 1.6);
const hh = (n) => String(n).padStart(2, "0");

let info = null, vol = null, terr = null, attivo = true;
let esag = 8, soglia = 15, modo = 0, asse = 0, pos = 0.5;
let scena, cam, rend, ctr, etichette = [], asseQuote = null;
let volMesh = null, volMat = null, texVol = null, texTer = null, texPal = null, volS = null, bboxEco = null, maxEco = -99, altoEco = 0;
let tessere = [], texGrossa = null;
const VICINO = 95;

// ---------------------------------------------------------------- colori
function hex(c) { return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)]; }
function coloreDbz(d) {
  for (let k = 0; k + 1 < SCALA.length; k++) {
    const [a, ca] = SCALA[k], [b, cb] = SCALA[k + 1];
    if (d <= b) { const f = Math.max(0, Math.min(1, (d - a) / (b - a))), x = hex(ca), y = hex(cb); return x.map((v, i) => Math.round(v + (y[i] - v) * f)); }
  }
  return [255, 255, 255];
}
function texturaPalette() {
  const n = 256, d = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) { const c = coloreDbz(-10 + 80 * i / (n - 1)); d.set([c[0], c[1], c[2], 255], i * 4); }
  const t = new THREE.DataTexture(d, n, 1, THREE.RGBAFormat); t.magFilter = t.minFilter = THREE.LinearFilter; t.needsUpdate = true; return t;
}

// ---------------------------------------------------------------- dati
async function scarica() {
  const i = await (await fetch("radar3d/info.json?c=" + Date.now())).json();
  if (info && i.ora_ms === info.ora_ms) return false;
  const v = "?v=" + i.ora_ms;
  const [a, b] = await Promise.all([fetch("radar3d/volume.bin" + v), fetch("radar3d/terreno.bin" + v)]);
  if (!a.ok || !b.ok) throw new Error("dati del radar 3D non disponibili");
  vol = new Uint8Array(await a.arrayBuffer()); terr = new Uint16Array(await b.arrayBuffer()); info = i;
  const t = new Date(i.ora_ms);
  $("t3-ora").textContent = GG[t.getDay()] + " " + t.getDate() + " " + MESI[t.getMonth()] + " · ore " + hh(t.getHours()) + ":" + hh(t.getMinutes());
  $("t3-crediti").textContent = "Radar-DPC · CC BY-SA 4.0 · CAPPI " + hh(t.getHours()) + ":" + hh(t.getMinutes()) + " · 1–10 km s.l.m.";
  preparaVolume();
  return true;
}

// volume "ammorbidito": i dati arrivano a celle di circa 1,4 km; una leggera sfumatura orizzontale toglie lo scalino tra
// una cella e l'altra senza spegnere i nuclei (si tiene il 40% del valore originale). Le celle sotto il rilievo si azzerano.
function preparaVolume() {
  const C = info.colonne, R = info.righe, L = info.quote_m.length, N = C * R;
  const dbz = new Float32Array(L * N);
  for (let k = 0; k < L; k++) for (let i = 0; i < N; i++) {
    const v = vol[k * N + i];
    dbz[k * N + i] = v && terr[i] / 1000 <= info.quote_m[k] / 1000 + 0.3 ? (v - 1) / 2 - 10 : -12;
  }
  const out = new Uint8Array(L * N), tmp = new Float32Array(N);
  bboxEco = [C, R, -1, -1]; maxEco = -99; altoEco = 0;
  for (let k = 0; k < L; k++) {
    const s = dbz.subarray(k * N, (k + 1) * N);
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const a = s[r * C + Math.max(0, c - 1)], b = s[r * C + c], d = s[r * C + Math.min(C - 1, c + 1)]; tmp[r * C + c] = (a + 2 * b + d) / 4;
    }
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const i = r * C + c, a = tmp[Math.max(0, r - 1) * C + c], b = tmp[i], d = tmp[Math.min(R - 1, r + 1) * C + c];
      const v = 0.6 * (a + 2 * b + d) / 4 + 0.4 * s[i];
      out[k * N + i] = Math.max(0, Math.min(255, Math.round((v + 12) * 3)));   // 0..255 = -12..73 dBZ
      if (s[i] >= 5) {
        if (c < bboxEco[0]) bboxEco[0] = c; if (r < bboxEco[1]) bboxEco[1] = r; if (c > bboxEco[2]) bboxEco[2] = c; if (r > bboxEco[3]) bboxEco[3] = r;
        if (s[i] > maxEco) maxEco = s[i]; altoEco = Math.max(altoEco, info.quote_m[k] / 1000);
      }
    }
  }
  volS = out;
  if (texVol) texVol.dispose();
  texVol = new THREE.Data3DTexture(out, C, R, L);
  texVol.format = THREE.RedFormat; texVol.type = THREE.UnsignedByteType;
  texVol.minFilter = texVol.magFilter = THREE.LinearFilter; texVol.unpackAlignment = 1;
  texVol.wrapS = texVol.wrapT = texVol.wrapR = THREE.ClampToEdgeWrapping; texVol.needsUpdate = true;
  const ter = new Uint8Array(N); for (let i = 0; i < N; i++) ter[i] = Math.min(255, Math.round(terr[i] / 12));
  if (texTer) texTer.dispose();
  texTer = new THREE.DataTexture(ter, C, R, THREE.RedFormat, THREE.UnsignedByteType);
  texTer.minFilter = texTer.magFilter = THREE.LinearFilter; texTer.unpackAlignment = 1; texTer.needsUpdate = true;
}
// dBZ in un punto (x est, z sud in km dal centro; h quota in km), interpolato come fa la scheda grafica
function dbzIn(x, z, h) {
  const C = info.colonne, R = info.righe, L = info.quote_m.length, g = geo();
  const fc = Math.max(0, Math.min(C - 1, (x + g.W / 2) / g.dx - 0.5)), fr = Math.max(0, Math.min(R - 1, (z + g.H / 2) / g.dz - 0.5));
  const fk = Math.max(0, Math.min(L - 1, h - 1));
  const c0 = Math.floor(fc), r0 = Math.floor(fr), k0 = Math.floor(fk), c1 = Math.min(C - 1, c0 + 1), r1 = Math.min(R - 1, r0 + 1), k1 = Math.min(L - 1, k0 + 1);
  const a = fc - c0, b = fr - r0, e = fk - k0, N = C * R;
  const V = (k, r, c) => volS[k * N + r * C + c];
  const piano = (k) => (V(k, r0, c0) * (1 - a) + V(k, r0, c1) * a) * (1 - b) + (V(k, r1, c0) * (1 - a) + V(k, r1, c1) * a) * b;
  return (piano(k0) * (1 - e) + piano(k1) * e) / 3 - 12;
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

// ---------------------------------------------------------------- il volume (shader)
// IMPORTANTE (Chrome su PC Windows): dentro il ciclo si legge con textureLod e non con texture. Su Windows Chrome
// traduce lo shader per DirectX, che con texture() dentro un ciclo e' costretto a "srotolare" tutti i 256 passi:
// lo shader diventa enorme, la compilazione blocca la scheda grafica e Chrome si impalla. Con textureLod il ciclo resta un ciclo.
const VERT = `
varying vec3 vW;
void main() { vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;
const FRAG = `
precision highp float; precision highp sampler3D;
uniform sampler3D uVol; uniform sampler2D uTer; uniform sampler2D uPal;
uniform vec3 uMin; uniform vec3 uMax; uniform vec2 uDom;
uniform float uEsag; uniform float uSoglia; uniform float uModo; uniform float uAsse; uniform float uTaglio; uniform float uPasso;
varying vec3 vW;
vec3 tc(vec3 p) { return vec3((p.x + uDom.x * 0.5) / uDom.x, (p.z + uDom.y * 0.5) / uDom.y, (p.y / uEsag - 0.5) / 10.0); }
float dbz(vec3 p) { return textureLod(uVol, tc(p), 0.0).r * 85.0 - 12.0; }
float suolo(vec3 p) { return textureLod(uTer, vec2((p.x + uDom.x * 0.5) / uDom.x, (p.z + uDom.y * 0.5) / uDom.y), 0.0).r * 3.06 * uEsag; }
vec3 pal(float d) { return textureLod(uPal, vec2(clamp((d + 10.0) / 80.0, 0.0, 1.0), 0.5), 0.0).rgb; }
vec3 normale(vec3 p, float d) {
  float ex = uPasso, ey = 0.5 * uEsag;
  vec3 g = vec3(dbz(p + vec3(ex, 0.0, 0.0)) - d, (dbz(p + vec3(0.0, ey, 0.0)) - d) * ex / ey, dbz(p + vec3(0.0, 0.0, ex)) - d);
  float l = length(g); return l > 1e-4 ? -g / l : vec3(0.0, 1.0, 0.0);
}
void dentro(inout vec4 acc, vec3 c, float a) { acc.rgb += (1.0 - acc.a) * a * c; acc.a += (1.0 - acc.a) * a; }
const vec3 LUCE = vec3(-0.458, 0.814, 0.356);
void main() {
  vec3 ro = cameraPosition, rd = normalize(vW - ro);
  vec3 inv = 1.0 / rd, t0 = (uMin - ro) * inv, t1 = (uMax - ro) * inv, a = min(t0, t1), b = max(t0, t1);
  float tn = max(max(a.x, a.y), max(a.z, 0.0)), tf = min(min(b.x, b.y), b.z);
  if (tf <= tn) discard;
  float n = clamp((tf - tn) / (uPasso * 0.7), 24.0, 256.0), dt = (tf - tn) / n;
  float jit = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  float t = tn + dt * jit;
  float tp = -1.0;
  if (uModo > 2.5) { float den = uAsse < 0.5 ? rd.z : rd.x; if (abs(den) > 1e-5) tp = ((uTaglio) - (uAsse < 0.5 ? ro.z : ro.x)) / den; if (tp < tn || tp > tf) tp = -1.0; }
  vec4 acc = vec4(0.0); vec3 glow = vec3(0.0);
  float prima = -20.0;
  float L0 = max(uSoglia, 8.0), L1 = max(L0 + 8.0, 35.0), L2 = max(L1 + 8.0, 50.0);
  bool terra = false;
  for (int i = 0; i < 256; i++) {
    if (float(i) >= n) break;
    vec3 p = ro + rd * t;
    if (tp > 0.0 && t >= tp) {
      vec3 q = ro + rd * tp; tp = -1.0;
      if (q.y >= suolo(q) && q.y <= 10.5 * uEsag) {
        float d = dbz(q), k = q.y / uEsag;
        float riga = 1.0 - smoothstep(0.0, 0.035 * max(1.0, 8.0 / uEsag), abs(fract(k + 0.5) - 0.5));
        if (d >= uSoglia - 3.0) dentro(acc, pal(d) * 1.1 + riga * 0.15, 0.93 * smoothstep(uSoglia - 3.0, uSoglia + 2.0, d));
        dentro(acc, mix(vec3(0.45, 0.75, 1.0), vec3(0.8, 0.95, 1.0), riga), 0.05 + 0.30 * riga);
      }
    }
    if (p.y < suolo(p)) { terra = true; break; }
    float d = dbz(p);
    if (uModo < 0.5 || uModo > 2.5) {
      if (d > uSoglia - 5.0) {
        float k = smoothstep(uSoglia - 4.0, uSoglia + 6.0, d) * (0.045 + 0.55 * smoothstep(22.0, 58.0, d));
        if (uModo > 2.5) k *= 0.18;
        float al = 1.0 - exp(-k * dt);
        vec3 c = pal(d);
        if (al > 0.004) { vec3 nn = normale(p, d); c *= 0.55 + 0.65 * max(dot(nn, LUCE), 0.0); }
        c += pal(d) * 0.45 * smoothstep(42.0, 62.0, d);
        dentro(acc, c, al);
      }
    } else if (uModo < 1.5) {
      float w = smoothstep(uSoglia - 4.0, uSoglia + 8.0, d);
      float bordo = clamp(abs(d - prima) / 6.0, 0.0, 1.0) * w;
      float sc = 0.78 + 0.22 * sin(p.y / uEsag * 6.2831 * 3.0);
      float e = w * (0.025 + 0.16 * smoothstep(18.0, 60.0, d)) + bordo * 0.10;
      glow += pal(d) * e * dt * sc * 1.6;
      acc.a += (1.0 - acc.a) * (1.0 - exp(-w * 0.012 * dt));
    } else {
      for (int j = 0; j < 3; j++) {
        float L = j == 0 ? L0 : (j == 1 ? L1 : L2);
        float al = j == 0 ? 0.26 : (j == 1 ? 0.45 : 0.85);
        bool entra = prima < L && d >= L, esce = prima >= L && d < L;
        if (entra || esce) {
          float f = clamp((L - prima) / (d - prima), 0.0, 1.0);
          vec3 q = p - rd * dt * (1.0 - f);
          vec3 nn = normale(q, L); if (dot(nn, rd) > 0.0) nn = -nn;
          float dif = max(dot(nn, LUCE), 0.0), rim = pow(1.0 - abs(dot(nn, rd)), 2.0);
          vec3 c = pal(L + 2.0) * (0.35 + 0.8 * dif) + rim * 0.5 * pal(L + 4.0) + vec3(0.25) * pow(max(dot(reflect(-LUCE, nn), -rd), 0.0), 24.0);
          dentro(acc, c, esce ? al * 0.55 : al);
        }
      }
    }
    if (acc.a > 0.985) break;
    prima = d; t += dt;
  }
  if (tp > 0.0 && !terra && acc.a < 0.985) {
    vec3 q = ro + rd * tp;
    if (q.y >= suolo(q) && q.y <= 10.5 * uEsag) {
      float d = dbz(q), k = q.y / uEsag;
      float riga = 1.0 - smoothstep(0.0, 0.035 * max(1.0, 8.0 / uEsag), abs(fract(k + 0.5) - 0.5));
      if (d >= uSoglia - 3.0) dentro(acc, pal(d) * 1.1, 0.93 * smoothstep(uSoglia - 3.0, uSoglia + 2.0, d));
      dentro(acc, mix(vec3(0.45, 0.75, 1.0), vec3(0.8, 0.95, 1.0), riga), 0.05 + 0.30 * riga);
    }
  }
  if (uModo > 0.5 && uModo < 1.5) { vec3 g = 1.0 - exp(-glow); gl_FragColor = vec4(g, max(acc.a, max(g.r, max(g.g, g.b)) * 0.55)); return; }
  gl_FragColor = acc;
}`;

function costruisciVolume() {
  const g = geo();
  if (!volMat) {
    texPal = texturaPalette();
    volMat = new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false,
      blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
      uniforms: { uVol: { value: null }, uTer: { value: null }, uPal: { value: texPal }, uMin: { value: new THREE.Vector3() }, uMax: { value: new THREE.Vector3() },
        uDom: { value: new THREE.Vector2() }, uEsag: { value: esag }, uSoglia: { value: soglia }, uModo: { value: modo }, uAsse: { value: asse },
        uTaglio: { value: 0 }, uPasso: { value: 1 } },
    });
    volMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), volMat); volMesh.renderOrder = 5; volMesh.frustumCulled = false; scena.add(volMesh);
  }
  const u = volMat.uniforms;
  u.uVol.value = texVol; u.uTer.value = texTer; u.uDom.value.set(g.W, g.H); u.uEsag.value = esag; u.uSoglia.value = soglia;
  u.uModo.value = modo; u.uAsse.value = asse; u.uPasso.value = Math.min(g.dx, g.dz);
  u.uTaglio.value = posizioneTaglio();
  // scatola da attraversare: solo dove c'e' eco (molto piu' veloce); intera nella sezione verticale
  let x0 = -g.W / 2, x1 = g.W / 2, z0 = -g.H / 2, z1 = g.H / 2, y1 = 10.5 * esag;
  if (modo !== 3 && bboxEco[2] >= 0) {
    x0 = -g.W / 2 + Math.max(0, bboxEco[0] - 2) * g.dx; x1 = -g.W / 2 + Math.min(g.C, bboxEco[2] + 3) * g.dx;
    z0 = -g.H / 2 + Math.max(0, bboxEco[1] - 2) * g.dz; z1 = -g.H / 2 + Math.min(g.R, bboxEco[3] + 3) * g.dz;
    y1 = Math.min(10.5, altoEco + 1.0) * esag;
  }
  volMesh.visible = modo === 3 || bboxEco[2] >= 0;
  u.uMin.value.set(x0, 0, z0); u.uMax.value.set(x1, y1, z1);
  volMesh.scale.set(x1 - x0, y1, z1 - z0); volMesh.position.set((x0 + x1) / 2, y1 / 2, (z0 + z1) / 2);
  $("t3-etichetta").textContent = bboxEco[2] >= 0 ? "eco fino a " + altoEco + " km · max " + Math.round(maxEco) + " dBZ" : "nessuna eco radar in questo momento";
  $("t3-modo").textContent = NOMI_MODO[modo];
}
// se la telecamera e' dentro la scatola si disegnano le facce interne, altrimenti quelle esterne (che rispettano il rilievo davanti)
function lato() {
  if (!volMesh) return;
  const u = volMat.uniforms, p = cam.position, m = 1;
  const dentro = p.x > u.uMin.value.x - m && p.x < u.uMax.value.x + m && p.y < u.uMax.value.y + m && p.z > u.uMin.value.z - m && p.z < u.uMax.value.z + m;
  const s = dentro ? THREE.BackSide : THREE.FrontSide;
  if (volMat.side !== s) { volMat.side = s; volMat.depthTest = !dentro; volMat.needsUpdate = true; }
}

// ---------------------------------------------------------------- sezione verticale
function posizioneTaglio() { const g = geo(); return asse === 0 ? -g.H / 2 + pos * g.H : -g.W / 2 + pos * g.W; }
function testoPosizione() {
  const [l0, l1, a0, a1] = info.riquadro;
  if (asse === 0) { const lat = a1 - pos * (a1 - a0); return "lat " + lat.toFixed(2).replace(".", ",") + "° N" + (Math.abs(lat - NERO_LAT) < 0.02 ? " (Nerola)" : ""); }
  const lon = l0 + pos * (l1 - l0); return "lon " + lon.toFixed(2).replace(".", ",") + "° E" + (Math.abs(lon - NERO_LON) < 0.025 ? " (Nerola)" : "");
}
function disegnaSezione() {
  const cv = $("t3-sez"); if (!cv || modo !== 3 || !info) return;
  const W = cv.clientWidth, H = cv.clientHeight, dpr = Math.min(2, window.devicePixelRatio || 1);
  if (!W) return;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  const x = cv.getContext("2d"); x.setTransform(dpr, 0, 0, dpr, 0, 0);
  const g = geo(), lung = asse === 0 ? g.W : g.H, taglio = posizioneTaglio();
  const m = { s: 44, d: 12, a: 26, b: 28 }, pw = W - m.s - m.d, ph = H - m.a - m.b, HMAX = 10.5;
  x.fillStyle = "#070d18"; x.fillRect(0, 0, W, H);
  // immagine dell'eco: una colonna per ogni pixel orizzontale, 0,05 km in verticale
  const nx = Math.min(600, Math.round(pw)), ny = 105, img = x.createImageData(nx, ny);
  const prof = new Float32Array(nx);
  for (let i = 0; i < nx; i++) {
    const s = -lung / 2 + (i + 0.5) / nx * lung, px = asse === 0 ? s : taglio, pz = asse === 0 ? taglio : s;
    prof[i] = quotaIn(px, pz);
    for (let j = 0; j < ny; j++) {
      const h = HMAX * (1 - (j + 0.5) / ny), o = (j * nx + i) * 4;
      if (h < prof[i]) { img.data.set([0, 0, 0, 0], o); continue; }
      const d = dbzIn(px, pz, h);
      if (d < soglia - 2) { img.data.set([0, 0, 0, 0], o); continue; }
      const c = coloreDbz(d), a = Math.min(1, (d - soglia + 2) / 5);
      img.data.set([c[0], c[1], c[2], Math.round(255 * a)], o);
    }
  }
  const tmp = document.createElement("canvas"); tmp.width = nx; tmp.height = ny; tmp.getContext("2d").putImageData(img, 0, 0);
  // righe delle quote
  x.font = "600 10.5px ui-monospace, Menlo, Consolas, monospace"; x.textBaseline = "middle"; x.textAlign = "right";
  for (let k = 0; k <= 10; k += 1) {
    const y = m.a + ph * (1 - k / HMAX);
    x.strokeStyle = k % 2 ? "#13243a" : "#1d3552"; x.lineWidth = 1; x.beginPath(); x.moveTo(m.s, y); x.lineTo(m.s + pw, y); x.stroke();
    if (k % 2 === 0 || k === 1) { x.fillStyle = "#8fb2c8"; x.fillText(k + (k ? " km" : ""), m.s - 6, y); }
  }
  x.imageSmoothingEnabled = true; x.drawImage(tmp, m.s, m.a, pw, ph);
  // rilievo
  x.beginPath(); x.moveTo(m.s, m.a + ph);
  for (let i = 0; i < nx; i++) x.lineTo(m.s + (i + 0.5) / nx * pw, m.a + ph * (1 - prof[i] / HMAX));
  x.lineTo(m.s + pw, m.a + ph); x.closePath();
  const gr = x.createLinearGradient(0, m.a + ph * 0.6, 0, m.a + ph); gr.addColorStop(0, "#5b4a35"); gr.addColorStop(1, "#2a2219");
  x.fillStyle = gr; x.fill(); x.strokeStyle = "#a88d65"; x.lineWidth = 1; x.stroke();
  // asse orizzontale in km e punti cardinali
  x.textAlign = "center"; x.textBaseline = "top"; x.fillStyle = "#8fb2c8";
  for (let k = 0; k <= lung; k += 25) x.fillText(k + " km", m.s + k / lung * pw, m.a + ph + 6);
  x.font = "800 12px system-ui, sans-serif"; x.fillStyle = "#6fe3ff"; x.textBaseline = "middle";
  x.textAlign = "left"; x.fillText(asse === 0 ? "OVEST" : "NORD", m.s + 4, 13);
  x.textAlign = "right"; x.fillText(asse === 0 ? "EST" : "SUD", m.s + pw - 4, 13);
  x.textAlign = "center"; x.fillStyle = "#dff6ff"; x.font = "700 11.5px system-ui, sans-serif";
  x.fillText("Sezione verticale " + (asse === 0 ? "Ovest → Est" : "Nord → Sud") + " · " + testoPosizione(), m.s + pw / 2, 13);
  // Nerola, se il taglio ci passa vicino
  const [nxk, nzk] = inXZ(NERO_LON, NERO_LAT);
  if (Math.abs((asse === 0 ? nzk : nxk) - taglio) < 4) {
    const s = asse === 0 ? nxk : nzk, xx = m.s + (s + lung / 2) / lung * pw;
    x.strokeStyle = "#ff5a6a"; x.setLineDash([3, 3]); x.beginPath(); x.moveTo(xx, m.a); x.lineTo(xx, m.a + ph); x.stroke(); x.setLineDash([]);
    x.fillStyle = "#ff5a6a"; x.font = "700 11px system-ui, sans-serif"; x.fillText("Nerola", xx, m.a + 8);
  }
}

// ---------------------------------------------------------------- rilievo e mappa
function caricaImg(src) { return new Promise((ok, no) => { const im = new Image(); im.onload = () => ok(im); im.onerror = no; im.src = src; }); }
async function preparaTexGrossa() {
  const v = "?v=" + info.ora_ms;
  const [m, c] = await Promise.all([caricaImg(info.mappa + v), caricaImg((info.confini || "lazio/confini.png") + v).catch(() => null)]);
  const cv = document.createElement("canvas"); cv.width = m.naturalWidth * 2; cv.height = m.naturalHeight * 2;
  const x = cv.getContext("2d"); x.drawImage(m, 0, 0, cv.width, cv.height); if (c) x.drawImage(c, 0, 0, cv.width, cv.height);
  texGrossa = new THREE.CanvasTexture(cv); texGrossa.colorSpace = THREE.SRGBColorSpace; texGrossa.anisotropy = 4;
}
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
const TINTA = 0x9aa8bb;     // la mappa un po' scurita, cosi' i colori del radar risaltano sullo sfondo notturno
function costruisciTerreno() {
  const g = geo(), NT = (info.dettaglio && info.dettaglio.tessere) || 4;
  tessere.forEach((t) => { scena.remove(t.grossa); t.grossa.geometry.dispose(); if (t.fine) { scena.remove(t.fine); t.fine.geometry.dispose(); } });
  const vecchie = tessere; tessere = [];
  const mat = new THREE.MeshLambertMaterial({ map: texGrossa, color: TINTA });
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
  t.fine = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({ map: d.tex, color: TINTA })); t.fine.visible = false; scena.add(t.fine);
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

// ---------------------------------------------------------------- scritte, asse delle quote
function etichetta(testo, forte, piccola) {
  const c = document.createElement("canvas"), x = c.getContext("2d"), f = forte ? 44 : (piccola ? 28 : 34);
  const font = (piccola ? "600 " : "700 ") + f + "px " + (piccola ? "ui-monospace, Menlo, Consolas, monospace" : "system-ui, sans-serif");
  x.font = font; c.width = Math.ceil(x.measureText(testo).width) + 24; c.height = f + 20;
  x.font = font; x.textBaseline = "middle";
  x.lineWidth = 8; x.strokeStyle = "rgba(4,10,20,.85)"; x.strokeText(testo, 12, c.height / 2);
  x.fillStyle = forte ? "#ff5a6a" : (piccola ? "#9fdcff" : "#eef6ff"); x.fillText(testo, 12, c.height / 2);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false, sizeAttenuation: false }));
  const h = forte ? 0.042 : (piccola ? 0.026 : 0.032); s.scale.set(h * c.width / c.height, h, 1); s.renderOrder = 10;
  return s;
}
function costruisciEtichette() {
  etichette.forEach((e) => scena.remove(e)); etichette = [];
  PAESI.forEach(([n, lo, la, forte]) => {
    const [x, z] = inXZ(lo, la), y = quotaIn(x, z) * esag;
    const s = etichetta(n, forte); s.center.set(0.5, 0); s.position.set(x, y + 0.6, z); scena.add(s); etichette.push(s);
    const punto = new THREE.Mesh(new THREE.SphereGeometry(0.35 * Math.max(1, esag / 4), 12, 8), new THREE.MeshBasicMaterial({ color: forte ? 0xff5a6a : 0xeef6ff }));
    punto.position.set(x, y + 0.3, z); scena.add(punto); etichette.push(punto);
  });
}
// asse verticale delle quote (1-10 km) all'angolo sud-ovest della zona con eco (o di Nerola se non piove)
function costruisciAsse() {
  if (asseQuote) { scena.remove(asseQuote); asseQuote.traverse((o) => { if (o.geometry) o.geometry.dispose(); }); }
  const g = geo(); asseQuote = new THREE.Group();
  let [x, z] = inXZ(NERO_LON, NERO_LAT); x -= 45; z += 30;      // a sud-ovest di Nerola, sempre in vista
  const pts = [new THREE.Vector3(x, 0, z), new THREE.Vector3(x, 10 * esag, z)];
  for (let k = 1; k <= 10; k++) { pts.push(new THREE.Vector3(x, k * esag, z), new THREE.Vector3(x + (k % 5 ? 1.5 : 3), k * esag, z)); }
  asseQuote.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x6fe3ff, transparent: true, opacity: 0.8, depthTest: false })));
  for (let k = 1; k <= 10; k++) {
    if (esag < 4 && k % 2) continue;
    const s = etichetta(k + " km", false, true); s.center.set(1.08, 0.5); s.position.set(x, k * esag, z); asseQuote.add(s);
  }
  asseQuote.renderOrder = 9; scena.add(asseQuote);
}

// ---------------------------------------------------------------- vista e disegno
function vistaIniziale() {
  const [x, z] = inXZ(NERO_LON, NERO_LAT);
  ctr.target.set(x, 3 * esag / 3, z);
  cam.position.set(x - 80, 70 + 5 * esag, z + 130);
  ctr.update(); lato(); disegna();
}
// qualita' automatica: se il dispositivo fatica (meno di ~20 immagini al secondo mentre giri la vista) si disegna a
// risoluzione piu' bassa, se va veloce si torna nitidi
const PROVA = /[?&]prova/.test(location.search);
const RMAX = PROVA ? 0.35 : Math.min(1.5, window.devicePixelRatio || 1);
let ratio = PROVA ? 0.35 : Math.min(RMAX, /Mobi|Android/i.test(navigator.userAgent) ? 0.8 : 1), ultimo = 0, media = 16;
function disegna() {
  if (!rend || !attivo) return;
  const ora = performance.now(), d = ora - ultimo; ultimo = ora;
  if (d < 250) {
    media = media * 0.8 + d * 0.2;
    const tetto = Math.min(RMAX, Math.sqrt(2.0e6 / Math.max(1, box.clientWidth * box.clientHeight)));
    if (media > 55 && ratio > 0.4) { ratio = Math.max(0.4, ratio * 0.8); rend.setPixelRatio(ratio); media = 30; }
    else if (media < 22 && ratio < tetto) { ratio = Math.min(tetto, ratio * 1.1); rend.setPixelRatio(ratio); media = 30; }
  }
  rend.render(scena, cam);
}
function dimensiona() {
  const w = box.clientWidth, h = box.clientHeight; if (!w || !h) return;
  rend.setSize(w, h); cam.aspect = w / h; cam.updateProjectionMatrix(); disegna(); disegnaSezione();
}
function ricostruisciTutto() { costruisciTerreno(); costruisciVolume(); costruisciEtichette(); costruisciAsse(); lato(); disegna(); disegnaSezione(); }

async function avvia() {
  try { await scarica(); } catch (e) { msg.textContent = "Il radar 3D non è ancora disponibile: riprova tra qualche minuto."; return; }
  scena = new THREE.Scene();
  scena.background = null;
  scena.fog = new THREE.Fog(0x070d18, 420, 1100);
  scena.add(new THREE.HemisphereLight(0xcfe6ff, 0x30343c, 1.5));
  const sole = new THREE.DirectionalLight(0xffffff, 1.3); sole.position.set(-120, 200, 80); scena.add(sole);
  cam = new THREE.PerspectiveCamera(42, 1, 0.5, 3000);
  try {
    rend = new THREE.WebGLRenderer({ antialias: !/Win|Mac|Linux x86/.test(navigator.platform) || (window.devicePixelRatio || 1) < 1.5, alpha: true, powerPreference: "high-performance" });
  } catch (e) { msg.textContent = "Questo dispositivo non supporta la grafica 3D (WebGL 2)."; return; }
  // su schermi grandi si limita il numero di punti da calcolare (ogni punto attraversa il volume fino a 256 volte)
  const wb = box.clientWidth || 1000, hb = box.clientHeight || 600;
  ratio = Math.min(ratio, Math.sqrt(1.1e6 / (wb * hb)));
  rend.setPixelRatio(ratio);
  rend.setClearColor(0x000000, 0);
  rend.domElement.addEventListener("webglcontextlost", (e) => {
    e.preventDefault(); attivo = false;
    const m = document.createElement("div"); m.style.cssText = "position:absolute;inset:auto 0 45%;text-align:center;color:#dff6ff;font-weight:600;padding:12px";
    m.textContent = "La scheda grafica si è fermata: ricarica la pagina per riaprire il radar 3D."; box.appendChild(m);
  });
  box.appendChild(rend.domElement); msg.remove();
  ctr = new OrbitControls(cam, rend.domElement);
  ctr.enableDamping = true; ctr.dampingFactor = 0.08; ctr.maxPolarAngle = Math.PI / 2.02; ctr.minDistance = 6; ctr.maxDistance = 700;
  ctr.addEventListener("change", () => { livelloDettaglio(); lato(); disegna(); });
  await preparaTexGrossa();
  ricostruisciTutto(); vistaIniziale();
  new ResizeObserver(dimensiona).observe(box); dimensiona();
  new ResizeObserver(disegnaSezione).observe($("t3-sezbox"));
  rend.setAnimationLoop(() => { if (attivo && ctr.update()) disegna(); });
  setInterval(async () => {
    if (!attivo || document.hidden) return;
    try { if (await scarica()) { costruisciVolume(); costruisciAsse(); disegna(); disegnaSezione(); } } catch (e) { /* riprova dopo */ }
  }, 300000);
}

// ---------------------------------------------------------------- comandi
$("t3-soglia").addEventListener("input", (e) => {
  soglia = +e.target.value; $("t3-soglia-v").textContent = soglia + " dBZ";
  $("t3-soglia-mm").textContent = "(circa " + String(Math.round(mmh(soglia) * 10) / 10).replace(".", ",") + " mm/h e oltre)";
  if (info && volMat) { volMat.uniforms.uSoglia.value = soglia; disegna(); disegnaSezione(); }
});
$("t3-esag").addEventListener("input", (e) => {
  esag = +e.target.value; $("t3-esag-v").textContent = esag === 1 ? "reale" : "×" + esag;
  if (info && scena) ricostruisciTutto();
});
document.querySelectorAll("[data-modo3]").forEach((b) => b.addEventListener("click", () => {
  modo = +b.dataset.modo3;
  document.querySelectorAll("[data-modo3]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  $("t3-tagliobox").style.display = modo === 3 ? "flex" : "none";
  $("t3-sezbox").style.display = modo === 3 ? "block" : "none";
  if (modo === 3 && info) $("t3-pos-v").textContent = testoPosizione();
  if (info && volMat) { costruisciVolume(); lato(); disegna(); disegnaSezione(); }
}));
function aggiornaTaglio() {
  if (!info) return;
  $("t3-pos-v").textContent = testoPosizione();
  if (volMat) { volMat.uniforms.uTaglio.value = posizioneTaglio(); volMat.uniforms.uAsse.value = asse; disegna(); }
  disegnaSezione();
}
$("t3-pos").addEventListener("input", (e) => { pos = +e.target.value / 1000; aggiornaTaglio(); });
$("t3-asse").addEventListener("change", (e) => {
  asse = +e.target.value;
  if (info) {   // il taglio parte sempre da Nerola
    const [l0, l1, a0, a1] = info.riquadro;
    pos = asse === 0 ? (a1 - NERO_LAT) / (a1 - a0) : (NERO_LON - l0) / (l1 - l0);
    $("t3-pos").value = Math.round(pos * 1000);
  }
  aggiornaTaglio();
});
$("t3-centro").addEventListener("click", () => { if (info) vistaIniziale(); });
function pieno(si) { box.classList.toggle("pieno", si); document.body.style.overflow = si ? "hidden" : ""; setTimeout(dimensiona, 50); }
$("t3-pieno").addEventListener("click", () => pieno(!box.classList.contains("pieno")));
$("t3-chiudi").addEventListener("click", () => pieno(false));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && box.classList.contains("pieno")) pieno(false); });
// legenda continua in dBZ
(function () {
  const b = $("t3-scala"); if (!b) return;
  const p = (d) => ((d - 5) / 60 * 100).toFixed(1);
  b.style.background = "linear-gradient(90deg," + SCALA.filter(([d]) => d >= 5 && d <= 65).map(([d, c]) => c + " " + p(d) + "%").join(",") + ")";
  b.innerHTML = [5, 15, 25, 35, 45, 55, 65].map((d) => '<span style="left:' + p(d) + '%">' + d + "</span>").join("");
})();
$("t3-soglia").dispatchEvent(new Event("input"));
// posizione iniziale del taglio: passa per Nerola
(function () { const a1 = 43.2, a0 = 41.05; pos = (a1 - NERO_LAT) / (a1 - a0); $("t3-pos").value = Math.round(pos * 1000); })();

window.mnRadar3D = { mostra() { attivo = true; if (rend) dimensiona(); }, pausa() { attivo = false; } };
avvia();
