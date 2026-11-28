/*
 * MeteoNerola – GIF animate fatte nel browser (nessun servizio esterno).
 * MNGif.finestra({...}) apre una finestrella per scegliere un tratto della sequenza (da / a), la velocità e la
 * grandezza, poi crea la GIF e la fa scaricare. Usata da webcam, radar e mappe dei modelli.
 * Il codificatore è scritto qui: tavolozza unica da 256 colori (taglio mediano) e compressione LZW standard.
 */
(function () {
	'use strict';
	if (window.MNGif) return;

	// ------------------------------------------------------------------ tavolozza (taglio mediano)
	function tavolozza(campioni, n) {
		var scatole = [campioni];
		while (scatole.length < n) {
			var mi = -1, mr = -1, mc = 0;
			for (var i = 0; i < scatole.length; i++) {
				var s = scatole[i]; if (s.length < 2) continue;
				for (var c = 0; c < 3; c++) {
					var lo = 255, hi = 0;
					for (var j = 0; j < s.length; j++) { var v = s[j][c]; if (v < lo) lo = v; if (v > hi) hi = v; }
					if (hi - lo > mr) { mr = hi - lo; mi = i; mc = c; }
				}
			}
			if (mi < 0 || mr <= 0) break;
			var b = scatole[mi]; b.sort(function (x, y) { return x[mc] - y[mc]; });
			var meta = b.length >> 1;
			scatole.splice(mi, 1, b.slice(0, meta), b.slice(meta));
		}
		var pal = scatole.map(function (s) {
			var r = 0, g = 0, bb = 0; for (var j = 0; j < s.length; j++) { r += s[j][0]; g += s[j][1]; bb += s[j][2]; }
			return [Math.round(r / s.length), Math.round(g / s.length), Math.round(bb / s.length)];
		});
		while (pal.length < n) pal.push([0, 0, 0]);
		return pal;
	}
	function indicizza(dati, pal, cache) {
		var out = new Uint8Array(dati.length / 4);
		for (var i = 0, p = 0; i < dati.length; i += 4, p++) {
			var k = ((dati[i] >> 3) << 10) | ((dati[i + 1] >> 3) << 5) | (dati[i + 2] >> 3);
			var v = cache[k];
			if (v === undefined) {
				var r = dati[i], g = dati[i + 1], b = dati[i + 2], best = 0, bd = 1e9;
				for (var c = 0; c < pal.length; c++) {
					var dr = r - pal[c][0], dg = g - pal[c][1], db = b - pal[c][2], d = 2 * dr * dr + 4 * dg * dg + 3 * db * db;
					if (d < bd) { bd = d; best = c; }
				}
				v = cache[k] = best;
			}
			out[p] = v;
		}
		return out;
	}

	// ------------------------------------------------------------------ scrittura del file GIF
	function Uscita() { this.b = []; }
	Uscita.prototype.byte = function (v) { this.b.push(v & 255); };
	Uscita.prototype.parola = function (v) { this.byte(v); this.byte(v >> 8); };
	Uscita.prototype.testo = function (s) { for (var i = 0; i < s.length; i++) this.byte(s.charCodeAt(i)); };

	function lzw(o, ind) {
		var minimo = 8, clear = 256, fine = 257;
		o.byte(minimo);
		var blocco = [], acc = 0, nbit = 0;
		function scrivi(codice, lung) {
			acc |= codice << nbit; nbit += lung;
			while (nbit >= 8) { blocco.push(acc & 255); acc >>>= 8; nbit -= 8; if (blocco.length === 255) { o.byte(255); for (var i = 0; i < 255; i++) o.byte(blocco[i]); blocco = []; } }
		}
		var dict = new Map(), prossimo = 258, lung = 9;
		scrivi(clear, lung);
		var w = ind[0];
		for (var i = 1; i < ind.length; i++) {
			var c = ind[i], chiave = w * 256 + c, x = dict.get(chiave);
			if (x !== undefined) { w = x; continue; }
			scrivi(w, lung);
			if (prossimo < 4096) {
				dict.set(chiave, prossimo++);
				if (prossimo > (1 << lung) && lung < 12) lung++;
			} else {
				scrivi(clear, lung); dict.clear(); prossimo = 258; lung = 9;
			}
			w = c;
		}
		scrivi(w, lung); scrivi(fine, lung);
		if (nbit > 0) blocco.push(acc & 255);
		for (var s = 0; s < blocco.length; s += 255) {
			var pezzo = blocco.slice(s, s + 255); o.byte(pezzo.length); for (var j = 0; j < pezzo.length; j++) o.byte(pezzo[j]);
		}
		o.byte(0);
	}

	// fotogrammi: array di ImageData della stessa misura; ritardo in millisecondi; pausaFinale: ms sull'ultimo
	function codifica(fotogrammi, ritardo, pausaFinale) {
		var W = fotogrammi[0].width, H = fotogrammi[0].height;
		var campioni = [], passo = Math.max(1, Math.floor(W * H * fotogrammi.length / 60000));
		fotogrammi.forEach(function (f) { for (var i = 0; i < f.data.length; i += 4 * passo) campioni.push([f.data[i], f.data[i + 1], f.data[i + 2]]); });
		var pal = tavolozza(campioni, 256), cache = {};
		var o = new Uscita();
		o.testo('GIF89a'); o.parola(W); o.parola(H); o.byte(0xF7); o.byte(0); o.byte(0);
		pal.forEach(function (c) { o.byte(c[0]); o.byte(c[1]); o.byte(c[2]); });
		o.byte(0x21); o.byte(0xFF); o.byte(11); o.testo('NETSCAPE2.0'); o.byte(3); o.byte(1); o.parola(0); o.byte(0);   // ripete all'infinito
		fotogrammi.forEach(function (f, k) {
			var cs = Math.round((k === fotogrammi.length - 1 && pausaFinale ? pausaFinale : ritardo) / 10);
			o.byte(0x21); o.byte(0xF9); o.byte(4); o.byte(0); o.parola(cs); o.byte(0); o.byte(0);
			o.byte(0x2C); o.parola(0); o.parola(0); o.parola(W); o.parola(H); o.byte(0);
			lzw(o, indicizza(f.data, pal, cache));
		});
		o.byte(0x3B);
		return new Blob([new Uint8Array(o.b)], { type: 'image/gif' });
	}

	// ------------------------------------------------------------------ finestrella di scelta
	var STILE = '.mngif{position:fixed;inset:0;z-index:99999;background:rgba(8,16,24,.55);display:flex;align-items:center;justify-content:center;padding:16px;font:15px/1.4 system-ui,-apple-system,Segoe UI,sans-serif}' +
		'.mngif__box{background:#fff;color:#14303f;border-radius:16px;max-width:440px;width:100%;padding:18px 18px 16px;box-shadow:0 20px 50px rgba(0,0,0,.35)}' +
		'.mngif h3{margin:0 0 4px;font-size:1.15rem}.mngif p{margin:0 0 12px;color:#56686f;font-size:.88rem}' +
		'.mngif label{display:grid;gap:3px;font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:#56686f;margin-bottom:10px}' +
		'.mngif select{font:inherit;font-size:.95rem;text-transform:none;letter-spacing:0;font-weight:400;color:#14303f;padding:8px 10px;border:1px solid #d2dbdc;border-radius:10px;background:#fff;width:100%;min-width:0}' +
		'.mngif__riga{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px}.mngif__bott{display:flex;gap:8px;justify-content:flex-end;margin-top:6px}' +
		'.mngif button{font:inherit;font-weight:700;padding:9px 16px;border-radius:10px;border:1px solid #d2dbdc;background:#fff;color:#14303f;cursor:pointer}' +
		'.mngif button.pieno{background:#14303f;color:#fff;border-color:#14303f}.mngif button:disabled{opacity:.5;cursor:default}' +
		'.mngif__stato{font-size:.88rem;color:#14303f;min-height:1.3em;margin:4px 0 0}.mngif__bar{height:6px;background:#e6eae8;border-radius:3px;overflow:hidden;margin-top:6px}' +
		'.mngif__bar span{display:block;height:100%;width:0;background:#24609a;transition:width .2s}';
	function el(tag, attr, testo) { var e = document.createElement(tag); for (var k in attr || {}) e.setAttribute(k, attr[k]); if (testo) e.textContent = testo; return e; }

	/*
	 * opz: titolo, etichette (array di testi, uno per fotogramma), da, a (indici proposti), larghezze (es. [480,720]),
	 *      disegna(i, canvas) -> Promise (deve disegnare il fotogramma i su un canvas della misura data),
	 *      misura(larghezza) -> {w, h} (misura del canvas), nomeFile, maxFotogrammi (default 120)
	 */
	function finestra(opz) {
		if (!document.getElementById('mngif-stile')) { var st = el('style', { id: 'mngif-stile' }); st.textContent = STILE; document.head.appendChild(st); }
		var n = opz.etichette.length, max = opz.maxFotogrammi || 120;
		var fondo = el('div', { 'class': 'mngif', role: 'dialog', 'aria-modal': 'true' }), box = el('div', { 'class': 'mngif__box' });
		box.appendChild(el('h3', null, '🎞️ ' + (opz.titolo || 'Crea una GIF animata')));
		box.appendChild(el('p', null, 'Scegli il tratto da mettere nella GIF: verrà scaricata sul tuo dispositivo e potrai condividerla dove vuoi.'));
		function scelta(nome, valore) {
			var l = el('label', null, nome), s = el('select');
			opz.etichette.forEach(function (t, i) { var o = el('option', { value: i }, t); s.appendChild(o); });
			s.value = valore; l.appendChild(s); return { l: l, s: s };
		}
		var r1 = el('div', { 'class': 'mngif__riga' });
		var da = scelta('Da', Math.max(0, Math.min(n - 1, opz.da || 0))), a = scelta('A', Math.max(0, Math.min(n - 1, opz.a == null ? n - 1 : opz.a)));
		r1.appendChild(da.l); r1.appendChild(a.l); box.appendChild(r1);
		var r2 = el('div', { 'class': 'mngif__riga' });
		var vl = el('label', null, 'Velocità'), vs = el('select');
		[['lenta', 700], ['normale', 350], ['veloce', 160]].forEach(function (x) { var o = el('option', { value: x[1] }, x[0]); vs.appendChild(o); });
		vs.value = String(opz.ritardo || 350); vl.appendChild(vs); r2.appendChild(vl);
		var gl = el('label', null, 'Grandezza'), gs = el('select');
		(opz.larghezze || [480, 720, 1024]).forEach(function (w) { gs.appendChild(el('option', { value: w }, w + ' px' + (w <= 480 ? ' (leggera)' : ''))); });
		gs.value = String((opz.larghezze || [480, 720])[Math.min(1, (opz.larghezze || [480, 720]).length - 1)]);
		gl.appendChild(gs); r2.appendChild(gl); box.appendChild(r2);
		var stato = el('p', { 'class': 'mngif__stato' }), barra = el('div', { 'class': 'mngif__bar' }), riemp = el('span');
		barra.appendChild(riemp); box.appendChild(stato); box.appendChild(barra);
		var bott = el('div', { 'class': 'mngif__bott' }), annulla = el('button', { type: 'button' }, 'Chiudi'), crea = el('button', { type: 'button', 'class': 'pieno' }, 'Crea GIF');
		bott.appendChild(annulla); bott.appendChild(crea); box.appendChild(bott);
		fondo.appendChild(box); document.body.appendChild(fondo);
		var fermo = false;
		function chiudi() { fermo = true; fondo.remove(); }
		annulla.onclick = chiudi; fondo.addEventListener('click', function (e) { if (e.target === fondo) chiudi(); });
		function conta() {
			var i0 = +da.s.value, i1 = +a.s.value, k = Math.abs(i1 - i0) + 1, salto = Math.max(1, Math.ceil(k / max));
			stato.textContent = Math.ceil(k / salto) + ' fotogrammi' + (salto > 1 ? ' (uno ogni ' + salto + ', per tenerla leggera)' : '');
		}
		da.s.onchange = conta; a.s.onchange = conta; conta();
		crea.onclick = async function () {
			crea.disabled = true; da.s.disabled = a.s.disabled = vs.disabled = gs.disabled = true;
			try {
				var i0 = Math.min(+da.s.value, +a.s.value), i1 = Math.max(+da.s.value, +a.s.value);
				var salto = Math.max(1, Math.ceil((i1 - i0 + 1) / max)), indici = [];
				for (var i = i0; i <= i1; i += salto) indici.push(i);
				var m = opz.misura(+gs.value), cv = el('canvas'); cv.width = m.w; cv.height = m.h;
				var ctx = cv.getContext('2d', { willReadFrequently: true }), fot = [];
				for (var k = 0; k < indici.length; k++) {
					if (fermo) return;
					stato.textContent = 'Preparo il fotogramma ' + (k + 1) + ' di ' + indici.length + '…';
					riemp.style.width = Math.round(80 * k / indici.length) + '%';
					ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = 1; ctx.clearRect(0, 0, m.w, m.h);
					await opz.disegna(indici[k], cv);
					fot.push(ctx.getImageData(0, 0, m.w, m.h));
				}
				stato.textContent = 'Creo la GIF…'; riemp.style.width = '85%';
				await new Promise(function (ok) { setTimeout(ok, 30); });
				var blob = codifica(fot, +vs.value, 1500);
				riemp.style.width = '100%';
				var url = URL.createObjectURL(blob), link = el('a', { href: url, download: (opz.nomeFile || 'meteonerola') + '.gif' });
				document.body.appendChild(link); link.click(); link.remove();
				stato.textContent = 'Fatto: GIF di ' + (blob.size / 1048576).toFixed(1).replace('.', ',') + ' MB scaricata.';
				setTimeout(function () { URL.revokeObjectURL(url); }, 60000);
				crea.textContent = 'Creane un’altra';
			} catch (e) {
				stato.textContent = 'Non sono riuscito a creare la GIF (' + e.message + ').';
			}
			crea.disabled = false; da.s.disabled = a.s.disabled = vs.disabled = gs.disabled = false;
		};
	}
	function carica(src) {
		return new Promise(function (ok, no) { var im = new Image(); im.onload = function () { ok(im); }; im.onerror = function () { no(new Error('immagine non caricata')); }; im.src = src; });
	}
	window.MNGif = { codifica: codifica, finestra: finestra, carica: carica };
})();
