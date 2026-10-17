/* ChromaDCNN project site. Every number comes from data.js (scripts/13_build_site_data.py).
   Colours are CSS tokens, so the theme toggle only redraws canvases and the 3D view. */
(() => {
  'use strict';
  const D = window.SITE_DATA;
  const SP = D.species;                                   // ['SNA','GUR','TAR','BCO']
  const SP_NAME = { SNA: 'Snapper', GUR: 'Gurnard', TAR: 'Tarakihi', BCO: 'Blue Cod' };
  const SP_LATIN = { SNA: 'Pagrus auratus', GUR: 'Chelidonichthys kumu', TAR: 'Nemadactylus macropterus', BCO: 'Parapercis colias' };
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const DUR = reduceMotion ? 0 : 700;
  const $ = id => document.getElementById(id);
  const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const spColor = s => `var(--${s})`;
  const fmt3 = d3.format('.3f'), fmt2 = d3.format('.2f'), fmtInt = d3.format(',');
  const mean = a => d3.mean(a);
  const decode = m => m.q.map(r => r.map(v => v / 255 * m.max));
  const renderers = [];                                   // redrawn on resize
  const canvasRenderers = [];                             // redrawn on theme change too
  const isPhone = () => window.innerWidth < 600;

  // ── Tooltip (DOM built with textContent: labels are data) ────────────────
  const tip = $('tooltip');
  function showTip(evt, head, rows) {
    tip.replaceChildren();
    if (head) { const h = document.createElement('div'); h.className = 'tt-head'; h.textContent = head; tip.append(h); }
    for (const r of rows) {
      const row = document.createElement('div'); row.className = 'row';
      const key = document.createElement('span'); key.className = 'key';
      if (r.color) { const i = document.createElement('i'); i.style.background = r.color; key.append(i); }
      key.append(document.createTextNode(r.label));
      const v = document.createElement('b'); v.textContent = r.value;
      row.append(key, v); tip.append(row);
    }
    const x = evt.clientX, y = evt.clientY, w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = `${Math.min(x + 14, window.innerWidth - w - 8)}px`;
    tip.style.top = `${y + 14 + h > window.innerHeight ? y - h - 10 : y + 14}px`;
    tip.classList.add('on');
  }
  const hideTip = () => tip.classList.remove('on');

  // ── Chart scaffolding ─────────────────────────────────────────────────────
  function frame(id, height, margin = { top: 12, right: 16, bottom: 34, left: 48 }) {
    const el = $(id); el.replaceChildren();
    const width = Math.max(280, el.clientWidth || 600);
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${width} ${height}`).attr('preserveAspectRatio', 'xMidYMid meet');
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    return { el, svg, g, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom, width, height, margin };
  }
  function axes(g, x, y, w, h, { xTitle, yTitle, xTicks = 8, yTicks = 5, yFmt, xFmt, grid = 'y' } = {}) {
    if (grid === 'y' || grid === 'both') g.append('g').attr('class', 'gridline').call(d3.axisLeft(y).ticks(yTicks).tickSize(-w).tickFormat(''));
    if (grid === 'x' || grid === 'both') g.append('g').attr('class', 'gridline').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(xTicks).tickSize(-h).tickFormat(''));
    g.append('g').attr('class', 'axis').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(xTicks).tickSizeOuter(0).tickFormat(xFmt || null));
    g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(yTicks).tickSizeOuter(0).tickFormat(yFmt || null)).call(s => s.select('.domain').remove());
    if (xTitle) g.append('text').attr('class', 'axis-title').attr('x', w).attr('y', h + 30).attr('text-anchor', 'end').text(xTitle);
    if (yTitle) g.append('text').attr('class', 'axis-title').attr('x', 0).attr('y', -4).attr('text-anchor', 'start').text(yTitle);
  }
  function legend(id, items) {
    const el = $(id); if (!el) return;
    el.replaceChildren();
    for (const it of items) {
      const s = document.createElement('span'); const i = document.createElement('i');
      if (it.kind) i.className = it.kind;
      i.style.background = it.color;
      if (it.dash) i.style.background = `repeating-linear-gradient(90deg, ${it.color} 0 4px, transparent 4px 7px)`;
      s.append(i, document.createTextNode(it.label)); el.append(s);
    }
  }
  function segmented(el, items, onPick, initial) {
    el.replaceChildren();
    items.forEach(([key, label, swatch]) => {
      const b = document.createElement('button');
      b.type = 'button'; b.dataset.key = key;
      if (swatch) { const i = document.createElement('i'); i.className = 'sw'; i.style.background = swatch; b.append(i); }
      b.append(document.createTextNode(label));
      b.setAttribute('aria-pressed', String(key === initial));
      b.onclick = () => { [...el.children].forEach(c => c.setAttribute('aria-pressed', String(c === b))); onPick(key); };
      el.appendChild(b);
    });
  }
  function bindSeg(id, attr, onPick) {
    const el = $(id);
    el.querySelectorAll('button').forEach(b => b.onclick = () => {
      el.querySelectorAll('button').forEach(c => c.setAttribute('aria-pressed', String(c === b)));
      onPick(b.dataset[attr]);
    });
  }
  const jitter = i => ((i * 7919) % 97) / 97 - 0.5;       // deterministic, so points never move on redraw

  // ── Canvas heatmaps ───────────────────────────────────────────────────────
  function makeLut(stops) {
    const f = d3.piecewise(d3.interpolateRgb, stops);
    return d3.range(256).map(i => { const c = d3.color(f(i / 255)); return [c.r, c.g, c.b]; });
  }
  const seqLut = () => makeLut(isDarkMode() ? [cssVar('--seq-0'), '#1a2740', '#193661', '#1c4d8f', '#2a78d6', '#6da7ec', '#cde2fb'] : ['--seq-0', '--seq-1', '--seq-2', '--seq-3', '--seq-4', '--seq-5'].map(cssVar));
  const isDarkMode = () => (document.documentElement.dataset.theme ? document.documentElement.dataset.theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches);
  function matrixCanvas(get, nx, ny, lut) {               // get(x, y) in [0,1]; y = 0 is the bottom row
    const c = document.createElement('canvas'); c.width = nx; c.height = ny;
    const ctx = c.getContext('2d'); const img = ctx.createImageData(nx, ny); const d = img.data;
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const v = Math.max(0, Math.min(1, get(x, y))); const [r, g, b] = lut[Math.round(v * 255)];
      const o = ((ny - 1 - y) * nx + x) * 4; d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0); return c;
  }
  function blit(canvas, small) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 300, h = canvas.clientHeight || 200;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d'); ctx.imageSmoothingEnabled = false;
    ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
    return ctx;
  }

  // ── Data shortcuts ────────────────────────────────────────────────────────
  const S = D.saliency;
  const EX = S.examples.map(e => {
    const Z = decode(e.chroma); const mx = d3.max(Z, r => d3.max(r));
    const Zn = Z.map(r => r.map(v => v / mx));
    const tic = Zn.map(r => d3.sum(r)); const tmx = d3.max(tic);
    const Sa = decode(e.saliency); const smx = d3.max(Sa, r => d3.max(r));
    return { species: e.species, sample: e.sample.replace(/^\d+ - /, ''), Z: Zn, tic: tic.map(v => v / tmx), S: Sa.map(r => r.map(v => v / smx)) };
  });
  const RT = S.grid_rt, MZ = S.grid_mz;                   // 100 RT centres (min), 200 m/z values
  const M = D.methods;
  const best = M[0];
  const scratch = M.find(m => m.method.includes('from_scratch'));
  const bestBaseline = M.find(m => !m.method.startsWith('ChromaDCNN'));
  const label = m => m.method.startsWith('ChromaDCNN') ? (m.method.includes('pretrain') ? 'ChromaDCNN, pretrained' : 'ChromaDCNN, from scratch') : m.method;
  const REPS = ['2D chromatogram', 'chroma-PCA', 'sum spectrum', 'max-projection'];
  const REP_VAR = { '2D chromatogram': '--rep-2d', 'chroma-PCA': '--rep-pca', 'sum spectrum': '--rep-sum', 'max-projection': '--rep-max' };
  const repColor = r => `var(${REP_VAR[r]})`;

  // ── Hero: the model's view of one run ─────────────────────────────────────
  const hero = { sp: 0, i: reduceMotion ? 62 : 0, playing: !reduceMotion, hover: false, last: 0 };
  function heroRender() {
    const ex = EX[hero.sp];
    blit($('hero-canvas'), matrixCanvas((x, y) => Math.pow(ex.Z[x][y], 0.5), 100, 200, seqLut()));
    const el = $('hero-tic'); const w = el.clientWidth || 400, h = el.clientHeight || 70;
    const svg = d3.select(el).attr('viewBox', `0 0 ${w} ${h}`); svg.selectAll('*').remove();
    const x = d3.scaleLinear([0, 99], [0, w]), y = d3.scaleLinear([0, 1], [h - 2, 6]);
    svg.append('path').attr('d', d3.area().x((d, i) => x(i)).y0(h - 2).y1(d => y(d)).curve(d3.curveMonotoneX)(ex.tic))
      .attr('fill', spColor(ex.species)).attr('opacity', 0.12);
    svg.append('path').attr('d', d3.line().x((d, i) => x(i)).y(d => y(d)).curve(d3.curveMonotoneX)(ex.tic))
      .attr('fill', 'none').attr('stroke', spColor(ex.species)).attr('stroke-width', 2).attr('stroke-linejoin', 'round');
    svg.append('text').attr('x', 2).attr('y', 12).attr('class', 'axis-title').text('summed √intensity');
    svg.append('circle').attr('id', 'hero-ticdot').attr('r', 5).attr('fill', spColor(ex.species)).attr('stroke', 'var(--surface)').attr('stroke-width', 2);
    hero.x = x; hero.y = y;
    $('hero-dot').style.background = spColor(ex.species);
    $('hero-state').textContent = `Reading a ${SP_NAME[ex.species]} run, bin by bin`;
    $('hero-caption').innerHTML = `Sample “${ex.sample}” (<i>${SP_LATIN[ex.species]}</i>). Hover the heatmap to scrub. Retention time runs left to right, <i>m/z</i> bottom to top; the right panel is the spectrum under the playhead.`;
    heroFrame();
  }
  function heroFrame() {
    const ex = EX[hero.sp]; const i = hero.i;
    const heat = $('hero-heat'); const w = heat.clientWidth || 400;
    $('hero-playhead').style.left = `${(i + 0.5) / 100 * w}px`;
    if (hero.x) d3.select('#hero-ticdot').attr('cx', hero.x(i)).attr('cy', hero.y(ex.tic[i]));
    const el = $('hero-spec'); const sw = el.clientWidth || 100, sh = el.clientHeight || 250;
    const svg = d3.select(el).attr('viewBox', `0 0 ${sw} ${sh}`);
    let path = svg.select('path.spec');
    if (path.empty()) {
      svg.append('text').attr('x', 2).attr('y', 12).attr('class', 'axis-title').text('spectrum');
      path = svg.append('path').attr('class', 'spec').attr('stroke', 'var(--accent)').attr('fill', 'none');
    }
    const row = ex.Z[i]; const mx = Math.max(1e-6, d3.max(row));
    let d = '';
    for (let j = 0; j < 200; j++) { const v = row[j] / mx; if (v < 0.01) continue; const yy = sh - (j + 0.5) / 200 * sh; d += `M0,${yy.toFixed(1)}H${(v * (sw - 4)).toFixed(1)}`; }
    path.attr('d', d).attr('stroke-width', Math.max(1, sh / 200 * 0.9));
    $('hero-corner').replaceChildren(Object.assign(document.createElement('span'), { textContent: `${RT[i].toFixed(1)} min` }), Object.assign(document.createElement('span'), { textContent: `bin ${i + 1}` }));
  }
  function heroLoop(ts) {
    if (hero.playing && !hero.hover && ts - hero.last > 60) { hero.i = (hero.i + 1) % 100; hero.last = ts; heroFrame(); }
    requestAnimationFrame(heroLoop);
  }
  segmented($('hero-species'), SP.map((s, i) => [i, s, spColor(s)]), k => { hero.sp = +k; heroRender(); }, 0);
  const heroPlayBtn = $('hero-play');
  const setHeroPlay = p => { hero.playing = p; heroPlayBtn.setAttribute('aria-pressed', String(p)); heroPlayBtn.textContent = p ? '⏸ Pause' : '▶ Play'; };
  setHeroPlay(hero.playing);
  heroPlayBtn.onclick = () => setHeroPlay(!hero.playing);
  $('hero-heat').addEventListener('pointermove', e => {
    const r = e.currentTarget.getBoundingClientRect();
    hero.hover = true; hero.i = Math.max(0, Math.min(99, Math.floor((e.clientX - r.left) / r.width * 100))); heroFrame();
  });
  $('hero-heat').addEventListener('pointerleave', () => { hero.hover = false; });
  canvasRenderers.push(heroRender);
  requestAnimationFrame(heroLoop);

  // ── Count-up stats ────────────────────────────────────────────────────────
  function countUp(el) {
    const target = +el.dataset.count, dec = +el.dataset.dec || 0, prefix = el.dataset.prefix || '';
    const fmt = v => prefix + (dec ? v.toFixed(dec) : fmtInt(Math.round(v)));
    if (reduceMotion) { el.textContent = fmt(target); return; }
    const t0 = performance.now(), dur = 1500;
    const step = now => { const p = Math.min(1, (now - t0) / dur); const e = 1 - Math.pow(1 - p, 3); el.textContent = fmt(target * e); if (p < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  }
  const countObs = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { countUp(e.target); countObs.unobserve(e.target); } }), { threshold: 0.4 });
  document.querySelectorAll('[data-count]').forEach(el => countObs.observe(el));

  // ── 01 Composition ────────────────────────────────────────────────────────
  let compMode = 'part';
  function drawComposition(animate = false) {
    const key = compMode === 'part' ? 'part' : 'individual';
    const src = compMode === 'part' ? D.composition : D.individuals;
    const names = [...new Set(src.map(r => r[key]))];
    const rows = names.map(n => {
      let x0 = 0; const segs = [];
      for (const s of SP) { const n_ = (src.find(r => r[key] === n && r.species === s) || { n: 0 }).n; if (n_) { segs.push({ sp: s, n: n_, x0, x1: x0 + n_ }); x0 += n_; } }
      return { name: n, segs, total: x0 };
    }).sort((a, b) => b.total - a.total);
    const rowH = 34;
    const { g, w, h } = frame('composition', rows.length * rowH + 40, { top: 6, right: 44, bottom: 28, left: compMode === 'part' ? 96 : 56 });
    const x = d3.scaleLinear([0, d3.max(rows, r => r.total)], [0, w]).nice();
    const y = d3.scaleBand(rows.map(r => r.name), [0, rows.length * rowH]).paddingInner(0.3);
    const bh = Math.min(24, y.bandwidth());
    g.append('g').attr('class', 'gridline').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6).tickSize(-h).tickFormat(''));
    g.append('g').attr('class', 'axis').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
    g.append('text').attr('class', 'axis-title').attr('x', w).attr('y', h + 30).attr('text-anchor', 'end').text('samples');
    const row = g.selectAll('g.row').data(rows).join('g').attr('class', 'row').attr('transform', d => `translate(0,${y(d.name) + (y.bandwidth() - bh) / 2})`);
    row.append('text').attr('class', 'row-label').attr('x', -10).attr('y', bh / 2).attr('dominant-baseline', 'middle').attr('text-anchor', 'end').text(d => d.name);
    const seg = row.selectAll('path').data(d => d.segs.map((s, i, a) => Object.assign({ last: i === a.length - 1, first: i === 0, row: d.name }, s))).join('path')
      .attr('fill', d => spColor(d.sp))
      .on('pointermove', (e, d) => showTip(e, d.row, [{ label: SP_NAME[d.sp], value: `${d.n} sample${d.n > 1 ? 's' : ''}`, color: cssVar('--' + d.sp) }]))
      .on('pointerleave', hideTip);
    const shape = (d, t) => {
      const a = x(d.x0) + (d.first ? 0 : 2), b = Math.max(a, x(d.x0 + (d.x1 - d.x0) * t) - (d.first ? 0 : 0)); const r = d.last ? Math.min(4, (b - a) / 2) : 0;
      return `M${a},0H${b - r}a${r},${r} 0 0 1 ${r},${r}V${bh - r}a${r},${r} 0 0 1 ${-r},${r}H${a}Z`;
    };
    if (animate && !reduceMotion) seg.attr('d', d => shape(d, 0.001)).transition().duration(DUR).delay((d, i) => i * 60).attr('d', d => shape(d, 1));
    else seg.attr('d', d => shape(d, 1));
    row.append('text').attr('class', 'row-sub').attr('x', d => x(d.total) + 6).attr('y', bh / 2).attr('dominant-baseline', 'middle').text(d => d.total);
    legend('composition-legend', SP.map(s => ({ label: SP_NAME[s], color: spColor(s), kind: 'dot' })));
    $('composition-caption').textContent = compMode === 'part'
      ? 'Six body parts per fish: frame, gonad, head, liver, rest of guts and skin. Blue Cod contributes 57 of the 103 samples.'
      : 'Six individuals in all: one Snapper, one Gurnard, one Tarakihi and three Blue Cod. Several replicates per body part.';
  }
  bindSeg('comp-mode', 'mode', m => { compMode = m; drawComposition(true); });
  renderers.push(() => drawComposition(false));

  // ── 01 Four traces ────────────────────────────────────────────────────────
  const ticState = { overlay: false };
  function drawTic(animate = false) {
    const overlay = ticState.overlay;
    const H = overlay ? 260 : 300;
    const { g, w, h } = frame('tic-chart', H, { top: 8, right: 12, bottom: 30, left: 30 });
    const x = d3.scaleLinear([5, 45], [0, w]);
    const rows = overlay ? 1 : 4, gap = 8, rowH = (h - gap * (rows - 1)) / rows;
    const y = d3.scaleLinear([0, 1], [rowH, 2]);
    g.append('g').attr('class', 'axis').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(isPhone() ? 4 : 8).tickSizeOuter(0).tickFormat(d => d + ' min'));
    const line = d3.line().x((d, i) => x(RT[i])).y(d => y(d)).curve(d3.curveMonotoneX);
    const grp = g.selectAll('g.sp').data(EX).join('g').attr('class', 'sp')
      .attr('transform', (d, c) => `translate(0,${overlay ? 0 : c * (rowH + gap)})`);
    grp.append('line').attr('x1', 0).attr('x2', w).attr('y1', rowH).attr('y2', rowH).attr('stroke', 'var(--grid)');
    grp.append('path').attr('d', d => line(d.tic)).attr('fill', 'none').attr('stroke', d => spColor(d.species)).attr('stroke-width', 2)
      .attr('stroke-linejoin', 'round').attr('opacity', overlay ? 0.85 : 1)
      .each(function () { if (animate && !reduceMotion) { const L = this.getTotalLength(); d3.select(this).attr('stroke-dasharray', `${L} ${L}`).attr('stroke-dashoffset', L).transition().duration(1400).ease(d3.easeCubicOut).attr('stroke-dashoffset', 0).on('end', () => d3.select(this).attr('stroke-dasharray', null)); } });
    if (!overlay) grp.append('text').attr('class', 'direct-label').attr('x', 4).attr('y', 12).text(d => SP_NAME[d.species]);
    const cross = g.append('line').attr('class', 'crosshair').attr('y1', 0).attr('y2', h).style('opacity', 0);
    g.append('rect').attr('width', w).attr('height', h).attr('fill', 'transparent')
      .on('pointermove', e => {
        const [mx] = d3.pointer(e); const rt = x.invert(mx); const i = Math.max(0, Math.min(99, Math.round((rt - RT[0]) / (RT[1] - RT[0]))));
        cross.attr('x1', x(RT[i])).attr('x2', x(RT[i])).style('opacity', 1);
        showTip(e, `RT ${RT[i].toFixed(1)} min`, EX.map(ex => ({ label: SP_NAME[ex.species], value: fmt2(ex.tic[i]), color: cssVar('--' + ex.species) })));
      }).on('pointerleave', () => { cross.style('opacity', 0); hideTip(); });
    legend('tic-legend', SP.map(s => ({ label: SP_NAME[s], color: spColor(s) })));
  }
  $('tic-overlay').onchange = e => { ticState.overlay = e.target.checked; drawTic(true); };
  renderers.push(() => drawTic(false));

  // ── 02 Pipeline ───────────────────────────────────────────────────────────
  const STAGES = [
    { name: 'Input', shape: '[200 × 1000]', params: 0, desc: 'One chromatogram: 200 retention-time bins (5–45 min, 12 s each) by 1000 m/z channels. Each bin is √-transformed and L2-normalised, so the network sees spectral shape rather than absolute abundance.' },
    { name: 'Spectral embedding', shape: 'Linear 1000 → 128', params: 128128, desc: 'One shared linear layer applied to every bin independently. It learns which ions, and which combinations of ions, matter. This is the layer that pretraining transfers.' },
    { name: 'ResBlock · dilation 1', shape: '[128 × 200] · field 13', params: 230144, desc: 'Two 1D convolutions (kernel 7) with batch-norm, GELU, dropout and a residual connection. Each conv sees 7 neighbouring bins; together they cover 13.' },
    { name: 'ResBlock · dilation 2', shape: '[128 × 200] · field 37', params: 230144, desc: 'Same block, taps spread two bins apart. The receptive field grows to 37 bins (7.4 min) at no extra parameter cost.' },
    { name: 'ResBlock · dilation 4', shape: '[128 × 200] · field 85', params: 230144, desc: 'Taps four bins apart. Every output position now sees 85 bins, about 17 minutes of the 40-minute run.' },
    { name: 'Max ∥ attention pool', shape: '[256]', params: 129, desc: 'Two summaries of the 200 positions: the channel-wise maximum (the strongest evidence anywhere) and a learned soft attention average (where the model chooses to look). They are concatenated.' },
    { name: 'Classifier', shape: 'Linear 256 → 4', params: 1028, desc: 'A linear head over the pooled 256 features gives one logit per species. Training uses a class-weighted cross-entropy loss.' },
  ];
  const TOTAL_PARAMS = d3.sum(STAGES, s => s.params);
  function drawPipeline() {
    const el = $('pipeline'); el.replaceChildren();
    STAGES.forEach((s, i) => {
      if (i) { const a = document.createElement('span'); a.className = 'pipe-arrow'; a.textContent = '→'; el.append(a); }
      const b = document.createElement('button'); b.type = 'button'; b.className = 'pipe-stage'; b.setAttribute('aria-pressed', 'false');
      const nm = document.createElement('span'); nm.className = 'nm'; nm.textContent = s.name;
      const sh = document.createElement('span'); sh.className = 'sh'; sh.textContent = s.shape;
      const bar = document.createElement('span'); bar.className = 'bar'; bar.style.width = `${Math.max(2, s.params / TOTAL_PARAMS * 100)}%`;
      b.append(nm, sh, bar); b.onclick = () => selectStage(i); el.append(b);
    });
    selectStage(1);
  }
  function selectStage(i) {
    document.querySelectorAll('.pipe-stage').forEach((b, j) => b.setAttribute('aria-pressed', String(i === j)));
    const s = STAGES[i]; const d = $('stage-detail'); d.replaceChildren();
    const left = document.createElement('div'); const h = document.createElement('h4'); h.textContent = s.name;
    const p = document.createElement('p'); p.textContent = s.desc; left.append(h, p);
    const right = document.createElement('div'); right.className = 'params';
    const b = document.createElement('b'); b.textContent = s.params ? fmtInt(s.params) : '—';
    const sp = document.createElement('span'); sp.textContent = s.params ? `parameters · ${(s.params / TOTAL_PARAMS * 100).toFixed(1)}% of ${fmtInt(TOTAL_PARAMS)}` : 'no parameters';
    right.append(b, sp); d.append(left, right);
  }
  drawPipeline();

  // ── 02 Receptive field ────────────────────────────────────────────────────
  const RF = { dil: [1, 1, 2, 2, 4, 4], k: 7, n: 200, pos: 120, mode: 'sym' };
  const RF_LABELS = ['input chromatogram (200 bins)', 'block 1 · conv 1 · dilation 1', 'block 1 · conv 2 · dilation 1', 'block 2 · conv 1 · dilation 2', 'block 2 · conv 2 · dilation 2', 'block 3 · conv 1 · dilation 4', 'block 3 · conv 2 · dilation 4 (output)'];
  function rfSets() {
    let set = new Set([RF.pos]); const rows = [new Set(set)];
    for (let l = RF.dil.length - 1; l >= 0; l--) {
      const d = RF.dil[l], next = new Set();
      for (const q of set) for (let j = 0; j < RF.k; j++) { const p = q + (RF.mode === 'sym' ? (j - 3) * d : -j * d); if (p >= 0 && p < RF.n) next.add(p); }
      set = next; rows.push(set);
    }
    return rows.reverse();                                 // rows[0] = input … rows[6] = output
  }
  let rfSvg = null;
  function drawRF() {
    const rowH = 30, cellH = 11, labelH = 13;
    const { g, w, h } = frame('rf-chart', 7 * rowH + 36, { top: 4, right: 8, bottom: 26, left: 8 });
    const cw = w / RF.n;
    const x = d3.scaleLinear([5, 45], [0, w]);
    const rows = g.selectAll('g.rf').data(RF_LABELS).join('g').attr('class', 'rf').attr('transform', (d, i) => `translate(0,${i * rowH})`);
    rows.append('text').attr('class', 'rf-row-label').attr('x', 0).attr('y', 10).text(d => d);
    rows.selectAll('rect').data((d, i) => d3.range(RF.n).map(b => ({ b, r: i }))).join('rect')
      .attr('class', 'rf-cell').attr('x', d => d.b * cw + 0.5).attr('y', labelH + 2).attr('width', Math.max(0.5, cw - 1)).attr('height', cellH).attr('rx', 1);
    g.append('path').attr('class', 'rf-cone').attr('fill', 'var(--accent)').attr('opacity', 0.1);
    g.append('g').attr('class', 'axis').attr('transform', `translate(0,${7 * rowH + 2})`).call(d3.axisBottom(x).ticks(isPhone() ? 4 : 8).tickSizeOuter(0).tickFormat(d => d + ' min'));
    rfSvg = { g, cw, rowH, labelH, cellH };
    g.append('rect').attr('width', w).attr('height', 7 * rowH).attr('fill', 'transparent').style('cursor', 'crosshair')
      .on('pointermove', e => { if (e.buttons || e.pointerType === 'touch') { const [mx] = d3.pointer(e); setRFPos(Math.round(mx / cw)); } })
      .on('pointerdown', e => { const [mx] = d3.pointer(e); setRFPos(Math.round(mx / cw)); });
    updateRF();
  }
  function setRFPos(p) { RF.pos = Math.max(0, Math.min(RF.n - 1, p)); $('rf-pos').value = RF.pos; $('rf-pos-v').textContent = RF.pos; updateRF(); }
  function updateRF() {
    if (!rfSvg) return;
    const sets = rfSets(); const { g, cw, rowH, labelH, cellH } = rfSvg;
    g.selectAll('g.rf').each(function (d, i) {
      d3.select(this).selectAll('rect').attr('class', c => 'rf-cell' + (i === 6 && c.b === RF.pos ? ' out' : sets[i].has(c.b) ? ' on' : ''));
    });
    const ext = sets.map(s => d3.extent([...s]));
    const top = i => i * rowH + labelH + 2, bot = i => top(i) + cellH;
    const left = ext.map((e, i) => `${e[0] * cw},${top(i)} ${e[0] * cw},${bot(i)}`).join(' ');
    const right = ext.map((e, i) => `${(e[1] + 1) * cw},${bot(i)} ${(e[1] + 1) * cw},${top(i)}`).reverse().join(' ');
    g.select('path.rf-cone').attr('d', `M${left.split(' ').join('L')}L${right.split(' ').join('L')}Z`);
    const [a, b] = ext[0]; const n = b - a + 1;
    const rt = p => (5 + p * 0.2).toFixed(1);
    const full = RF.mode === 'sym' ? 85 : 85;
    $('rf-caption').textContent = `Output bin ${RF.pos} (RT ${rt(RF.pos)} min) is computed from input bins ${a}–${b}: ${n} bins, ${(n * 0.2).toFixed(1)} min` +
      (n < full ? ` (clipped by the edge of the run; ${full} bins in the interior)` : '') +
      (RF.mode === 'sym' ? '. Fine-tuning pads both sides, so each position sees 42 bins before and after it.' : '. Pretraining pads only the left, so a position sees the 84 bins before it and nothing after: the next bin can be predicted without peeking.');
  }
  $('rf-pos').oninput = e => setRFPos(+e.target.value);
  bindSeg('rf-mode', 'mode', m => { RF.mode = m; updateRF(); });
  renderers.push(drawRF);

  // ── 02 Synthetic chromatogram + next-frame game ──────────────────────────
  const SYN = { nb: 200, nmz: 300, K: 12, sig: 6, seed: 7, t: 0, playing: !reduceMotion, last: 0, Z: null, tic: null, big: null };
  function synGenerate() {
    const rng = d3.randomLcg(SYN.seed / 1000 + 0.123);
    const K = SYN.K, nb = SYN.nb, nmz = SYN.nmz;
    const spectra = d3.range(K).map(() => {
      const s = new Float32Array(nmz); const np = 4 + Math.floor(rng() * 12);
      for (let p = 0; p < np; p++) { const m = Math.floor(nmz * Math.pow(rng(), 1.4)); s[m] = Math.max(s[m], Math.pow(rng(), 2)); }
      s[Math.floor(nmz * (0.55 + 0.45 * rng()))] = 0.3 + 0.7 * rng();  // a heavier "molecular ion"
      const mx = d3.max(s); for (let i = 0; i < nmz; i++) s[i] /= mx; return s;
    });
    const mu = d3.range(K).map(() => rng() * nb);
    const sig = d3.range(K).map(() => Math.max(1, SYN.sig + (rng() * 2 - 1) * 2.5));
    const amp = d3.range(K).map(() => 0.3 + rng());
    const Z = [], tic = [];
    for (let b = 0; b < nb; b++) {
      const row = new Float32Array(nmz); let tot = 0;
      for (let k = 0; k < K; k++) { const p = amp[k] * Math.exp(-0.5 * ((b - mu[k]) / sig[k]) ** 2); if (p < 1e-4) continue; const s = spectra[k]; for (let m = 0; m < nmz; m++) if (s[m]) { row[m] += p * s[m]; } tot += p; }
      tic.push(tot);
      let n2 = 0; for (let m = 0; m < nmz; m++) { row[m] = Math.sqrt(row[m]); n2 += row[m] * row[m]; }
      n2 = Math.sqrt(n2) || 1; for (let m = 0; m < nmz; m++) row[m] /= n2;
      Z.push(row);
    }
    const zmx = d3.max(Z, r => d3.max(r)); SYN.Z = Z.map(r => Array.from(r, v => v / zmx));
    const tmx = d3.max(tic) || 1; SYN.tic = tic.map(v => v / tmx);
    SYN.t = 0;
  }
  function synRender() {
    if (!SYN.Z) synGenerate();
    const canvas = $('syn-canvas');
    const small = matrixCanvas((x, y) => Math.pow(SYN.Z[x][y], 0.7), SYN.nb, SYN.nmz, seqLut());
    const ctx = blit(canvas, small);
    SYN.big = document.createElement('canvas'); SYN.big.width = canvas.width; SYN.big.height = canvas.height;
    SYN.big.getContext('2d').drawImage(canvas, 0, 0);
    // TIC strip
    const { g, w, h } = frame('syn-tic', 86, { top: 6, right: 8, bottom: 22, left: 8 });
    const x = d3.scaleLinear([0, SYN.nb - 1], [0, w]), y = d3.scaleLinear([0, 1], [h, 4]);
    g.append('path').attr('d', d3.area().x((d, i) => x(i)).y0(h).y1(d => y(d))(SYN.tic)).attr('fill', 'var(--accent)').attr('opacity', 0.12);
    g.append('path').attr('d', d3.line().x((d, i) => x(i)).y(d => y(d))(SYN.tic)).attr('fill', 'none').attr('stroke', 'var(--accent)').attr('stroke-width', 2);
    g.append('g').attr('class', 'axis').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0).tickFormat(d => `bin ${d}`));
    g.append('text').attr('class', 'axis-title').attr('x', 2).attr('y', 10).text('total ion current');
    SYN.marker = g.append('line').attr('class', 'crosshair').attr('y1', 0).attr('y2', h).attr('stroke', 'var(--ink)');
    SYN.x = x;
    synFrame();
  }
  function synFrame() {
    const canvas = $('syn-canvas'); if (!SYN.big) return;
    const ctx = canvas.getContext('2d'); const W = canvas.width, H = canvas.height;
    ctx.drawImage(SYN.big, 0, 0);
    const cw = W / SYN.nb, xt = (SYN.t + 1) * cw;
    ctx.fillStyle = cssVar('--surface'); ctx.globalAlpha = 0.82; ctx.fillRect(xt + cw, 0, W - xt - cw, H); ctx.globalAlpha = 1;
    ctx.strokeStyle = cssVar('--GUR'); ctx.lineWidth = Math.max(2, cw * 0.6); ctx.strokeRect(xt + ctx.lineWidth / 2, ctx.lineWidth / 2, Math.max(2, cw - ctx.lineWidth), H - ctx.lineWidth);
    ctx.strokeStyle = cssVar('--ink'); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(xt, 0); ctx.lineTo(xt, H); ctx.stroke();
    if (SYN.marker) SYN.marker.attr('x1', SYN.x(SYN.t)).attr('x2', SYN.x(SYN.t));
    $('syn-caption').textContent = `Context: bins 1–${SYN.t + 1} (unmasked). Target: bin ${SYN.t + 2} (orange). ${SYN.K} compounds, peak widths around ${SYN.sig} bins. Every bin is √-transformed and L2-normalised, exactly like the fish data.`;
  }
  function synLoop(ts) {
    if (SYN.playing && ts - SYN.last > 45) { SYN.t = (SYN.t + 1) % (SYN.nb - 1); SYN.last = ts; synFrame(); }
    requestAnimationFrame(synLoop);
  }
  $('syn-k').oninput = e => { SYN.K = +e.target.value; $('syn-k-v').textContent = SYN.K; synGenerate(); synRender(); };
  $('syn-sig').oninput = e => { SYN.sig = +e.target.value; $('syn-sig-v').textContent = SYN.sig; synGenerate(); synRender(); };
  $('syn-new').onclick = () => { SYN.seed = Math.floor(Math.random() * 1e6); synGenerate(); synRender(); };
  $('syn-play').onclick = () => { SYN.playing = !SYN.playing; $('syn-play').setAttribute('aria-pressed', String(SYN.playing)); $('syn-play').textContent = SYN.playing ? '⏸' : '▶'; };
  $('syn-play').textContent = SYN.playing ? '⏸' : '▶';
  $('syn-wrap').addEventListener('pointermove', e => { const r = e.currentTarget.getBoundingClientRect(); SYN.t = Math.max(0, Math.min(SYN.nb - 2, Math.floor((e.clientX - r.left) / r.width * SYN.nb) - 1)); synFrame(); });
  canvasRenderers.push(synRender);
  requestAnimationFrame(synLoop);

  // ── 03 Results strip plot ─────────────────────────────────────────────────
  const RES = { metric: 'ba', active: new Set(REPS) };
  function drawChips() {
    const el = $('rep-chips'); el.replaceChildren();
    REPS.forEach(r => {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'chip'; b.setAttribute('aria-pressed', String(RES.active.has(r)));
      const i = document.createElement('span'); i.className = 'sw'; i.style.background = repColor(r);
      b.append(i, document.createTextNode(r));
      b.onclick = () => { if (RES.active.has(r)) { if (RES.active.size > 1) RES.active.delete(r); } else RES.active.add(r); drawChips(); drawResults(true); };
      el.append(b);
    });
  }
  let resSvg = null;
  function drawResults(animate = false) {
    const metric = RES.metric;
    const rows = M.filter(m => RES.active.has(m.representation)).map(m => ({ m, key: m.method + '|' + m.representation, vals: m[metric], mean: mean(m[metric]) }))
      .sort((a, b) => b.mean - a.mean);
    const phone = isPhone(), rowH = phone ? 50 : 36, left = phone ? 8 : 230, cy = phone ? 34 : rowH / 2;
    const height = rows.length * rowH + 44;
    const el = $('results-chart');
    const width = Math.max(280, el.clientWidth || 600);
    const margin = { top: 6, right: 16, bottom: 34, left };
    const w = width - left - margin.right, h = rows.length * rowH;
    const x = d3.scaleLinear([0, 1], [0, w]);
    const y = d3.scaleBand(rows.map(r => r.key), [0, h]);
    if (!resSvg || resSvg.width !== width) {
      el.replaceChildren();
      const svg = d3.select(el).append('svg');
      const g = svg.append('g').attr('transform', `translate(${left},${margin.top})`);
      g.append('g').attr('class', 'gridline grid-x');
      g.append('g').attr('class', 'axis axis-x');
      g.append('line').attr('class', 'ref-line chance');
      g.append('text').attr('class', 'axis-title xt').attr('text-anchor', 'end');
      g.append('g').attr('class', 'rows');
      resSvg = { svg, g, width };
      animate = false;
    }
    const { svg, g } = resSvg;
    svg.attr('viewBox', `0 0 ${width} ${height}`);
    const t = animate && !reduceMotion ? d3.transition().duration(DUR).ease(d3.easeCubicInOut) : d3.transition().duration(0);
    g.select('.grid-x').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5).tickSize(-h).tickFormat(''));
    g.select('.axis-x').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
    g.select('.chance').attr('x1', x(0.25)).attr('x2', x(0.25)).attr('y1', 0).attr('y2', h);
    g.select('.xt').attr('x', w).attr('y', h + 30).text(metric === 'ba' ? 'balanced accuracy per fold (rule: chance, 0.25)' : 'macro F1 per fold');
    const mname = metric === 'ba' ? 'Balanced acc.' : 'Macro F1';
    const row = g.select('.rows').selectAll('g.row').data(rows, d => d.key)
      .join(enter => {
        const r = enter.append('g').attr('class', 'row').attr('transform', d => `translate(0,${y(d.key)})`).style('opacity', 0);
        if (phone) { const t = r.append('text').attr('class', 'row-label').attr('x', 0).attr('y', 12).text(d => label(d.m)); t.append('tspan').attr('class', 'row-sub').attr('dx', 6).text(d => d.m.representation); }
        else { r.append('text').attr('class', 'row-label').attr('x', -10).attr('y', rowH / 2 - 3).attr('text-anchor', 'end').text(d => label(d.m));
        r.append('text').attr('class', 'row-sub').attr('x', -10).attr('y', rowH / 2 + 10).attr('text-anchor', 'end').text(d => d.m.representation); }
        r.append('rect').attr('class', 'iqr').attr('fill', 'none').attr('stroke-width', 1.5).attr('rx', 2);
        r.append('line').attr('class', 'med').attr('stroke-width', 2);
        r.append('g').attr('class', 'pts');
        r.append('circle').attr('class', 'mean').attr('r', 6).attr('stroke', 'var(--surface)').attr('stroke-width', 2);
        return r;
      }, update => update, exit => exit.transition(t).style('opacity', 0).remove());
    row.transition(t).attr('transform', d => `translate(0,${y(d.key)})`).style('opacity', 1);
    row.select('.row-label').attr('font-weight', d => d.m === best ? 700 : null);
    row.each(function (d) {
      const r = d3.select(this), c = repColor(d.m.representation), vals = d.vals;
      const q1 = d3.quantile(vals, 0.25), q3 = d3.quantile(vals, 0.75), med = d3.median(vals);
      r.select('.iqr').attr('stroke', c).attr('y', cy - 8).attr('height', 16).transition(t).attr('x', x(q1)).attr('width', Math.max(1, x(q3) - x(q1)));
      r.select('.med').attr('stroke', c).attr('y1', cy - 8).attr('y2', cy + 8).transition(t).attr('x1', x(med)).attr('x2', x(med));
      r.select('.mean').attr('fill', c).attr('cy', cy).transition(t).attr('cx', x(d.mean));
      const pts = r.select('.pts').selectAll('circle').data(vals.map((v, i) => ({ v, i })))
        .join(enter => enter.append('circle').attr('r', 4).attr('stroke', 'var(--surface)').attr('stroke-width', 1.5).attr('opacity', 0.6)
          .attr('cy', p => cy + jitter(p.i) * 20).attr('cx', p => x(p.v)));
      pts.attr('fill', c)
        .on('pointermove', (e, p) => showTip(e, `${label(d.m)} · ${d.m.representation}`, [{ label: `seed ${Math.floor(p.i / 5)}, fold ${p.i % 5}`, value: fmt3(p.v), color: cssVar(REP_VAR[d.m.representation]) }, { label: `mean ± SD`, value: `${fmt3(d.mean)} ± ${fmt3(d3.deviation(vals))}` }]))
        .on('pointerleave', hideTip)
        .transition(t).attr('cx', p => x(p.v));
    });
    row.select('.pts').raise(); row.select('.mean').raise();
  }
  drawChips();
  bindSeg('metric-seg', 'metric', m => { RES.metric = m; drawResults(true); drawTable(); });
  renderers.push(() => { resSvg = null; drawResults(false); });

  // ── 03 Table ──────────────────────────────────────────────────────────────
  M.forEach((m, i) => { m.rank = i + 1; m.label = label(m); m.f1mean = mean(m.f1); });
  const TBL = { key: 'mean', dir: -1 };
  const COLS = [['rank', '#'], ['label', 'Method'], ['representation', 'Input'], ['mean', 'Balanced acc.'], ['f1mean', 'Macro F1'], ['abs_r', '|r| vs best'], ['p', 'p (Bonferroni)']];
  function drawTable() {
    const rows = M.slice().sort((a, b) => { const x = a[TBL.key], y = b[TBL.key]; if (x == null) return 1; if (y == null) return -1; return (typeof x === 'string' ? x.localeCompare(y) : x - y) * TBL.dir; });
    const tbl = $('results-table'); tbl.replaceChildren();
    const thead = tbl.createTHead(); const hr = thead.insertRow();
    for (const [k, t] of COLS) { const th = document.createElement('th'); th.textContent = t + (k === TBL.key ? (TBL.dir > 0 ? ' ↑' : ' ↓') : ''); th.onclick = () => { if (TBL.key === k) TBL.dir *= -1; else { TBL.key = k; TBL.dir = ['mean', 'f1mean'].includes(k) ? -1 : 1; } drawTable(); }; hr.append(th); }
    const tb = tbl.createTBody();
    for (const m of rows) {
      const tr = tb.insertRow(); if (m === best) tr.className = 'best';
      const cells = [String(m.rank), m.label, null, null, fmt3(m.f1mean), m.abs_r == null ? '—' : fmt3(m.abs_r), m.p == null ? '—' : m.p.toExponential(1)];
      cells.forEach((c, i) => {
        const td = tr.insertCell();
        if (i === 2) { const tag = document.createElement('span'); tag.className = 'tag'; const dot = document.createElement('i'); dot.style.background = repColor(m.representation); tag.append(dot, document.createTextNode(m.representation)); td.append(tag); }
        else if (i === 3) { const bar = document.createElement('span'); bar.className = 'bar'; bar.style.width = `${(m.mean * 80).toFixed(0)}px`; bar.style.background = repColor(m.representation); td.append(bar, document.createTextNode(`${fmt3(m.mean)} ± ${fmt3(m.sd)}`)); }
        else td.textContent = c;
      });
    }
  }
  drawTable();

  // ── Shared: violin + box + points ─────────────────────────────────────────
  function violin(g, vals, cx, halfW, y, color, opts = {}) {
    const [lo, hi] = d3.extent(vals);
    const bw = opts.bandwidth || 0.045;
    const ticks = d3.range(lo, hi + 1e-9, (hi - lo) / 60 || 0.01);
    const kde = ticks.map(t => [t, d3.mean(vals, v => { const u = (t - v) / bw; return Math.abs(u) <= 1 ? 0.75 * (1 - u * u) / bw : 0; })]);
    const kmax = d3.max(kde, d => d[1]) || 1;
    const area = d3.area().y(d => y(d[0])).x0(d => cx - d[1] / kmax * halfW).x1(d => cx + d[1] / kmax * halfW).curve(d3.curveCatmullRom);
    const grp = g.append('g');
    grp.append('path').attr('d', area(kde)).attr('fill', color).attr('opacity', 0.14);
    grp.append('path').attr('d', area(kde)).attr('fill', 'none').attr('stroke', color).attr('stroke-width', 1.5);
    const q1 = d3.quantile(vals, 0.25), q3 = d3.quantile(vals, 0.75), med = d3.median(vals);
    grp.append('rect').attr('x', cx - 7).attr('width', 14).attr('y', y(q3)).attr('height', Math.max(1, y(q1) - y(q3))).attr('fill', 'var(--surface)').attr('stroke', color).attr('stroke-width', 1.5).attr('rx', 2);
    grp.append('line').attr('x1', cx - 7).attr('x2', cx + 7).attr('y1', y(med)).attr('y2', y(med)).attr('stroke', color).attr('stroke-width', 2);
    const pts = grp.selectAll('circle.pt').data(vals.map((v, i) => ({ v, i }))).join('circle').attr('class', 'pt')
      .attr('cx', d => cx + jitter(d.i) * halfW * 1.1).attr('cy', d => y(d.v)).attr('r', 3.5).attr('fill', color).attr('opacity', 0.55)
      .attr('stroke', 'var(--surface)').attr('stroke-width', 1)
      .on('pointermove', (e, d) => showTip(e, opts.name, [{ label: `seed ${Math.floor(d.i / 5)}, fold ${d.i % 5}`, value: fmt3(d.v), color: opts.hex }]))
      .on('pointerleave', hideTip);
    grp.append('circle').attr('cx', cx).attr('cy', y(mean(vals))).attr('r', 5.5).attr('fill', color).attr('stroke', 'var(--surface)').attr('stroke-width', 2);
    if (opts.label) grp.append('text').attr('class', 'direct-label').attr('x', cx).attr('y', y.range()[0] + 16).attr('text-anchor', 'middle').text(opts.label);
    if (opts.animate && !reduceMotion) { grp.style('opacity', 0).transition().duration(DUR).delay(opts.delay || 0).style('opacity', 1); }
    return grp;
  }

  // ── 04 Pretraining ────────────────────────────────────────────────────────
  function drawPaired(animate = false) {
    const a = scratch.ba, b = best.ba;
    const { g, w, h } = frame('paired-chart', 360, { top: 10, right: 16, bottom: 40, left: 44 });
    const size = Math.min(w, h);
    const x = d3.scaleLinear([0, 1], [0, size]), y = d3.scaleLinear([0, 1], [size, 0]);
    const gg = g.append('g').attr('transform', `translate(${(w - size) / 2},0)`);
    axes(gg, x, y, size, size, { xTitle: 'from scratch', yTitle: 'pretrained', xTicks: 5, yTicks: 5, grid: 'both' });
    gg.append('line').attr('class', 'ref-line').attr('x1', x(0)).attr('y1', y(0)).attr('x2', x(1)).attr('y2', y(1));
    gg.append('text').attr('class', 'row-sub').attr('x', x(0.97)).attr('y', y(0.97) - 8).attr('text-anchor', 'end').text('equal');
    const pts = a.map((v, i) => ({ s: v, p: b[i], i, win: b[i] > v, tie: b[i] === v }));
    const dots = gg.selectAll('circle').data(pts).join('circle').attr('cx', d => x(d.s)).attr('cy', d => y(d.p)).attr('r', 5.5)
      .attr('fill', d => d.win ? 'var(--accent)' : 'var(--GUR)').attr('stroke', 'var(--surface)').attr('stroke-width', 1.5).attr('opacity', 0.85)
      .on('pointermove', (e, d) => showTip(e, `seed ${Math.floor(d.i / 5)}, fold ${d.i % 5}`, [{ label: 'from scratch', value: fmt3(d.s), color: cssVar('--muted') }, { label: 'pretrained', value: fmt3(d.p), color: cssVar('--accent') }, { label: 'gain', value: (d.p - d.s >= 0 ? '+' : '') + fmt3(d.p - d.s) }]))
      .on('pointerleave', hideTip);
    if (animate && !reduceMotion) dots.attr('r', 0).transition().duration(300).delay((d, i) => i * 35).ease(d3.easeBackOut).attr('r', 5.5);
    const wins = pts.filter(p => p.win).length, ties = pts.filter(p => p.tie).length;
    $('paired-caption').textContent = `Pretraining wins on ${wins} of 50 folds${ties ? `, ties on ${ties}` : ''}, losing on ${50 - wins - ties}. Points above the diagonal favour pretraining; the mean gain is +${fmt3(mean(b) - mean(a))}.`;
    legend('paired-legend', [{ label: 'pretrained better', color: 'var(--accent)', kind: 'dot' }, { label: 'from scratch better', color: 'var(--GUR)', kind: 'dot' }]);
  }
  $('paired-replay').onclick = () => drawPaired(true);
  renderers.push(() => drawPaired(false));

  function drawPretrainDist(animate = false) {
    const { g, w, h } = frame('pretrain-dist', 360, { top: 10, right: 12, bottom: 40, left: 44 });
    const y = d3.scaleLinear([0, 1.02], [h, 0]);
    g.append('g').attr('class', 'gridline').call(d3.axisLeft(y).ticks(5).tickSize(-w).tickFormat(''));
    g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(5).tickSizeOuter(0)).call(s => s.select('.domain').remove());
    g.append('text').attr('class', 'axis-title').attr('x', 0).attr('y', -2).text('balanced accuracy');
    g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0.25)).attr('y2', y(0.25)).attr('class', 'ref-line');
    g.append('text').attr('class', 'row-sub').attr('x', w).attr('y', y(0.25) - 5).attr('text-anchor', 'end').text('chance');
    violin(g, scratch.ba, w * 0.28, w * 0.16, y, 'var(--muted)', { label: 'from scratch', name: 'ChromaDCNN, from scratch', hex: cssVar('--muted'), animate, delay: 0 });
    violin(g, best.ba, w * 0.72, w * 0.16, y, 'var(--accent)', { label: 'pretrained', name: 'ChromaDCNN, pretrained', hex: cssVar('--accent'), animate, delay: 150 });
    const kp = $('pretrain-kpis'); kp.replaceChildren();
    [[`${fmt3(mean(scratch.ba))} ± ${fmt3(scratch.sd)}`, 'from scratch, mean ± SD'], [`${fmt3(mean(best.ba))} ± ${fmt3(best.sd)}`, 'pretrained, mean ± SD'], [`${(scratch.p).toExponential(1)}`, 'Mann–Whitney p, Bonferroni']].forEach(([v, k]) => {
      const d = document.createElement('div'); d.className = 'kpi'; const vv = document.createElement('div'); vv.className = 'v'; vv.textContent = v; const kk = document.createElement('div'); kk.className = 'k'; kk.textContent = k; d.append(vv, kk); kp.append(d);
    });
  }
  renderers.push(() => drawPretrainDist(false));

  // ── 05 Ablation ───────────────────────────────────────────────────────────
  const ABL = { collapsed: false, alpha: 0, anim: null, ex: EX[2] };
  const ABL_2D = mean(D.ablation['2D']), ABL_1D = mean(D.ablation['1D-mz']);
  const ablRowMean = (() => { const Z = ABL.ex.Z; const m = d3.range(200).map(j => d3.mean(Z, r => r[j])); const mx = d3.max(m); return m.map(v => v / mx); })();
  function ablDraw(alpha) {
    const Z = ABL.ex.Z;
    blit($('abl-canvas'), matrixCanvas((x, y) => Math.pow((1 - alpha) * Z[y][x] + alpha * ablRowMean[x], 0.6), 200, 100, seqLut()));
    const acc = ABL_2D + (ABL_1D - ABL_2D) * alpha;
    $('abl-acc').textContent = fmt3(acc);
    $('abl-acc-k').textContent = alpha > 0.5 ? 'balanced accuracy, RT removed (sum spectrum tiled)' : 'balanced accuracy, 2D input (from scratch)';
    const delta = (ABL_1D - ABL_2D) * alpha * 100;
    const de = $('abl-delta'); de.textContent = alpha < 0.02 ? '—' : `${delta.toFixed(1)} pp`; de.style.color = alpha < 0.02 ? '' : 'var(--bad)';
  }
  function ablToggle() {
    ABL.collapsed = !ABL.collapsed;
    $('abl-toggle').textContent = ABL.collapsed ? 'Restore RT' : 'Collapse RT';
    $('abl-toggle').classList.toggle('primary', !ABL.collapsed);
    if (ABL.anim) cancelAnimationFrame(ABL.anim);
    const from = ABL.alpha, to = ABL.collapsed ? 1 : 0, t0 = performance.now(), dur = reduceMotion ? 0 : 1100;
    const step = now => {
      const p = dur ? Math.min(1, (now - t0) / dur) : 1; const e = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
      ABL.alpha = from + (to - from) * e; ablDraw(ABL.alpha);
      if (p < 1) ABL.anim = requestAnimationFrame(step);
    };
    ABL.anim = requestAnimationFrame(step);
  }
  $('abl-toggle').onclick = ablToggle;
  canvasRenderers.push(() => ablDraw(ABL.alpha));

  function drawAblation(animate = false) {
    const A = D.ablation;
    const { g, w, h } = frame('ablation-chart', 360, { top: 10, right: 12, bottom: 40, left: 44 });
    const y = d3.scaleLinear([0, 1.02], [h, 0]);
    g.append('g').attr('class', 'gridline').call(d3.axisLeft(y).ticks(5).tickSize(-w).tickFormat(''));
    g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(5).tickSizeOuter(0)).call(s => s.select('.domain').remove());
    g.append('text').attr('class', 'axis-title').attr('x', 0).attr('y', -2).text('balanced accuracy');
    g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0.429)).attr('y2', y(0.429)).attr('class', 'ref-line');
    g.append('text').attr('class', 'row-sub').attr('x', w).attr('y', y(0.429) - 5).attr('text-anchor', 'end').text('PLS-DA, sum spectrum (0.429)');
    violin(g, A['2D'], w * 0.28, w * 0.16, y, 'var(--rep-2d)', { label: '2D chromatogram', name: '2D chromatogram (RT × m/z)', hex: cssVar('--rep-2d'), animate });
    violin(g, A['1D-mz'], w * 0.72, w * 0.16, y, 'var(--rep-sum)', { label: 'RT removed', name: 'sum spectrum tiled over RT', hex: cssVar('--rep-sum'), animate, delay: 150 });
    legend('ablation-legend', [{ label: `2D chromatogram: ${fmt3(ABL_2D)} ± ${fmt3(d3.deviation(A['2D']))}`, color: 'var(--rep-2d)', kind: 'dot' }, { label: `RT removed: ${fmt3(ABL_1D)} ± ${fmt3(d3.deviation(A['1D-mz']))}`, color: 'var(--rep-sum)', kind: 'dot' }]);
  }
  renderers.push(() => drawAblation(false));

  // ── 06 Saliency ───────────────────────────────────────────────────────────
  $('sal-ba').textContent = fmt3(mean(S.fold_ba));
  const SAL = { sp: 2, view: 'sal' };
  function salRender2D() {
    const ex = EX[SAL.sp]; const name = SP_NAME[ex.species];
    const mat = SAL.view === 'chroma' ? ex.Z : ex.S; const gamma = SAL.view === 'chroma' ? 0.55 : 0.45;
    blit($('sal-canvas'), matrixCanvas((x, y) => Math.pow(mat[y][x], gamma), 200, 100, seqLut()));
    $('sal-ramp-lo').textContent = SAL.view === 'chroma' ? '√intensity 0' : 'saliency low';
    $('sal-ramp-hi').textContent = SAL.view === 'chroma' ? 'max' : 'high';
    $('sal-caption').textContent = SAL.view === 'chroma'
      ? `${name}: the first held-out ${name} sample after binning and per-bin normalisation (m/z 50–449 shown). Hover for values.`
      : `${name}: mean gradient saliency over held-out ${name} samples, normalised to its maximum. Hover for values; hover the RT profile below to mark a retention time here.`;
  }
  $('sal-wrap').addEventListener('pointermove', e => {
    if (SAL.view === '3d') return;
    const r = e.currentTarget.getBoundingClientRect(); const ex = EX[SAL.sp];
    const xi = Math.max(0, Math.min(199, Math.floor((e.clientX - r.left) / r.width * 200)));
    const yi = Math.max(0, Math.min(99, 99 - Math.floor((e.clientY - r.top) / r.height * 100)));
    showTip(e, `RT ${RT[yi].toFixed(1)} min · m/z ${MZ[xi]}`, [{ label: 'saliency', value: fmt2(ex.S[yi][xi]), color: cssVar('--' + ex.species) }, { label: '√intensity', value: fmt2(ex.Z[yi][xi]) }]);
  });
  $('sal-wrap').addEventListener('pointerleave', hideTip);
  let plotlyP = null;
  const loadPlotly = () => plotlyP || (plotlyP = new Promise((res, rej) => { const s = document.createElement('script'); s.src = 'https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js'; s.onload = res; s.onerror = rej; document.head.appendChild(s); }));
  function salRender3D() {
    const el = $('sal-3d'); const ex = EX[SAL.sp]; const name = SP_NAME[ex.species];
    el.innerHTML = '<div class="loading">Loading the 3D view…</div>';
    loadPlotly().then(() => {
      if (SAL.view !== '3d') return;
      const z = ex.Z; const nz = z.flat().filter(v => v > 0).sort((a, b) => a - b); const thr = nz[Math.floor(nz.length * 0.8)] || 0;
      const zt = z.map(r => r.map(v => v < thr ? 0 : v));
      const sal = ex.S.map(r => r.map(v => Math.pow(v, 0.6)));
      const ink = cssVar('--ink-2'), line = cssVar('--grid');
      const stops = ['--seq-0', '--seq-1', '--seq-2', '--seq-3', '--seq-4', '--seq-5'].map((v, i) => [i / 5, cssVar(v)]);
      const ax = t => ({ title: { text: t, font: { size: 11, color: ink } }, gridcolor: line, backgroundcolor: 'rgba(0,0,0,0)', showbackground: false, tickfont: { size: 10, color: ink } });
      el.innerHTML = '';
      window.Plotly.newPlot(el, [{
        type: 'surface', x: MZ, y: RT, z: zt, surfacecolor: sal, cmin: 0, cmax: 1, colorscale: stops, showscale: true,
        colorbar: { title: { text: 'saliency', side: 'right' }, tickvals: [0, 1], ticktext: ['low', 'high'], thickness: 10, len: 0.6, outlinewidth: 0, tickfont: { color: ink }, titlefont: { color: ink } },
        contours: { z: { show: false } }, lighting: { ambient: 0.9, diffuse: 0.3, specular: 0.05 },
        hovertemplate: 'RT %{y:.1f} min<br>m/z %{x}<br>√I %{z:.2f}<br>saliency %{surfacecolor:.2f}<extra></extra>',
      }], {
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', margin: { l: 0, r: 0, t: 0, b: 0 }, font: { family: 'Inter, system-ui, sans-serif', color: ink },
        scene: { xaxis: ax('m/z'), yaxis: ax('RT (min)'), zaxis: Object.assign(ax('√intensity'), { showticklabels: false }), camera: { eye: { x: 1.55, y: -1.45, z: 0.95 } }, aspectratio: { x: 1.1, y: 1.4, z: 0.55 } },
      }, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['toImage'] });
      $('sal-caption').textContent = `${name}: height is the first ${name} sample's chromatogram (top 20% of peaks); colour is the species' mean gradient saliency. Drag to rotate, scroll to zoom.`;
    }).catch(() => { el.innerHTML = '<div class="loading">The 3D view needs plotly.js from the CDN, which could not be loaded.</div>'; });
  }
  function salRender() {
    const is3d = SAL.view === '3d';
    $('sal-2d').style.display = is3d ? 'none' : ''; $('sal-3d').style.display = is3d ? '' : 'none';
    if (is3d) salRender3D(); else salRender2D();
  }
  segmented($('sal-species'), SP.map((s, i) => [i, SP_NAME[s], spColor(s)]), k => { SAL.sp = +k; salRender(); }, 2);
  bindSeg('sal-view', 'view', v => { SAL.view = v; salRender(); });
  canvasRenderers.push(salRender);

  function pairRange(C) { const v = []; for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) v.push(C[i][j]); return d3.extent(v); }
  function drawProfiles() {
    // RT profile
    {
      const { g, w, h } = frame('rt-profile', 300, { top: 10, right: 12, bottom: 34, left: 36 });
      const x = d3.scaleLinear([5, 45], [0, w]), y = d3.scaleLinear([0, 1], [h, 0]);
      [[18, 22], [40, 44]].forEach(([a, b]) => g.append('rect').attr('x', x(a)).attr('width', x(b) - x(a)).attr('y', 0).attr('height', h).attr('fill', 'var(--accent)').attr('opacity', 0.08));
      axes(g, x, y, w, h, { xTitle: 'retention time (min)', yTitle: 'normalised saliency', xTicks: 8, yTicks: 4 });
      const line = d3.line().x((d, i) => x(S.rt[i])).y(d => y(d));
      g.selectAll('path.sp').data(SP).join('path').attr('class', 'sp').attr('d', (s, c) => line(S.rt_profile[c])).attr('fill', 'none').attr('stroke', s => spColor(s)).attr('stroke-width', 1.8).attr('stroke-linejoin', 'round');
      const cross = g.append('line').attr('class', 'crosshair').attr('y1', 0).attr('y2', h).style('opacity', 0);
      g.append('rect').attr('width', w).attr('height', h).attr('fill', 'transparent')
        .on('pointermove', e => {
          const [mx] = d3.pointer(e); const i = Math.max(0, Math.min(199, Math.round((x.invert(mx) - 5) / 40 * 199)));
          cross.attr('x1', x(S.rt[i])).attr('x2', x(S.rt[i])).style('opacity', 1);
          showTip(e, `RT ${S.rt[i].toFixed(1)} min`, SP.map((s, c) => ({ label: SP_NAME[s], value: fmt2(S.rt_profile[c][i]), color: cssVar('--' + s) })));
          const ln = $('sal-rtline'); ln.style.top = `${(1 - (S.rt[i] - 5) / 40) * 100}%`; ln.style.opacity = 0.9;
        }).on('pointerleave', () => { cross.style('opacity', 0); hideTip(); $('sal-rtline').style.opacity = 0; });
      const [rl, rh] = pairRange(S.rt_corr);
      $('rt-caption').textContent = `Shaded: 18–22 and 40–44 min. Pairwise correlation between species ${fmt2(rl)}–${fmt2(rh)}: retention time is where the species differ.`;
      legend('rt-legend', SP.map(s => ({ label: SP_NAME[s], color: spColor(s) })));
    }
    // m/z profile
    {
      const { g, w, h } = frame('mz-profile', 300, { top: 10, right: 12, bottom: 34, left: 36 });
      const x = d3.scaleLinear(d3.extent(S.mz), [0, w]), y = d3.scaleLinear([0, 1], [h, 0]);
      axes(g, x, y, w, h, { xTitle: 'm/z', yTitle: 'normalised saliency', xTicks: 8, yTicks: 4 });
      const line = d3.line().x((d, i) => x(S.mz[i])).y(d => y(d));
      g.selectAll('path.sp').data(SP).join('path').attr('class', 'sp').attr('d', (s, c) => line(S.mz_profile[c])).attr('fill', 'none').attr('stroke', s => spColor(s)).attr('stroke-width', 1.4).attr('opacity', 0.85).attr('stroke-linejoin', 'round');
      const cross = g.append('line').attr('class', 'crosshair').attr('y1', 0).attr('y2', h).style('opacity', 0);
      g.append('rect').attr('width', w).attr('height', h).attr('fill', 'transparent')
        .on('pointermove', e => {
          const [mx] = d3.pointer(e); const i = Math.max(0, Math.min(S.mz.length - 1, Math.round(x.invert(mx) - S.mz[0])));
          cross.attr('x1', x(S.mz[i])).attr('x2', x(S.mz[i])).style('opacity', 1);
          showTip(e, `m/z ${S.mz[i]}`, SP.map((s, c) => ({ label: SP_NAME[s], value: fmt2(S.mz_profile[c][i]), color: cssVar('--' + s) })));
        }).on('pointerleave', () => { cross.style('opacity', 0); hideTip(); });
      const [ml, mh] = pairRange(S.mz_corr);
      $('mz-caption').textContent = `Pairwise correlation ${fmt3(ml)}–${fmt3(mh)}: every species relies on the same ions. The difference lies in when they elute.`;
      legend('mz-legend', SP.map(s => ({ label: SP_NAME[s], color: spColor(s) })));
    }
  }
  renderers.push(drawProfiles);

  function drawShares(animate = false) {
    const W = Object.keys(S.window_share).sort((a, b) => parseFloat(a) - parseFloat(b));
    const { g, w, h } = frame('share-chart', 300, { top: 10, right: 12, bottom: 34, left: 40 });
    const x0 = d3.scaleBand(W, [0, w]).paddingInner(0.35).paddingOuter(0.15);
    const x1 = d3.scaleBand(SP, [0, x0.bandwidth()]).paddingInner(0.12);
    const y = d3.scaleLinear([0, d3.max(W, k => d3.max(S.window_share[k]))], [h, 0]).nice();
    axes(g, x0, y, w, h, { xTitle: 'retention-time window (min)', yTitle: 'share of RT saliency', yTicks: 5, yFmt: d3.format('.0%') });
    const bw = Math.min(24, x1.bandwidth());
    const bars = g.selectAll('g.win').data(W).join('g').attr('class', 'win').attr('transform', k => `translate(${x0(k)},0)`)
      .selectAll('path').data(k => SP.map((s, c) => ({ k, s, v: S.window_share[k][c] }))).join('path')
      .attr('fill', d => spColor(d.s))
      .on('pointermove', (e, d) => showTip(e, `${d.k} min`, [{ label: SP_NAME[d.s], value: d3.format('.1%')(d.v), color: cssVar('--' + d.s) }]))
      .on('pointerleave', hideTip);
    const shape = (d, t) => { const xx = x1(d.s) + (x1.bandwidth() - bw) / 2, top = y(d.v * t), r = Math.min(4, (h - top) / 2); return `M${xx},${h}V${top + r}a${r},${r} 0 0 1 ${r},${-r}H${xx + bw - r}a${r},${r} 0 0 1 ${r},${r}V${h}Z`; };
    if (animate && !reduceMotion) bars.attr('d', d => shape(d, 0.001)).transition().duration(DUR).delay((d, i) => i * 40).attr('d', d => shape(d, 1));
    else bars.attr('d', d => shape(d, 1));
    legend('share-legend', SP.map(s => ({ label: SP_NAME[s], color: spColor(s), kind: 'dot' })));
  }
  renderers.push(() => drawShares(false));

  // ── Page chrome: theme, nav, reveal, progress, copy ───────────────────────
  function renderAll() { renderers.forEach(r => r()); }
  function renderCanvases() { canvasRenderers.forEach(r => r()); }
  renderAll(); renderCanvases();
  let resizeT = null;
  window.addEventListener('resize', () => { clearTimeout(resizeT); resizeT = setTimeout(() => { renderAll(); renderCanvases(); }, 180); });

  const root = document.documentElement;
  const isDark = () => (root.dataset.theme ? root.dataset.theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches);
  $('theme-btn').onclick = () => {
    const next = isDark() ? 'light' : 'dark'; root.dataset.theme = next;
    try { localStorage.setItem('chroma-theme', next); } catch (e) {}
    renderCanvases();
  };
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (!root.dataset.theme) renderCanvases(); });

  // First-view animations: fire each chart's intro once, when it scrolls in.
  const intro = { composition: () => drawComposition(true), 'tic-chart': () => drawTic(true), 'paired-chart': () => drawPaired(true), 'pretrain-dist': () => drawPretrainDist(true), 'ablation-chart': () => drawAblation(true), 'share-chart': () => drawShares(true) };
  const introObs = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { intro[e.target.id](); introObs.unobserve(e.target); } }), { threshold: 0.3 });
  if (!reduceMotion) Object.keys(intro).forEach(id => introObs.observe($(id)));

  const revealObs = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); revealObs.unobserve(e.target); } }), { threshold: 0.1 });
  document.querySelectorAll('.reveal').forEach(el => revealObs.observe(el));
  const links = [...document.querySelectorAll('.nav-links a')];
  const secObs = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id)); }), { rootMargin: '-40% 0px -55% 0px' });
  document.querySelectorAll('section.chapter').forEach(s => secObs.observe(s));
  const prog = $('progress');
  const onScroll = () => { const d = document.documentElement; prog.style.width = `${(d.scrollTop / (d.scrollHeight - d.clientHeight) * 100) || 0}%`; };
  window.addEventListener('scroll', onScroll, { passive: true }); onScroll();
  document.querySelectorAll('[data-copy]').forEach(b => b.onclick = () => {
    const text = $(b.dataset.copy).innerText;
    navigator.clipboard?.writeText(text).then(() => { b.textContent = 'Copied'; setTimeout(() => b.textContent = 'Copy', 1500); });
  });
})();
