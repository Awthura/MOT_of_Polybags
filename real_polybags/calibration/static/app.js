/* OVGU AMS calibration tool — client.
   Vanilla JS, no build step: this runs next to a conveyor on whatever machine
   is to hand, so a toolchain would be a liability rather than a convenience. */

const $ = (id) => document.getElementById(id);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  let j;
  try { j = await r.json(); } catch { j = { ok: false, error: `HTTP ${r.status}` }; }
  if (!r.ok && !j.error) j.error = `HTTP ${r.status}`;
  return j;
};
const post = (path, body) => api(path, {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body || {})
});

let toastTimer = null;
function toast(msg, ms = 2600) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), ms);
}

// ── Setup ───────────────────────────────────────────────────────────────────

let CONFIG = null;

async function loadConfig() {
  CONFIG = await api('/api/config');
  const src = $('source');
  src.innerHTML = '';
  CONFIG.sources.forEach(s => {
    const o = document.createElement('option');
    o.value = s.id;
    o.textContent = s.label + (s.available ? '' : ' — unavailable');
    o.disabled = !s.available;
    o.dataset.note = s.note;
    src.appendChild(o);
  });
  const b = $('board');
  b.innerHTML = '';
  CONFIG.boards.forEach(bd => {
    const o = document.createElement('option');
    o.value = bd.id;
    o.textContent = `${bd.name} — ${bd.squares[0]}x${bd.squares[1]}, ${bd.square_mm}mm (${bd.paper})`;
    b.appendChild(o);
  });
  syncSourceUI();
}

function syncSourceUI() {
  const opt = $('source').selectedOptions[0];
  $('source-note').textContent = opt ? opt.dataset.note : '';
  $('folder-field').style.display = $('source').value === 'folder' ? '' : 'none';
}
$('source').addEventListener('change', syncSourceUI);

$('btn-start').addEventListener('click', async () => {
  const r = await post('/api/session', {
    source: $('source').value,
    board: $('board').value,
    camera: $('camera').value,
    folder: $('folder').value,
  });
  if (!r.ok) { toast('Could not start: ' + r.error, 5000); return; }

  REFERENCE = r.truth || r.factory_intrinsics || null;
  $('sec-capture').classList.remove('hidden');
  $('sec-results').classList.add('hidden');
  $('sec-belt').classList.add('hidden');
  $('sec-map').classList.add('hidden');
  $('sec-save').classList.add('hidden');
  // Cache-bust so restarting a session does not reuse the old MJPEG stream.
  $('stream').src = '/api/stream?t=' + Date.now();
  $('hdr-status').textContent = `${r.camera} · ${r.source} · ${r.board}`;
  refreshShots();
  toast('Session started');
  if (r.factory_intrinsics) toast('Factory intrinsics available — will be compared', 4500);
});

// ── Capture ─────────────────────────────────────────────────────────────────

let REFERENCE = null;

function renderCoverage(cov) {
  const g = $('cov-grid');
  g.innerHTML = '';
  cov.cells.flat().forEach(v => {
    const d = document.createElement('div');
    d.className = 'cov-cell' + (v ? ' on' : '');
    g.appendChild(d);
  });
  $('cov-shots').textContent = cov.n_shots;
  $('cov-seen').textContent = `${cov.seen}/${cov.total}`;
  $('cov-advice').textContent = cov.advice;
  $('btn-calibrate').disabled = cov.n_shots < 5;
  $('cap-hint').textContent = cov.n_shots < 5
    ? `${5 - cov.n_shots} more shot(s) before calibration is possible`
    : '';
}

async function refreshShots() {
  const r = await api('/api/shots');
  renderCoverage(r.coverage);
  const box = $('shots');
  box.innerHTML = '';
  r.shots.forEach(s => {
    const d = document.createElement('div');
    d.className = 'shot';
    d.innerHTML = `<img src="data:image/jpeg;base64,${s.thumb}" alt="${s.source}">
                   <button class="danger" title="remove">x</button>`;
    d.querySelector('button').onclick = async () => {
      await api(`/api/shots/${s.i}`, { method: 'DELETE' });
      refreshShots();
    };
    box.appendChild(d);
  });
}

$('btn-capture').addEventListener('click', async () => {
  const r = await post('/api/capture');
  if (!r.ok) { toast(r.error, 3500); return; }
  renderCoverage(r.coverage);
  refreshShots();
});

// Space bar captures — the operator has both hands on the board.
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !$('sec-capture').classList.contains('hidden')
      && e.target.tagName !== 'INPUT') {
    e.preventDefault();
    $('btn-capture').click();
  }
});

// ── Calibrate ───────────────────────────────────────────────────────────────

$('btn-calibrate').addEventListener('click', async () => {
  $('btn-calibrate').disabled = true;
  toast('Calibrating…');
  const r = await post('/api/calibrate');
  $('btn-calibrate').disabled = false;
  if (!r.ok) { toast(r.error, 5000); return; }

  $('sec-results').classList.remove('hidden');
  $('sec-belt').classList.remove('hidden');
  $('sec-map').classList.remove('hidden');
  $('sec-save').classList.remove('hidden');
  $('report').textContent = r.report;

  const w = $('warnings');
  w.innerHTML = '';
  (r.warnings || []).forEach(msg => {
    const li = document.createElement('li');
    li.textContent = msg;
    w.appendChild(li);
  });
  if (!r.warnings || !r.warnings.length) {
    w.innerHTML = '<li style="color:var(--ok)">No issues detected.</li>';
  }

  if (r.truth && r.truth_delta) {
    $('truth-box').classList.remove('hidden');
    const d = r.truth_delta;
    const cls = (v, good) => v <= good ? 'ok' : (v <= good * 3 ? 'warn' : 'bad');
    $('truth-stats').innerHTML = `
      <div class="stat"><span class="k">fx</span><span class="v">${r.fx.toFixed(1)} vs ${r.truth.fx.toFixed(1)}
        <span class="pill ${cls(d.fx_pct, 2)}">${d.fx_pct.toFixed(2)}%</span></span></div>
      <div class="stat"><span class="k">fy</span><span class="v">${r.fy.toFixed(1)} vs ${r.truth.fy.toFixed(1)}
        <span class="pill ${cls(d.fy_pct, 2)}">${d.fy_pct.toFixed(2)}%</span></span></div>
      <div class="stat"><span class="k">cx</span><span class="v">${r.cx.toFixed(1)} vs ${r.truth.cx.toFixed(1)}
        <span class="pill ${cls(d.cx_px, 15)}">${d.cx_px.toFixed(1)} px</span></span></div>
      <div class="stat"><span class="k">cy</span><span class="v">${r.cy.toFixed(1)} vs ${r.truth.cy.toFixed(1)}
        <span class="pill ${cls(d.cy_px, 15)}">${d.cy_px.toFixed(1)} px</span></span></div>`;
  }
  $('sec-results').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  toast('Intrinsics calibrated');
});

// ── Extrinsics + click-to-validate ──────────────────────────────────────────

let HAVE_EXTRINSICS = false;

$('btn-extr').addEventListener('click', async () => {
  const r = await post('/api/extrinsics', {
    origin_x_mm: parseFloat($('ox').value) || 0,
    origin_y_mm: parseFloat($('oy').value) || 0,
  });
  if (!r.ok) { toast(r.error, 5000); return; }
  HAVE_EXTRINSICS = true;
  const p = r.camera_position_mm;
  const errCls = r.reproj_error_px < 1 ? 'ok' : (r.reproj_error_px < 2 ? 'warn' : 'bad');
  $('extr-out').innerHTML = `
    <div class="stat"><span class="k">camera position (belt frame)</span>
      <span class="v">[${p.map(v => v.toFixed(0)).join(', ')}] mm</span></div>
    <div class="stat"><span class="k">height above belt</span>
      <span class="v">${r.height_above_belt_mm.toFixed(0)} mm</span></div>
    <div class="stat"><span class="k">reprojection error</span>
      <span class="v">${r.reproj_error_px.toFixed(3)} px
      <span class="pill ${errCls}">${errCls === 'ok' ? 'good' : errCls}</span></span></div>
    <div class="stat"><span class="k">points used</span><span class="v">${r.n_points}</span></div>
    ${(r.warnings || []).map(w => `<div class="note" style="color:var(--warn)">${w}</div>`).join('')}`;
  $('click-hint').classList.remove('hidden');
  $('click-hint').textContent =
    'Now click anywhere on the preview to read that point in belt millimetres — '
    + 'the quickest way to sanity-check the mapping against a tape measure.';
  toast('Extrinsics solved — click the preview to validate');
});

$('stream').addEventListener('click', async (e) => {
  if (!HAVE_EXTRINSICS) return;
  const img = e.target;
  const rect = img.getBoundingClientRect();
  // The preview is scaled to fit; convert back to native pixel coordinates,
  // otherwise every clicked point would be wrong by the display scale factor.
  const sx = img.naturalWidth / rect.width;
  const sy = img.naturalHeight / rect.height;
  const x = (e.clientX - rect.left) * sx;
  const y = (e.clientY - rect.top) * sy;

  const m = $('marker');
  m.style.display = 'block';
  m.style.left = (e.clientX - rect.left) + 'px';
  m.style.top = (e.clientY - rect.top) + 'px';

  const r = await post('/api/to_belt', { x, y });
  if (!r.ok) { toast(r.error, 3500); return; }
  toast(`belt: X ${r.x_mm.toFixed(1)} mm, Y ${r.y_mm.toFixed(1)} mm  ·  `
        + `${r.mm_per_px.toFixed(3)} mm/px here`, 5000);
});

// ── Save ────────────────────────────────────────────────────────────────────

$('btn-map').addEventListener('click', async () => {
  toast('Building belt map…');
  const r = await post('/api/beltmap', {
    belt_width_mm: parseFloat($('bw').value) || 700,
    belt_length_mm: parseFloat($('bl').value) || 1400,
    mm_per_px: parseFloat($('bres').value) || 2,
  });
  if (!r.ok) { toast(r.error, 6000); return; }

  $('map-img').src = 'data:image/png;base64,' + r.png;
  $('map-img').style.display = 'block';
  $('map-empty').style.display = 'none';

  const cov = r.coverage;
  const mm2cm2 = (v) => (v / 100).toFixed(0);
  let html = `<div class="stat"><span class="k">cameras on map</span>
      <span class="v">${r.cameras.join(', ')}</span></div>`;
  html += `<div class="stat"><span class="k">belt covered</span>
      <span class="v">${mm2cm2(cov.covered_mm2)} / ${mm2cm2(cov.map_area_mm2)} cm²</span></div>`;
  for (const [n, a] of Object.entries(cov.per_camera_mm2)) {
    html += `<div class="stat"><span class="k">${n}</span><span class="v">${mm2cm2(a)} cm²</span></div>`;
  }
  html += `<div style="margin-top:10px"><strong style="font-size:12px;text-transform:uppercase;
      letter-spacing:.6px;color:var(--ink-dim)">Overlap (measured)</strong></div>`;
  const pairs = Object.entries(cov.pairwise);
  if (!pairs.length) {
    html += `<p class="note">Only one camera calibrated — nothing to compare.</p>`;
  } else {
    pairs.forEach(([pair, v]) => {
      const pct = (v.fraction_of_smaller * 100).toFixed(1);
      const cls = v.overlap_mm2 > 0 ? 'ok' : 'warn';
      html += `<div class="stat"><span class="k">${pair.replace('|', ' / ')}</span>
        <span class="v">${mm2cm2(v.overlap_mm2)} cm²
        <span class="pill ${cls}">${pct}%</span></span></div>`;
    });
    html += `<p class="note">${cov.any_overlap
      ? 'Cameras share belt area — they can be associated geometrically in the overlap.'
      : 'No overlap found. Cameras are still in one frame via the belt, but a bag is never seen by two at once, so hand-off depends on belt travel rather than shared view.'}</p>`;
  }
  if (r.skipped && r.skipped.length) {
    html += `<p class="note" style="color:var(--warn)">Skipped: ` +
      r.skipped.map(s => `${s.camera} (${s.why})`).join('; ') + `</p>`;
  }
  $('map-stats').innerHTML = html;
  toast('Belt map built');
});

$('btn-save').addEventListener('click', async () => {
  const r = await post('/api/save', { notes: $('notes').value });
  if (!r.ok) { toast(r.error, 4000); return; }
  toast('Saved to ' + r.path, 4000);
  const s = await api('/api/results');
  $('results-summary').textContent = s.summary;
});

loadConfig();
setInterval(() => {
  if (!$('sec-capture').classList.contains('hidden')) refreshShots();
}, 4000);
