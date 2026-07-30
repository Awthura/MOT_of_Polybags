/* OVGU AMS calibration tool — intrinsics page.
   Shared helpers live in common.js. Extrinsics moved to extrinsics.js: they are
   rig work rather than bench work, and the two are separate sittings. */

// ── Setup ───────────────────────────────────────────────────────────────────

let CONFIG = null;

async function loadConfig() {
  CONFIG = await api('/api/config');
  fillSourceAndBoardSelects(CONFIG, $('source'), $('board'));
  syncSourceUI();
  // Prefill the belt dimensions from the rig plan, so the figure entered once
  // on the rig page is the one the map is built with here.
  const p = await api('/api/plan');
  if (p.ok && p.plan && p.plan.belt) {
    $('bw').value = p.plan.belt.width_mm;
    $('bl').value = p.plan.belt.length_mm;
  }
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
  $('btn-capture-all').classList.toggle('hidden', r.source !== 'folder');
  refreshShots();
  // If this camera was calibrated before, its intrinsics come back with the
  // session, and the rig page can solve its pose without recapturing anything —
  // which is what makes "all intrinsics first, then all extrinsics against one
  // board placement" possible.
  if (r.loaded_intrinsics && !r.loaded_intrinsics.error) {
    const L = r.loaded_intrinsics;
    $('sec-belt').classList.remove('hidden');
    $('sec-map').classList.remove('hidden');
    $('sec-save').classList.remove('hidden');
    $('report').textContent =
      `Reloaded saved intrinsics for ${r.camera}\n` +
      `  measured ${L.created || 'previously'} at ${L.image_size[0]}x${L.image_size[1]}\n` +
      `  fx ${L.fx.toFixed(1)}  fy ${L.fy.toFixed(1)}  ` +
      `cx ${L.cx.toFixed(1)}  cy ${L.cy.toFixed(1)}  RMS ${L.rms.toFixed(3)} px\n\n` +
      `Capture new shots above to re-measure, or go to the rig page and solve\n` +
      `this camera's belt pose.`;
    $('sec-results').classList.remove('hidden');
    toast(`Loaded saved intrinsics for ${r.camera} — ready for extrinsics`, 5000);
  } else {
    toast('Session started');
  }
  if (r.loaded_intrinsics && r.loaded_intrinsics.error) {
    toast(r.loaded_intrinsics.error, 6000);
  }
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

// Folder source only. The preview cycles a folder at 15 fps, so clicking
// Capture samples it at random and takes some views twice — and a duplicated
// view is weighted twice in the fit while looking like a bigger sample.
$('btn-capture-all').addEventListener('click', async () => {
  $('btn-capture-all').disabled = true;
  toast('Reading every frame in the folder…');
  const r = await post('/api/capture_all');
  $('btn-capture-all').disabled = false;
  if (!r.ok) { toast(r.error, 5000); return; }
  renderCoverage(r.coverage);
  refreshShots();
  const bad = (r.rejected || []).length;
  toast(`Added ${r.added} of ${r.n_files} frames`
        + (bad ? ` — ${bad} had no detectable board` : ''), bad ? 6000 : 3000);
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

// ── Belt map and save ───────────────────────────────────────────────────────

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
