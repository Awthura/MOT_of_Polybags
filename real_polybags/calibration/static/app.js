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
  const sourceId = $('source').value;
  $('source-note').textContent = opt ? opt.dataset.note : '';
  $('folder-field').style.display = sourceId === 'folder' ? '' : 'none';

  // A device field is shown wherever more than one physical unit of that kind
  // could plausibly need distinguishing:
  //  - live sources with >1 unit enumerated (two Baslers; potentially two
  //    RealSense) -> a picker, via the datalist, and a value is required;
  //  - the folder source -> always shown but never required, since a replay
  //    has no live hardware to ask; typing a serial here just asserts one for
  //    traceability (read off the recorder's console output at record time).
  // Hidden for synthetic and any source with exactly one device: nothing
  // there needs disambiguating.
  const devices = (CONFIG.sources.find(s => s.id === sourceId) || {}).devices || [];
  const devField = $('device-field');
  const devInput = $('device');
  const devList = $('device-list');
  devList.innerHTML = '';
  devices.forEach(d => {
    const o = document.createElement('option');
    o.value = d.serial;
    o.textContent = `${d.serial}${d.model ? ' — ' + d.model : ''}`;
    devList.appendChild(o);
  });

  if (devices.length > 1) {
    devField.style.display = '';
    $('device-note').textContent =
      `${devices.length} connected — pick the one physically mounted as "${$('camera').value}".`;
  } else if (sourceId === 'folder') {
    devField.style.display = '';
    devInput.value = '';
    $('device-note').textContent =
      'No live camera to confirm identity from. Optionally type the serial the '
      + 'recorder printed for this footage — recorded as asserted, not '
      + 'hardware-confirmed.';
  } else {
    devField.style.display = 'none';
    devInput.value = '';
    $('device-note').textContent = devices.length === 1
      ? `Only one device connected (S/N ${devices[0].serial}) — no picker needed.`
      : '';
  }
}
$('source').addEventListener('change', syncSourceUI);

$('btn-start').addEventListener('click', async () => {
  const r = await post('/api/session', {
    source: $('source').value,
    board: $('board').value,
    camera: $('camera').value,
    folder: $('folder').value,
    device_serial: $('device').value || '',
  });
  if (!r.ok) { toast('Could not start: ' + r.error, 5000); return; }

  REFERENCE = r.truth || r.factory_intrinsics || null;
  $('sec-capture').classList.remove('hidden');
  $('sec-results').classList.add('hidden');
  $('sec-belt').classList.add('hidden');
  $('sec-map').classList.add('hidden');
  $('sec-save').classList.add('hidden');
  $('factory-box').classList.add('hidden');
  // Cache-bust so restarting a session does not reuse the old MJPEG stream.
  $('stream').src = '/api/stream?t=' + Date.now();
  $('hdr-status').textContent = `${r.camera} · ${r.source} · ${r.board}`;
  $('btn-capture-all').classList.toggle('hidden', r.source !== 'folder');
  refreshShots();

  // Which physical camera did this session actually connect to? The camera
  // *name* is just a label typed above; this is what the hardware itself (or
  // the operator, for a folder replay) reports.
  const db = $('device-banner');
  if (r.device) {
    db.classList.remove('hidden');
    db.className = 'banner ' + (r.device.asserted ? 'warn' : 'ok');
    db.textContent = r.device.asserted
      ? `Device serial ${r.device.serial} — operator-asserted, not hardware-confirmed `
        + `(this source has no live camera to ask). Cross-check against what the `
        + `recorder printed to console when this footage was captured.`
      : `Connected: S/N ${r.device.serial}${r.device.model ? ' (' + r.device.model + ')' : ''}. `
        + `Recorded into the saved calibration for traceability.`;
  } else {
    db.classList.add('hidden');
  }

  // A RealSense reports its own factory calibration — offer to adopt it
  // directly rather than always requiring a board capture.
  const fb = $('factory-banner');
  if (r.factory_intrinsics) {
    fb.classList.remove('hidden');
    fb.className = 'banner ok';
    fb.innerHTML = `This camera reports factory intrinsics
      (fx ${r.factory_intrinsics.fx.toFixed(1)}, fy ${r.factory_intrinsics.fy.toFixed(1)},
      model ${esc(r.factory_intrinsics.model || 'unknown')}). Use them directly, or
      capture a board below to fit your own and compare —
      the acceptance test for this method.
      <button id="btn-use-factory" style="margin-left:10px">Use factory intrinsics</button>`;
    $('btn-use-factory').addEventListener('click', useFactoryIntrinsics);
  } else {
    fb.classList.add('hidden');
  }

  // If this camera was calibrated before, its intrinsics come back with the
  // session, and the rig page can solve its pose without recapturing anything —
  // which is what makes "all intrinsics first, then all extrinsics against one
  // board placement" possible.
  if (r.loaded_intrinsics && !r.loaded_intrinsics.error) {
    const L = r.loaded_intrinsics;
    $('sec-belt').classList.remove('hidden');
    $('sec-map').classList.remove('hidden');
    $('sec-save').classList.remove('hidden');
    const rmsLine = L.rms == null
      ? '  (factory-provided — no fitted RMS to show)'
      : `  fx ${L.fx.toFixed(1)}  fy ${L.fy.toFixed(1)}  ` +
        `cx ${L.cx.toFixed(1)}  cy ${L.cy.toFixed(1)}  RMS ${L.rms.toFixed(3)} px`;
    $('report').textContent =
      `Reloaded saved intrinsics for ${r.camera} (${L.method || 'board'})\n` +
      `  measured ${L.created || 'previously'} at ${L.image_size[0]}x${L.image_size[1]}\n` +
      rmsLine + `\n\n` +
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
});

// ── Clear saved calibration ─────────────────────────────────────────────────
// Removes intrinsics AND any pose for the named camera — results/<camera>.json
// holds both halves together, so there is no smaller unit to clear. Most
// useful for wiping a synthetic rehearsal result before real work, or
// discarding a bad calibration to force a clean redo.

$('btn-clear-camera').addEventListener('click', async () => {
  const camera = $('camera').value.trim();
  if (!camera) { toast('Enter a camera name first', 3000); return; }
  if (!confirm(`Delete the saved calibration for "${camera}"?\n\n`
              + `This removes both intrinsics and any solved pose — there is `
              + `no smaller unit to clear. Cannot be undone.`)) return;
  const r = await api(`/api/results/${encodeURIComponent(camera)}`, { method: 'DELETE' });
  if (!r.ok) { toast(r.error || 'Could not clear', 5000); return; }
  toast(r.deleted ? `Cleared saved calibration for ${camera}` : `${camera} had nothing saved`, 4000);
  // If the camera just cleared is the one in the active session, the results
  // shown on screen are now stale — they describe a file that no longer
  // exists.
  if ($('hdr-status').textContent.startsWith(camera + ' ')) {
    $('sec-results').classList.add('hidden');
    $('sec-belt').classList.add('hidden');
    $('sec-map').classList.add('hidden');
    $('sec-save').classList.add('hidden');
  }
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

/* Shared by a board fit (/api/calibrate) and an adopted factory calibration
   (/api/factory_intrinsics) — same results panel, different provenance. */
function showIntrinsicsResult(r, { factory = false } = {}) {
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

  $('truth-box').classList.add('hidden');
  $('factory-box').classList.add('hidden');

  if (factory) {
    // There is no separate "truth" to compare against here — the factory
    // numbers ARE what got adopted, not a fit being checked against a
    // reference. Say so plainly rather than showing an empty comparison.
    $('factory-box').classList.remove('hidden');
    $('factory-box').innerHTML =
      `<strong>Adopted directly from the sensor's factory calibration`
      + `${r.model ? ' (' + esc(r.model) + ')' : ''}.</strong> Not fitted here — `
      + `there is no reprojection error or frame coverage to show, and no board `
      + `capture was needed for this camera.`;
  } else if (r.truth && r.truth_delta) {
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
}

$('btn-calibrate').addEventListener('click', async () => {
  $('btn-calibrate').disabled = true;
  toast('Calibrating…');
  const r = await post('/api/calibrate');
  $('btn-calibrate').disabled = false;
  if (!r.ok) { toast(r.error, 5000); return; }
  showIntrinsicsResult(r, { factory: false });
  toast('Intrinsics calibrated');
});

/* The RealSense reports its own factory intrinsics — this adopts them
   directly instead of requiring a board capture. Still worth doing the board
   fit once and comparing (see the "vs known reference" panel) to establish
   that this tool's method agrees with an independent source; this is the
   fast path for every session after that. */
async function useFactoryIntrinsics() {
  const btn = $('btn-use-factory');
  if (btn) btn.disabled = true;
  toast('Adopting factory intrinsics…');
  const r = await post('/api/factory_intrinsics', {});
  if (btn) btn.disabled = false;
  if (!r.ok) { toast(r.error, 6000); return; }
  showIntrinsicsResult(r, { factory: true });
  toast('Factory intrinsics adopted — ready to save or solve extrinsics');
}

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

  // When a georeferenced workspace map exists it defines the frame, and the
  // belt width/length above are not used — say so rather than leaving two
  // stale-looking numbers implying otherwise.
  const wsNote = $('map-ws-note');
  if (r.workspace) {
    wsNote.className = 'banner ok';
    wsNote.textContent =
      `Rendered against the uploaded workspace map (${r.workspace.source.toUpperCase()}, `
      + `${r.workspace.mm_per_px.toFixed(2)} mm/px). The belt width and length above `
      + `were not used — remove the map on the rig page to go back to them.`;
  } else {
    wsNote.className = 'banner';
    wsNote.textContent =
      'Rendered from the belt width and length above. Upload a workspace map on '
      + 'the rig page to render against a real floor plan or scan instead.';
  }
  wsNote.classList.remove('hidden');

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
