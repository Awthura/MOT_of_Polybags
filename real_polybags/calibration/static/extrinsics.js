/* OVGU AMS calibration tool — extrinsics (rig session) page.

   The rig phase, on its own page because it is its own sitting: intrinsics are
   bench work that needs no conveyor, extrinsics need the rig in its final state
   and are void the moment a camera is re-aimed.

   Two things this page does that the intrinsics page cannot:
   - a status board across ALL cameras, rebuilt from results/*.json, so "what is
     still outstanding" is answered before walking up to the belt;
   - anticipated measurements, recorded up front, so each solve is checked
     against an expectation rather than merely displayed. */

let CONFIG = null;
let PLAN = null;
let ROSTER = [];
let HAVE_EXTRINSICS = false;
let SESSION_CAMERA = null;

const num = (v) => { const f = parseFloat(v); return Number.isFinite(f) ? f : null; };
const fmt = (v, unit, d = 0) =>
  (v === null || v === undefined || !Number.isFinite(v)) ? '—' : `${v.toFixed(d)} ${unit}`;

const STATE_PILL = {
  solved: 'ok', ready: 'info', provisional: 'warn', blocked: 'idle',
};
const STATE_WORD = {
  solved: 'solved', ready: 'ready', provisional: 'provisional', blocked: 'blocked',
};

// ── Plan + status board ─────────────────────────────────────────────────────

async function loadPlan() {
  const r = await api('/api/plan');
  if (!r.ok) { toast('Could not load the plan: ' + r.error, 5000); return; }
  PLAN = r.plan;
  ROSTER = r.roster;
  if (PLAN.load_error) toast(PLAN.load_error, 6000);
  renderPlan(r.summary);
}

function renderPlan(summary) {
  $('plan-headline').textContent = summary.headline;

  $('belt-w').value = PLAN.belt.width_mm;
  $('belt-l').value = PLAN.belt.length_mm;
  document.querySelectorAll('input[name=method]').forEach(el => {
    el.checked = (el.value === PLAN.method);
  });
  syncMethodNote();

  const body = $('roster-body');
  body.innerHTML = '';
  ROSTER.forEach((row, i) => {
    const p = row.plan;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="cam">${esc(row.name)}${row.in_plan ? ''
        : ' <span class="pill idle" title="has a saved calibration but is not in the plan">unlisted</span>'}${
        row.synthetic ? ' <span class="pill bad" title="measured from the synthetic camera — a rehearsal result">synthetic</span>' : ''}${
        row.device_serial ? `<div class="note" style="margin-top:2px">S/N ${esc(row.device_serial)}${
          row.device_asserted ? ' <span title="operator-asserted, not hardware-confirmed">(asserted)</span>' : ''}</div>` : ''}</td>
      <td><span class="pill ${STATE_PILL[row.state]}">${STATE_WORD[row.state]}</span></td>
      <td class="mono">${row.has_intrinsics
        ? `${row.image_size ? row.image_size.join('×') : '?'}` + (
            row.intrinsics_method === 'factory'
              ? ` · <span class="pill info" title="adopted from the sensor's factory calibration, not fitted here">factory</span>`
              : ` · rms ${Number.isFinite(row.intrinsics_rms_px) ? row.intrinsics_rms_px.toFixed(2) : '?'} px`)
        : '—'}</td>
      <td class="mono">${Number.isFinite(row.height_above_belt_mm)
        ? `${row.height_above_belt_mm.toFixed(0)} mm · ${
            Number.isFinite(row.extr_error_px) ? row.extr_error_px.toFixed(2) : '?'} px`
        : (row.state === 'provisional' ? 'homography only' : '—')}</td>
      <td class="anticipated"><input type="number" step="10" size="7" class="cell"
            data-i="${i}" data-k="expected_height_mm"
            value="${p.expected_height_mm === null ? '' : p.expected_height_mm}"
            placeholder="—"></td>
      <td class="anticipated"><div class="offset-pair">
        <input type="number" step="1" class="cell" title="X — across the belt"
            data-i="${i}" data-k="ox" value="${p.origin_offset_mm[0]}">
        <input type="number" step="1" class="cell" title="Y — along travel"
            data-i="${i}" data-k="oy" value="${p.origin_offset_mm[1]}">
      </div></td>
      <td class="anticipated" style="text-align:center"><input type="checkbox" class="cell"
            data-i="${i}" data-k="offset_measured"
            ${p.offset_measured ? 'checked' : ''}></td>
      <td><button class="ghost small" data-solve="${esc(row.name)}">Solve</button>
        <button class="danger small" data-clear="${esc(row.name)}"
                title="Delete the saved calibration for this camera — intrinsics and any pose"
                ${row.has_intrinsics || row.state !== 'blocked' || row.error ? '' : 'disabled'}>Clear</button></td>`;
    body.appendChild(tr);

    if (row.pending.length || row.error) {
      const pr = document.createElement('tr');
      pr.className = 'pending';
      const items = row.error
        ? [`unreadable calibration file: ${row.error}`]
        : row.pending;
      pr.innerHTML = `<td colspan="8"><ul>${
        items.map(t => `<li>${esc(t)}</li>`).join('')}</ul></td>`;
      body.appendChild(pr);
    }
  });

  body.querySelectorAll('input.cell').forEach(el => {
    el.addEventListener('change', () => {
      const p = planEntryFor(el.dataset.i);
      const k = el.dataset.k;
      if (k === 'offset_measured') p.offset_measured = el.checked;
      else if (k === 'ox') p.origin_offset_mm[0] = num(el.value) ?? 0;
      else if (k === 'oy') p.origin_offset_mm[1] = num(el.value) ?? 0;
      else p.expected_height_mm = num(el.value);
      markDirty();
    });
  });
  body.querySelectorAll('button[data-solve]').forEach(el => {
    el.addEventListener('click', () => {
      $('camera').value = el.dataset.solve;
      onCameraPicked();
      $('camera').scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  });
  body.querySelectorAll('button[data-clear]').forEach(el => {
    el.addEventListener('click', async () => {
      const name = el.dataset.clear;
      if (!confirm(`Delete the saved calibration for "${name}"?\n\n`
                  + `This removes both intrinsics and any solved pose — there `
                  + `is no smaller unit to clear. Cannot be undone.`)) return;
      const r = await api(`/api/results/${encodeURIComponent(name)}`, { method: 'DELETE' });
      if (!r.ok) { toast(r.error || 'Could not clear', 5000); return; }
      toast(r.deleted ? `Cleared ${name}` : `${name} had nothing saved`, 3500);
      PLAN = r.plan; ROSTER = r.roster;
      renderPlan(r.summary);
    });
  });

  const sel = $('camera');
  const keep = sel.value;
  sel.innerHTML = '';
  ROSTER.forEach(row => {
    const o = document.createElement('option');
    o.value = row.name;
    o.textContent = `${row.name} — ${STATE_WORD[row.state]}`;
    sel.appendChild(o);
  });
  if (keep && ROSTER.some(r => r.name === keep)) sel.value = keep;
  onCameraPicked();
  // Keep the correspondence-route picker on the same roster, so a camera
  // added or cleared above shows up there without a reload.
  if (typeof corrCameras === 'function') corrCameras();
}

/* A roster row is not necessarily a plan entry — a camera with a saved
   calibration but no plan line is listed too, so the two are indexed by name
   rather than assumed parallel.

   Typing an anticipated value into an unlisted camera's row adopts it into the
   plan. The alternative is an input that accepts a number and quietly discards
   it, which is worse than not offering the input at all. */
function planEntryFor(rosterIdx) {
  const name = ROSTER[Number(rosterIdx)].name;
  let p = PLAN.cameras.find(c => c.name === name);
  if (!p) {
    p = { name, expected_height_mm: null, origin_offset_mm: [0, 0],
          offset_measured: false, note: '' };
    PLAN.cameras.push(p);
  }
  return p;
}

let DIRTY = false;
function markDirty() {
  DIRTY = true;
  $('btn-save-plan').textContent = 'Save plan •';
}

function collectPlan() {
  PLAN.belt.width_mm = num($('belt-w').value) ?? PLAN.belt.width_mm;
  PLAN.belt.length_mm = num($('belt-l').value) ?? PLAN.belt.length_mm;
  const m = document.querySelector('input[name=method]:checked');
  PLAN.method = m ? m.value : 'A';
  return PLAN;
}

async function savePlan(quiet) {
  const r = await post('/api/plan', collectPlan());
  if (!r.ok) { toast('Could not save the plan: ' + r.error, 5000); return; }
  PLAN = r.plan; ROSTER = r.roster;
  DIRTY = false;
  $('btn-save-plan').textContent = 'Save plan';
  renderPlan(r.summary);
  if (!quiet) toast('Plan saved');
}

$('btn-save-plan').addEventListener('click', () => savePlan(false));
$('btn-refresh').addEventListener('click', async () => {
  if (DIRTY) { await savePlan(true); toast('Plan saved and refreshed'); return; }
  await loadPlan();
  toast('Refreshed');
});

$('btn-add-cam').addEventListener('click', async () => {
  const name = $('new-cam').value.trim();
  if (!name) { toast('Enter a camera name first', 3000); return; }
  if (PLAN.cameras.some(c => c.name === name)) {
    toast(`${name} is already in the plan`, 3000); return;
  }
  PLAN.cameras.push({
    name, expected_height_mm: null, origin_offset_mm: [0, 0],
    offset_measured: false, note: '',
  });
  $('new-cam').value = '';
  await savePlan(true);
  toast(`Added ${name}`);
});

document.querySelectorAll('input[name=method]').forEach(el =>
  el.addEventListener('change', () => { syncMethodNote(); markDirty(); }));
$('belt-w').addEventListener('change', markDirty);
$('belt-l').addEventListener('change', markDirty);

function syncMethodNote() {
  const m = document.querySelector('input[name=method]:checked');
  $('method-note').innerHTML = (m && m.value === 'B')
    ? `<strong>Method B.</strong> Solve the first camera at offset 0, 0 — that
       <em>defines</em> the belt origin. For every other camera, move the board
       into its view, measure the displacement from that first placement
       (X across the belt, Y along travel), and enter it. Accuracy here is your
       tape measure's accuracy, and it propagates directly into cross-camera
       agreement.`
    : `<strong>Method A.</strong> Stop the conveyor, lay the board flat on the
       belt inside the shared view, and <em>do not move it</em> until every
       camera has been solved — leaving all offsets at 0. Every camera is then
       solved against one physical placement, so they share an origin exactly,
       with no measurement and therefore no measurement error. Use this wherever
       the cameras can see the same patch of belt.`;
}

function onCameraPicked() {
  const name = $('camera').value;
  const row = ROSTER.find(r => r.name === name);
  if (!row) return;
  const p = row.plan;
  $('ox').value = p.origin_offset_mm[0];
  $('oy').value = p.origin_offset_mm[1];
  $('offset-hint').textContent = p.expected_height_mm === null
    ? 'No expected height recorded for this camera — the solve will not be checkable. Add one in the table above.'
    : `Expecting roughly ${p.expected_height_mm.toFixed(0)} mm above the belt.`;
  const b = $('intr-state');
  if (row.has_intrinsics) {
    b.className = 'banner ok';
    b.textContent = `${name} has saved intrinsics (${
      row.image_size ? row.image_size.join('×') : '?'}) — they will reload when the session starts.`;
  } else {
    b.className = 'banner bad';
    b.textContent = `${name} has no saved intrinsics, so its pose cannot be solved. `
      + 'Calibrate its lens on the Intrinsics page first — that needs no rig access.';
  }
  b.classList.remove('hidden');
}
$('camera').addEventListener('change', onCameraPicked);

// ── Session ─────────────────────────────────────────────────────────────────

async function loadConfig() {
  CONFIG = await api('/api/config');
  fillSourceAndBoardSelects(CONFIG, $('source'), $('board'));
  syncSourceUI();
}

function syncSourceUI() {
  const opt = $('source').selectedOptions[0];
  const sourceId = $('source').value;
  $('source-note').textContent = opt ? opt.dataset.note : '';
  $('folder-field').style.display = sourceId === 'folder' ? '' : 'none';

  // Same logic as the intrinsics page (duplicated rather than shared — the
  // two pages load independently and this is a handful of lines): a picker
  // for live sources with >1 unit, a free-typed assertion for the folder
  // source (no live hardware to ask), hidden otherwise.
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
      `${devices.length} connected — pick the one physically mounted as `
      + `"${$('camera').value}".`;
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
  const camera = $('camera').value;
  const r = await post('/api/session', {
    source: $('source').value, board: $('board').value,
    camera, folder: $('folder').value,
    device_serial: $('device').value || '',
  });
  if (!r.ok) { toast('Could not start: ' + r.error, 5000); return; }

  SESSION_CAMERA = r.camera;
  HAVE_EXTRINSICS = false;
  $('checks').innerHTML = '<p class="note">Nothing solved yet in this session.</p>';
  $('extr-out').innerHTML = '';
  $('click-hint').classList.add('hidden');
  $('marker').style.display = 'none';
  $('hdr-status').textContent = `${r.camera} · ${r.source} · ${r.board}`;
  // Cache-bust so restarting a session does not reuse the old MJPEG stream.
  $('stream').src = '/api/stream?t=' + Date.now();

  const db = $('device-banner');
  if (r.device) {
    db.classList.remove('hidden');
    db.className = 'banner ' + (r.device.asserted ? 'warn' : 'ok');
    db.textContent = r.device.asserted
      ? `Device serial ${r.device.serial} — operator-asserted, not `
        + `hardware-confirmed. Cross-check against the recorder's console output.`
      : `Connected: S/N ${r.device.serial}${r.device.model ? ' (' + r.device.model + ')' : ''}.`;
  } else {
    db.classList.add('hidden');
  }

  const loaded = r.loaded_intrinsics;
  if (loaded && !loaded.error) {
    $('solve-body').classList.remove('hidden');
    $('sec-save').classList.remove('hidden');
    const mismatch = loaded.synthetic && r.source !== 'synthetic';
    const rmsPart = loaded.rms == null ? 'factory-provided'
                                       : `RMS ${loaded.rms.toFixed(3)} px`;
    $('intr-state').className = 'banner ' + (mismatch ? 'bad' : 'ok');
    $('intr-state').textContent = mismatch
      ? `The saved intrinsics for ${r.camera} were measured from the SYNTHETIC `
        + `camera, but this session is running on '${r.source}'. They describe a `
        + `virtual lens, and the resolution matches, so nothing else would catch `
        + `this. Delete results/${r.camera}.json and calibrate the real lens first — `
        + `solving is refused until then.`
      : `Reloaded intrinsics for ${r.camera} (${loaded.method || 'board'}): measured `
        + `${loaded.created || 'previously'} at ${loaded.image_size[0]}×`
        + `${loaded.image_size[1]}, ${rmsPart}`
        + `${loaded.synthetic ? ' (synthetic — rehearsal only)' : ''}. `
        + 'Lay the board flat on the belt and solve.';
    toast(mismatch ? 'Synthetic intrinsics on a real camera — solving is blocked'
                   : 'Intrinsics reloaded — ready to solve',
          mismatch ? 8000 : 2600);
  } else {
    $('solve-body').classList.add('hidden');
    $('sec-save').classList.add('hidden');
    $('intr-state').className = 'banner bad';
    $('intr-state').textContent = loaded && loaded.error
      ? loaded.error
      : `No saved intrinsics for ${r.camera}. Calibrate its lens on the `
        + 'Intrinsics page first — extrinsics cannot be solved without them.';
  }
});

// ── Solve, and check against what was anticipated ───────────────────────────

$('btn-extr').addEventListener('click', async () => {
  const r = await post('/api/extrinsics', {
    origin_x_mm: num($('ox').value) ?? 0,
    origin_y_mm: num($('oy').value) ?? 0,
  });
  if (!r.ok) { toast(r.error, 6000); return; }
  HAVE_EXTRINSICS = true;

  const p = r.camera_position_mm;
  $('extr-out').innerHTML = `
    <div class="stat"><span class="k">camera position (belt frame)</span>
      <span class="v">[${p.map(v => v.toFixed(0)).join(', ')}] mm</span></div>
    <div class="stat"><span class="k">points used</span><span class="v">${r.n_points}</span></div>
    ${(r.warnings || []).map(w =>
      `<div class="note" style="color:var(--warn)">${esc(w)}</div>`).join('')}`;

  renderChecks(r);
  $('click-hint').classList.remove('hidden');
  $('click-hint').textContent =
    'Click anywhere on the preview to read that point in belt millimetres — '
    + 'two points a known distance apart, against a tape, tests the whole chain.';

  const v = r.verdict;
  toast(v === 'bad' ? 'Solved, but a check failed — read the panel before saving'
      : v === 'warn' ? 'Solved with a warning'
      : v === 'unset' ? 'Solved — but nothing to check it against'
      : 'Solved, all checks passed',
    v === 'ok' ? 3000 : 6000);
});

function renderChecks(r) {
  const rows = (r.checks || []).map(c => `
    <tr class="chk ${c.status}">
      <td>${esc(c.label)}</td>
      <td class="mono">${esc(c.measured)}</td>
      <td class="mono">${esc(c.expected)}</td>
      <td class="mono">${esc(c.delta)}</td>
      <td><span class="pill ${c.status === 'unset' ? 'idle' : c.status}">${
        c.status === 'unset' ? 'no ref' : c.status}</span></td>
    </tr>` + (c.hint ? `<tr class="chk-hint"><td colspan="5">${esc(c.hint)}</td></tr>` : ''))
    .join('');
  $('checks').innerHTML = `
    <div class="tablewrap"><table class="checks">
      <thead><tr><th>check</th><th>measured</th><th>anticipated</th><th>Δ</th><th></th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`;
}

$('stream').addEventListener('click', async (e) => {
  if (!HAVE_EXTRINSICS) return;
  const img = e.target;
  const rect = img.getBoundingClientRect();
  // The preview is scaled to fit; convert back to native pixel coordinates,
  // otherwise every clicked point would be wrong by the display scale factor.
  const x = (e.clientX - rect.left) * (img.naturalWidth / rect.width);
  const y = (e.clientY - rect.top) * (img.naturalHeight / rect.height);

  const m = $('marker');
  m.style.display = 'block';
  m.style.left = (e.clientX - rect.left) + 'px';
  m.style.top = (e.clientY - rect.top) + 'px';

  const r = await post('/api/to_belt', { x, y });
  if (!r.ok) { toast(r.error, 3500); return; }
  toast(`belt: X ${r.x_mm.toFixed(1)} mm, Y ${r.y_mm.toFixed(1)} mm  ·  `
        + `${r.mm_per_px.toFixed(3)} mm/px here`, 5000);
});

$('btn-save').addEventListener('click', async () => {
  const r = await post('/api/save', { notes: $('notes').value });
  if (!r.ok) { toast(r.error, 4000); return; }
  toast('Saved to ' + r.path, 4000);
  // The status board is the point of this page, so it must reflect the save
  // immediately — the next camera is chosen from it.
  await loadPlan();
  if (SESSION_CAMERA) $('camera').value = SESSION_CAMERA;
  onCameraPicked();
});

// ── Workspace map ───────────────────────────────────────────────────────────
// Generalizes the world plane past "a belt of width x length": upload a
// top-down map of whatever plane this rig watches and georeference it. A GLB
// arrives already georeferenced (glTF fixes metres and the model origin); a
// PNG is a picture until an origin and a scale are supplied, and the server
// refuses to use a half-georeferenced map rather than guessing units.

let WS = null;              // the server's map metadata, or null
let WS_PICK = null;         // 'origin' | 'scale' | 'axis' while picking
let WS_SCALE_PTS = [];

async function loadWorkspace() {
  const r = await api('/api/workspace');
  WS = r.ok ? r.map : null;
  if (WS && r.png) $('ws-img').src = 'data:image/png;base64,' + r.png;
  renderWorkspace();
}

function renderWorkspace() {
  const banner = $('ws-state');
  const body = $('ws-body');
  if (!WS) {
    banner.className = 'banner';
    banner.classList.remove('hidden');
    banner.textContent =
      'No workspace map. The belt width and length above define the map '
      + 'instead — which is fine for a conveyor, and limiting for anything else.';
    body.classList.add('hidden');
    return;
  }

  body.classList.remove('hidden');
  const geo = WS.is_georeferenced;
  banner.className = 'banner ' + (geo ? 'ok' : 'warn');
  banner.classList.remove('hidden');
  banner.textContent = geo
    ? `Map ready (${WS.source.toUpperCase()}, ${WS.image_size[0]}×${WS.image_size[1]} px, `
      + `${WS.mm_per_px.toFixed(3)} mm/px). The belt map now renders against it.`
    : `Map uploaded (${WS.source.toUpperCase()}, ${WS.image_size[0]}×${WS.image_size[1]} px) `
      + `but not yet georeferenced — set an origin and a scale below. Until both `
      + `exist it is a picture, not a map, and the belt map keeps using the typed `
      + `belt dimensions.`;

  $('ws-origin-val').textContent = WS.origin_px
    ? `${WS.origin_px[0].toFixed(0)}, ${WS.origin_px[1].toFixed(0)} px` : '—';
  $('ws-scale-val').textContent = WS.mm_per_px
    ? `${WS.mm_per_px.toFixed(3)} mm/px` : '—';
  if (WS.mm_per_px) $('ws-mmpp').value = WS.mm_per_px.toFixed(3);

  placeMarker('ws-marker-origin', WS.origin_px);

  const s = $('ws-status');
  if (geo) {
    const w = WS.image_size[0] * WS.mm_per_px, h = WS.image_size[1] * WS.mm_per_px;
    s.innerHTML = `<div class="stat"><span class="k">covers</span>`
      + `<span class="v">${(w / 1000).toFixed(2)} × ${(h / 1000).toFixed(2)} m</span></div>`
      + (WS.source === 'glb'
         ? `<p class="note">Scale and origin came from the glTF model itself — `
           + `no hand georeferencing was needed or used.</p>` : '');
  } else {
    s.innerHTML = '';
  }
}

/* Position an absolutely-placed marker at a point in MAP-IMAGE pixels,
   accounting for the <img> being scaled to fit its container. */
function placeMarker(id, pt_px) {
  const el = $(id), img = $('ws-img');
  if (!pt_px || !img.naturalWidth) { el.classList.add('hidden'); return; }
  const k = img.clientWidth / img.naturalWidth;
  el.style.left = (pt_px[0] * k + img.offsetLeft) + 'px';
  el.style.top = (pt_px[1] * k + img.offsetTop) + 'px';
  el.classList.remove('hidden');
}

function setPicking(mode, hint) {
  WS_PICK = mode;
  $('ws-viewport').classList.toggle('picking', mode !== null);
  $('ws-click-hint').textContent = hint || (WS && WS.is_georeferenced
    ? 'Map is georeferenced. Re-click any step to change it.'
    : 'Set an origin and a scale to georeference this map.');
}

$('ws-img').addEventListener('click', async (e) => {
  if (!WS_PICK) return;
  const img = e.target;
  const rect = img.getBoundingClientRect();
  // Back to native map pixels — the displayed image is scaled to fit, and
  // clicking in display space would georeference against the wrong units.
  const k = img.naturalWidth / rect.width;
  const p = [(e.clientX - rect.left) * k, (e.clientY - rect.top) * k];

  if (WS_PICK === 'origin') {
    setPicking(null);
    await georef({ origin_px: p });
  } else if (WS_PICK === 'axis') {
    setPicking(null);
    await georef({ y_point_px: p });
  } else if (WS_PICK === 'scale') {
    WS_SCALE_PTS.push(p);
    placeMarker(WS_SCALE_PTS.length === 1 ? 'ws-marker-a' : 'ws-marker-b', p);
    if (WS_SCALE_PTS.length >= 2) {
      setPicking(null, 'Two points marked — enter the real distance between '
                     + 'them and press Apply.');
      $('ws-dist').focus();
    } else {
      $('ws-click-hint').textContent = 'Now click the second point.';
    }
  }
});

async function georef(payload) {
  const r = await post('/api/workspace/georef', payload);
  if (!r.ok) { toast(r.error, 6000); return; }
  WS = r.map;
  if (r.png) $('ws-img').src = 'data:image/png;base64,' + r.png;
  renderWorkspace();
  toast('Map updated');
}

$('btn-ws-upload').addEventListener('click', async () => {
  const f = $('ws-file').files[0];
  if (!f) { toast('Choose a .png, .jpg or .glb first', 3000); return; }
  const fd = new FormData();
  fd.append('file', f);
  toast('Uploading…');
  const r = await api('/api/workspace/upload', { method: 'POST', body: fd });
  if (!r.ok) { toast(r.error, 7000); return; }
  WS = r.map;
  if (r.png) $('ws-img').src = 'data:image/png;base64,' + r.png;
  WS_SCALE_PTS = [];
  ['ws-marker-a', 'ws-marker-b'].forEach(i => $(i).classList.add('hidden'));
  renderWorkspace();
  setPicking(null);
  toast(WS.is_georeferenced
    ? 'Map uploaded and georeferenced from the model'
    : 'Map uploaded — now set the origin and scale', 5000);
});

$('btn-ws-clear').addEventListener('click', async () => {
  if (!confirm('Remove the workspace map?\n\nThe belt map will go back to '
             + 'using the typed belt width and length.')) return;
  const r = await api('/api/workspace', { method: 'DELETE' });
  if (!r.ok) { toast(r.error || 'Could not remove', 5000); return; }
  WS = null;
  WS_SCALE_PTS = [];
  renderWorkspace();
  toast(r.deleted ? 'Map removed' : 'There was no map to remove');
});

$('btn-ws-pick-origin').addEventListener('click', () =>
  setPicking('origin', 'Click the point on the map that is world (0, 0).'));

$('btn-ws-pick-axis').addEventListener('click', () => {
  if (!WS || !WS.origin_px) {
    toast('Set the origin first — the axis is defined relative to it', 4000);
    return;
  }
  setPicking('axis', 'Click a point lying along world +Y from the origin.');
});

$('btn-ws-pick-scale').addEventListener('click', () => {
  WS_SCALE_PTS = [];
  ['ws-marker-a', 'ws-marker-b'].forEach(i => $(i).classList.add('hidden'));
  setPicking('scale', 'Click the first of two points whose real separation '
                    + 'you have measured.');
});

$('btn-ws-apply-scale').addEventListener('click', async () => {
  if (WS_SCALE_PTS.length < 2) {
    toast('Click two points on the map first', 3500); return;
  }
  const d = parseFloat($('ws-dist').value);
  if (!Number.isFinite(d) || d <= 0) {
    toast('Enter the real distance between the two points, in mm', 4000); return;
  }
  await georef({ scale_points: WS_SCALE_PTS, distance_mm: d });
});

$('btn-ws-apply-mmpp').addEventListener('click', async () => {
  const v = parseFloat($('ws-mmpp').value);
  if (!Number.isFinite(v) || v <= 0) {
    toast('Enter a positive mm-per-pixel value', 4000); return;
  }
  await georef({ mm_per_px: v });
});

// Markers are positioned in display space, so they must follow a resize.
window.addEventListener('resize', () => {
  if (WS) {
    placeMarker('ws-marker-origin', WS.origin_px);
    if (WS_SCALE_PTS[0]) placeMarker('ws-marker-a', WS_SCALE_PTS[0]);
    if (WS_SCALE_PTS[1]) placeMarker('ws-marker-b', WS_SCALE_PTS[1]);
  }
});
$('ws-img').addEventListener('load', () => {
  if (WS) placeMarker('ws-marker-origin', WS.origin_px);
});

// ── Point correspondences (one map + one frame per camera) ──────────────────
// The board-free extrinsics route. Click a feature on the frozen camera frame,
// then the same feature on the workspace map above; four such pairs determine
// the plane mapping, and more make it measurable. World coordinates come from
// the map's georeference, so B2 must be complete first.

let CORR = { pairs: [], pending: null, imageSize: null, solved: false };
let CORR_VALIDATING = false;

function corrCameras() {
  const sel = $('corr-camera');
  const keep = sel.value;
  sel.innerHTML = '';
  ROSTER.forEach(row => {
    const o = document.createElement('option');
    o.value = row.name;
    o.textContent = `${row.name}${row.has_intrinsics ? '' : ' — no intrinsics'}`;
    sel.appendChild(o);
  });
  if (keep && ROSTER.some(r => r.name === keep)) sel.value = keep;
}

function corrRender() {
  const rows = $('corr-rows');
  rows.innerHTML = '';
  CORR.pairs.forEach((p, i) => {
    const tr = document.createElement('tr');
    const err = p.residual_mm === undefined ? '—'
      : `${p.residual_mm.toFixed(0)} mm`;
    const cls = p.inlier === false ? ' style="color:var(--bad)"' : '';
    tr.innerHTML = `<td>${i + 1}</td>
      <td class="mono">${p.image.map(v => v.toFixed(0)).join(', ')}</td>
      <td class="mono">${p.world.map(v => v.toFixed(0)).join(', ')}</td>
      <td class="mono"${cls}>${err}${p.inlier === false ? ' ✕' : ''}</td>
      <td><button class="danger small" data-drop="${i}">×</button></td>`;
    rows.appendChild(tr);
  });
  rows.querySelectorAll('button[data-drop]').forEach(b =>
    b.addEventListener('click', () => {
      CORR.pairs.splice(Number(b.dataset.drop), 1);
      CORR.solved = false;
      $('btn-corr-save').disabled = true;
      corrRender();
    }));

  const n = CORR.pairs.length;
  const half = CORR.pending ? ' · one half-finished pair — now click the map' : '';
  $('corr-hint').textContent =
    `${n} pair${n === 1 ? '' : 's'}${half}. `
    + (n < 4 ? `Need at least ${4 - n} more before solving.`
             : n === 4 ? 'Four is the minimum — a fifth gives a real error estimate.'
                       : 'Click the frame, then the matching point on the map.');
}

function corrBanner(cls, text) {
  const b = $('corr-state');
  b.className = 'banner ' + cls;
  b.classList.remove('hidden');
  b.textContent = text;
}

function corrReady() {
  if (!WS || !WS.is_georeferenced) {
    corrBanner('warn', 'Upload and georeference a workspace map in B2 first — '
      + 'without a scale and origin there are no world coordinates to click.');
    return false;
  }
  return true;
}

$('btn-corr-grab').addEventListener('click', async () => {
  if (!corrReady()) return;
  const r = await post('/api/correspondence/frame', {});
  if (!r.ok) { toast(r.error, 6000); return; }
  $('corr-img').src = 'data:image/png;base64,' + r.png;
  CORR = { pairs: [], pending: null, imageSize: r.image_size, solved: false };
  $('corr-camera').value = r.camera;
  $('corr-body').classList.remove('hidden');
  $('btn-corr-save').disabled = true;
  $('corr-result').innerHTML = '';
  $('corr-validate-row').classList.add('hidden');
  corrBanner('ok', `Frozen a frame from ${r.camera} `
                 + `(${r.image_size[0]}×${r.image_size[1]}). Now click matching `
                 + `points on the frame and the map.`);
  corrRender();
});

$('corr-img').addEventListener('click', async (e) => {
  const img = e.target;
  const rect = img.getBoundingClientRect();
  const k = img.naturalWidth / rect.width;
  const p = [(e.clientX - rect.left) * k, (e.clientY - rect.top) * k];

  if (CORR_VALIDATING) {
    CORR_VALIDATING = false;
    $('corr-viewport').classList.remove('picking');
    const v = await post('/api/correspondence/validate', { x: p[0], y: p[1] });
    if (!v.ok) { toast(v.error, 5000); return; }
    toast(`that pixel is world X ${v.x_mm.toFixed(0)} mm, Y ${v.y_mm.toFixed(0)} mm`
          + ' — check it against the map', 7000);
    return;
  }
  if (!corrReady()) return;
  CORR.pending = p;
  corrRender();
  toast('Now click the same feature on the workspace map above', 4000);
});

/* The map click completes a pair. Registered here rather than in the B2
   handler so the map keeps its own georeferencing behaviour untouched. */
$('ws-img').addEventListener('click', (e) => {
  if (WS_PICK || !CORR.pending || !WS || !WS.is_georeferenced) return;
  const img = e.target;
  const rect = img.getBoundingClientRect();
  const k = img.naturalWidth / rect.width;
  const mapPx = [(e.clientX - rect.left) * k, (e.clientY - rect.top) * k];
  // Map pixels -> world mm, using the same georeference the server holds.
  const world = [(mapPx[0] - WS.origin_px[0]) * WS.mm_per_px,
                 (mapPx[1] - WS.origin_px[1]) * WS.mm_per_px];
  CORR.pairs.push({ image: CORR.pending, world });
  CORR.pending = null;
  CORR.solved = false;
  $('btn-corr-save').disabled = true;
  corrRender();
});

$('btn-corr-undo').addEventListener('click', () => {
  if (CORR.pending) CORR.pending = null;
  else CORR.pairs.pop();
  CORR.solved = false;
  $('btn-corr-save').disabled = true;
  corrRender();
});

$('btn-corr-clear').addEventListener('click', () => {
  CORR.pairs = [];
  CORR.pending = null;
  CORR.solved = false;
  $('btn-corr-save').disabled = true;
  $('corr-result').innerHTML = '';
  corrRender();
});

$('btn-corr-solve').addEventListener('click', async () => {
  if (!corrReady()) return;
  if (CORR.pairs.length < 4) {
    toast(`Need at least 4 pairs, have ${CORR.pairs.length}`, 4000); return;
  }
  const r = await post('/api/correspondence/solve', {
    camera: $('corr-camera').value,
    pairs: CORR.pairs.map(p => ({ image: p.image, world: p.world })),
    image_size: CORR.imageSize,
  });
  if (!r.ok) { toast(r.error, 7000); return; }

  CORR.pairs.forEach((p, i) => {
    p.residual_mm = r.residuals_mm[i];
    p.inlier = r.inliers[i] === 1;
  });
  CORR.solved = true;
  $('btn-corr-save').disabled = false;
  $('corr-validate-row').classList.remove('hidden');
  corrRender();

  const pos = r.camera_position_mm;
  const cls = r.rms_mm < 20 ? 'ok' : r.rms_mm < 60 ? 'warn' : 'bad';
  $('corr-result').innerHTML = `
    <div class="stat"><span class="k">RMS on the plane</span>
      <span class="v">${r.rms_mm.toFixed(1)} mm
      <span class="pill ${cls}">${cls === 'ok' ? 'good' : cls}</span></span></div>
    <div class="stat"><span class="k">worst inlier</span>
      <span class="v">${r.max_mm.toFixed(1)} mm</span></div>
    <div class="stat"><span class="k">points used</span>
      <span class="v">${r.n_inliers} of ${r.n_points}</span></div>
    ${r.used_intrinsics && Number.isFinite(pos[2])
      ? `<div class="stat"><span class="k">camera height</span>
           <span class="v">${Math.abs(pos[2]).toFixed(0)} mm</span></div>`
      : `<div class="note">No intrinsics for this camera — the plane mapping
           is solved, but distortion is uncorrected and there is no camera
           position.</div>`}
    ${(r.warnings || []).map(w =>
       `<div class="note" style="color:var(--warn)">${esc(w)}</div>`).join('')}`;

  corrBanner(cls === 'bad' ? 'bad' : 'ok',
    `Solved ${$('corr-camera').value}: ${r.rms_mm.toFixed(1)} mm RMS across `
    + `${r.n_inliers} of ${r.n_points} points. Validate a point before saving.`);
  toast(`Solved — ${r.rms_mm.toFixed(1)} mm RMS`, 5000);
});

$('btn-corr-validate').addEventListener('click', () => {
  CORR_VALIDATING = true;
  $('corr-viewport').classList.add('picking');
  toast('Click any point on the frame — its world coordinates come back', 5000);
});

$('btn-corr-save').addEventListener('click', async () => {
  const r = await post('/api/correspondence/save', {
    notes: 'solved from workspace-map correspondences',
  });
  if (!r.ok) { toast(r.error, 5000); return; }
  toast('Saved to ' + r.path, 4000);
  PLAN = r.plan; ROSTER = r.roster;
  renderPlan(r.summary);
  corrCameras();
});

loadConfig().then(loadPlan).then(loadWorkspace).then(corrCameras);
