/* OVGU AMS calibration tool — helpers shared by both pages.
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

/* Populate the source and board pickers, which both pages need identically. */
function fillSourceAndBoardSelects(config, sourceEl, boardEl) {
  sourceEl.innerHTML = '';
  config.sources.forEach(s => {
    const o = document.createElement('option');
    o.value = s.id;
    o.textContent = s.label + (s.available ? '' : ' — unavailable');
    o.disabled = !s.available;
    o.dataset.note = s.note;
    sourceEl.appendChild(o);
  });
  boardEl.innerHTML = '';
  config.boards.forEach(bd => {
    const o = document.createElement('option');
    o.value = bd.id;
    o.textContent = `${bd.name} — ${bd.squares[0]}x${bd.squares[1]}, ${bd.square_mm}mm (${bd.paper})`;
    boardEl.appendChild(o);
  });
}

/* Escape anything that came from disk before it goes near innerHTML. Camera
   names and notes are operator-typed and land in the roster table; they are not
   hostile, but a stray angle bracket silently eating the rest of a row is the
   kind of bug that gets blamed on the calibration instead of the markup. */
function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
