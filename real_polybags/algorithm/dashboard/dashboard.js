/*
 * Camera-system digital-twin dashboard (consumer) — side-by-side RAW + FUSED.
 *
 * Subscribes over MQTT-WebSocket to both streams and draws them on two maps:
 *   - RAW   (/polybags)        left  : per-camera detections, one dot per (cam,id)
 *   - FUSED (/polybags_fused)  right : global objects after dedup + re-ID
 * The two maps share the belt image and pan/zoom (drag/scroll either).
 *
 * World mm -> map pixel mirrors worldmap.WorkspaceMap.mm_to_px:
 *     map_px = origin_px + world_mm / mm_per_px ; then pan/zoom to screen.
 */
"use strict";

const PALETTE = {
  basler_1: "#e23c3c", basler_2: "#3cd23c",
  lucid: "#28b4eb", rgbd_1_color: "#c828c8",
};
const FALLBACK = ["#e2a03c", "#8c8cff", "#e2e23c", "#3ce2c8"];
const FUSED_COLOR = "#ffd23f";

const statusEl = document.getElementById("status");
const state = {
  cfg: null, mapImg: null,
  raw: new Map(), fused: new Map(),
  counts: null,                     // latest line-crossing counts from the streamer
  playback: null,                   // { paused, loop, pos_s, duration_s }
  ttlMs: 1000,
  view: { scale: 1, x: 0, y: 0 },   // shared across both panels
  fitted: false, drag: null,
};

// The two side-by-side maps. `data` is the live Map each one renders.
const PANELS = [
  { key: "raw", cv: document.getElementById("map-raw"),
    countEl: document.getElementById("count-raw"),
    legendEl: document.getElementById("legend-raw"), data: state.raw },
  { key: "fused", cv: document.getElementById("map-fused"),
    countEl: document.getElementById("count-fused"),
    legendEl: document.getElementById("legend-fused"), data: state.fused },
];
PANELS.forEach(p => { p.ctx = p.cv.getContext("2d"); });

function worldToMap(x, y) {
  const g = state.cfg.georef;
  return [g.origin_px[0] + x / g.mm_per_px, g.origin_px[1] + y / g.mm_per_px];
}
function mapToScreen(px, py) {
  const v = state.view;
  return [px * v.scale + v.x, py * v.scale + v.y];
}
function colorFor(cam, i) { return PALETTE[cam] || FALLBACK[i % FALLBACK.length]; }

// ── sizing + fit ─────────────────────────────────────────────────────────────
function resize() {
  const dpr = window.devicePixelRatio || 1;
  for (const p of PANELS) {
    const r = p.cv.getBoundingClientRect();
    p.cv.width = Math.round(r.width * dpr);
    p.cv.height = Math.round(r.height * dpr);
    p.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  if (state.mapImg && !state.fitted) fitMap();
}
function fitMap() {
  const r = PANELS[0].cv.getBoundingClientRect();
  const img = state.mapImg;
  const s = Math.min(r.width / img.width, r.height / img.height) * 0.96;
  state.view.scale = s;
  state.view.x = (r.width - img.width * s) / 2;
  state.view.y = (r.height - img.height * s) / 2;
  state.fitted = true;
}

// ── pan + zoom (either panel drives the shared view) ─────────────────────────
for (const p of PANELS) {
  p.cv.addEventListener("mousedown", (e) => {
    state.drag = { x: e.clientX, y: e.clientY, vx: state.view.x, vy: state.view.y };
  });
  p.cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = p.cv.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const f = Math.exp(-e.deltaY * 0.0015);
    const v = state.view;
    v.x = mx - (mx - v.x) * f;
    v.y = my - (my - v.y) * f;
    v.scale *= f;
  }, { passive: false });
}
window.addEventListener("mouseup", () => { state.drag = null; });
window.addEventListener("mousemove", (e) => {
  if (!state.drag) return;
  state.view.x = state.drag.vx + (e.clientX - state.drag.x);
  state.view.y = state.drag.vy + (e.clientY - state.drag.y);
});

// ── render ───────────────────────────────────────────────────────────────────
const BAG_PX = 6;

function render() {
  const now = Date.now();
  for (const p of PANELS) {
    const r = p.cv.getBoundingClientRect();
    const ctx = p.ctx;
    ctx.clearRect(0, 0, r.width, r.height);
    ctx.fillStyle = "#0d0f13";
    ctx.fillRect(0, 0, r.width, r.height);
    if (state.mapImg) {
      const v = state.view;
      ctx.drawImage(state.mapImg, v.x, v.y,
        state.mapImg.width * v.scale, state.mapImg.height * v.scale);
    }
    drawCountLines(ctx, p.key);
    let live = 0;
    for (const [k, t] of p.data) {
      if (now - t.rx > state.ttlMs) { p.data.delete(k); continue; }
      live++;
      drawTrack(ctx, t, p.key);
    }
    const noun = p.key === "fused" ? "object" : "detection";
    const lines = state.counts && state.counts.lines;
    let passed = "";
    if (lines) passed = lines.map(l =>
      `<div class="passed">⎯ ${l.name}: <b>≈${l.flux_max}</b>` +
      `<span class="ref"> (cross ${l.crossing_max})</span></div>`
    ).join("");
    p.countEl.innerHTML =
      `<div>${live} ${noun}${live === 1 ? "" : "s"} now</div>` + passed;
  }
  requestAnimationFrame(render);
}

// Draw each counting line (world y, spanning its x-segment) with its running count.
function drawCountLines(ctx, key) {
  const cls = (state.cfg && state.cfg.count_lines) || [];
  const lines = (state.counts && state.counts.lines) || [];
  ctx.save();
  for (const cl of cls) {
    const xs = cl.x_mm || [-290, 290];
    const [ax, ay] = mapToScreen(...worldToMap(xs[0], cl.y_mm));
    const [bx, by] = mapToScreen(...worldToMap(xs[1], cl.y_mm));
    ctx.strokeStyle = "#35d07f"; ctx.lineWidth = 2; ctx.setLineDash([8, 6]);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    ctx.setLineDash([]);
    const cnt = lines.find(l => l.name === cl.name);
    const n = cnt ? cnt.flux_max : "";
    ctx.fillStyle = "#7be6a6"; ctx.font = "bold 12px monospace";
    ctx.fillText(`⎯ ${cl.name}: ≈${n}`, Math.min(ax, bx) + 3, Math.min(ay, by) - 4);
  }
  ctx.restore();
}

function drawTrack(ctx, t, key) {
  const [px, py] = worldToMap(t.x, t.y);
  const [sx, sy] = mapToScreen(px, py);
  const fused = key === "fused";
  const reid = fused && t.reid;         // ID restored across the gap
  const col = fused ? FUSED_COLOR : colorFor(t.cam, 0);
  const h = fused ? BAG_PX + 2 : BAG_PX;

  ctx.fillStyle = col; ctx.globalAlpha = 0.85;
  ctx.fillRect(sx - h, sy - h, 2 * h, 2 * h);
  ctx.globalAlpha = 1;
  // re-ID'd objects get a bright white outline so they stand out
  ctx.lineWidth = reid ? 2.5 : 1.5;
  ctx.strokeStyle = reid ? "#ffffff" : "rgba(0,0,0,.5)";
  ctx.strokeRect(sx - h, sy - h, 2 * h, 2 * h);

  ctx.fillStyle = reid ? "#ffffff" : "rgba(231,237,243,.95)";
  ctx.font = "bold 11px monospace";
  const label = fused
    ? `${t.id}${reid ? "*" : ""}${t.n_cams > 1 ? " (" + t.n_cams + ")" : ""}`
    : `${t.id}`;
  ctx.fillText(label, sx + h + 2, sy - h - 2);
}

// ── playback controls (pause / loop) ─────────────────────────────────────────
async function control(action) {
  try {
    const s = await (await fetch(`/api/control?action=${action}`,
      { method: "POST" })).json();
    reflectPlayback(s);
  } catch (e) { /* streamer busy or offline; ignore */ }
}
function reflectPlayback(s) {
  if (!s) return;
  state.playback = s;
  const bp = document.getElementById("btn-play");
  const bl = document.getElementById("btn-loop");
  if (bp) bp.textContent = s.paused ? "▶ Play" : "⏸ Pause";
  if (bl) {
    bl.textContent = s.loop ? "🔁 Loop: on" : "➡ Loop: off";
    bl.classList.toggle("off", !s.loop);
  }
}
function wireControls() {
  const bp = document.getElementById("btn-play");
  const bl = document.getElementById("btn-loop");
  if (bp) bp.onclick = () => control("toggle");
  if (bl) bl.onclick = () =>
    control(state.playback && state.playback.loop ? "loop_off" : "loop_on");
}

// ── MQTT (both topics) ───────────────────────────────────────────────────────
function connectMqtt() {
  const m = state.cfg.mqtt;
  const host = location.hostname || m.host || "localhost";
  const client = mqtt.connect(`ws://${host}:${m.ws_port}`, { reconnectPeriod: 1500 });
  const rawTopic = m.topic, fusedTopic = m.fused_topic;

  client.on("connect", () => {
    statusEl.textContent = "MQTT ● live"; statusEl.className = "on";
    client.subscribe(rawTopic);
    if (fusedTopic) client.subscribe(fusedTopic);
  });
  client.on("reconnect", () => { statusEl.textContent = "MQTT ◦ reconnecting"; statusEl.className = "off"; });
  client.on("close", () => { statusEl.textContent = "MQTT ◦ offline"; statusEl.className = "off"; });
  client.on("error", () => { statusEl.textContent = "MQTT ✕ error"; statusEl.className = "off"; });

  client.on("message", (topic, payload) => {
    let msg;
    try { msg = JSON.parse(payload.toString()); } catch { return; }
    const rx = Date.now();
    if (msg.counts) state.counts = msg.counts;   // same object on both topics
    if (msg.playback) {                          // keep buttons in sync (e.g. auto-pause at end)
      const p = state.playback;
      if (!p || p.paused !== msg.playback.paused || p.loop !== msg.playback.loop)
        reflectPlayback(msg.playback);
      else state.playback = msg.playback;
    }
    if (topic === fusedTopic) {
      for (const p of (msg.polybags || []))
        state.fused.set(p.id, { id: p.id, x: p.x_mm, y: p.y_mm,
          conf: p.conf, metric: p.metric, n_cams: p.n_cams || 1,
          reid: !!p.reid, rx });
    } else {
      for (const p of (msg.polybags || []))
        state.raw.set(`${p.cam}#${p.id}`, { cam: p.cam, id: p.id,
          x: p.x_mm, y: p.y_mm, conf: p.conf, metric: p.metric, rx });
    }
  });
}

// ── legends ──────────────────────────────────────────────────────────────────
function buildLegends() {
  PANELS[0].legendEl.innerHTML = state.cfg.cameras.map((c, i) =>
    `<div class="row"><span class="dot" style="background:${colorFor(c.name, i)}"></span>${c.name}${c.metric ? "" : " · homog."}</div>`
  ).join("") + `<div class="hint">⎯ = max single-camera count</div>`;
  const reid = state.cfg.reid_enabled;
  PANELS[1].legendEl.innerHTML =
    `<div class="row"><span class="dot" style="background:${FUSED_COLOR}"></span>fused object (n cams)</div>` +
    `<div class="row"><span class="dot" style="background:#35d07f"></span>count line ⎯</div>` +
    `<div class="hint">deduped${reid ? " + re-ID" : " · re-ID off"}</div>`;
}

// ── boot ─────────────────────────────────────────────────────────────────────
async function boot() {
  state.cfg = await (await fetch("/api/config")).json();
  state.ttlMs = (state.cfg.mqtt.track_ttl_s || 1.0) * 1000;
  document.getElementById("hdr-sub").textContent =
    `conveyor · session ${state.cfg.session} · raw vs fused`;
  wireControls();
  reflectPlayback(state.cfg.playback);
  buildLegends();

  const img = new Image();
  img.onload = () => { state.mapImg = img; resize(); };
  img.src = state.cfg.map_url;

  window.addEventListener("resize", resize);
  resize();
  connectMqtt();
  requestAnimationFrame(render);
}
boot();
