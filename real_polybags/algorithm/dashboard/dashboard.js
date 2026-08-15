/*
 * AMS digital-twin dashboard (consumer).
 *
 * Pure static page: it subscribes to the MQTT `/polybags` topic over WebSocket
 * and draws each camera's polybag positions on the top-down conveyor map. It is
 * fully decoupled from the streamer — any publisher of the same message shape
 * drives it. No cross-camera fusion: every (camera, id) is its own dot.
 *
 * World mm -> map pixel mirrors worldmap.WorkspaceMap.mm_to_px:
 *     map_px = origin_px + world_mm / mm_per_px
 * Map pixel -> screen applies the pan/zoom view.
 */
"use strict";

const PALETTE = {
  basler_1:     "#e23c3c",
  basler_2:     "#3cd23c",
  lucid:        "#28b4eb",
  rgbd_1_color: "#c828c8",
};
const FALLBACK = ["#e2a03c", "#8c8cff", "#e2e23c", "#3ce2c8"];

const cv = document.getElementById("map");
const ctx = cv.getContext("2d");
const statusEl = document.getElementById("status");
const countEl = document.getElementById("count");
const legendEl = document.getElementById("legend");

const state = {
  cfg: null,
  mapImg: null,
  tracks: new Map(),        // "cam#id" -> {cam,id,x,y,conf,metric,rx}
  ttlMs: 1000,
  view: { scale: 1, x: 0, y: 0 },
  fitted: false,
  drag: null,
};

// ── coordinate transforms ────────────────────────────────────────────────────
function worldToMap(x, y) {
  const g = state.cfg.georef;
  return [g.origin_px[0] + x / g.mm_per_px, g.origin_px[1] + y / g.mm_per_px];
}
function mapToScreen(px, py) {
  const v = state.view;
  return [px * v.scale + v.x, py * v.scale + v.y];
}

function colorFor(cam, i) {
  return PALETTE[cam] || FALLBACK[i % FALLBACK.length];
}

// ── canvas sizing + initial fit ──────────────────────────────────────────────
function resize() {
  const dpr = window.devicePixelRatio || 1;
  const r = cv.getBoundingClientRect();
  cv.width = Math.round(r.width * dpr);
  cv.height = Math.round(r.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (state.mapImg && !state.fitted) fitMap();
}
function fitMap() {
  const r = cv.getBoundingClientRect();
  const img = state.mapImg;
  const s = Math.min(r.width / img.width, r.height / img.height) * 0.96;
  state.view.scale = s;
  state.view.x = (r.width - img.width * s) / 2;
  state.view.y = (r.height - img.height * s) / 2;
  state.fitted = true;
}

// ── pan + zoom ───────────────────────────────────────────────────────────────
cv.addEventListener("mousedown", (e) => {
  state.drag = { x: e.clientX, y: e.clientY, vx: state.view.x, vy: state.view.y };
});
window.addEventListener("mouseup", () => { state.drag = null; });
window.addEventListener("mousemove", (e) => {
  if (!state.drag) return;
  state.view.x = state.drag.vx + (e.clientX - state.drag.x);
  state.view.y = state.drag.vy + (e.clientY - state.drag.y);
});
cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = cv.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const f = Math.exp(-e.deltaY * 0.0015);
  const v = state.view;
  // zoom about the cursor: keep the map point under the cursor fixed
  v.x = mx - (mx - v.x) * f;
  v.y = my - (my - v.y) * f;
  v.scale *= f;
}, { passive: false });

// ── render loop ──────────────────────────────────────────────────────────────
function render() {
  const r = cv.getBoundingClientRect();
  ctx.clearRect(0, 0, r.width, r.height);
  ctx.fillStyle = "#0d0f13";
  ctx.fillRect(0, 0, r.width, r.height);

  if (state.mapImg) {
    const v = state.view;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(state.mapImg, v.x, v.y,
      state.mapImg.width * v.scale, state.mapImg.height * v.scale);
  }

  // prune stale, then draw
  const now = Date.now();
  let live = 0;
  for (const [k, t] of state.tracks) {
    if (now - t.rx > state.ttlMs) { state.tracks.delete(k); continue; }
    live++;
    drawTrack(t);
  }
  countEl.textContent = `${live} polybag${live === 1 ? "" : "s"}`;
  requestAnimationFrame(render);
}

// Small fixed-size square marker (screen px) standing in for a polybag — not
// the real footprint. Same style for every camera. Independent of zoom.
const BAG_PX = 6;   // half-size of the square, screen pixels

function drawTrack(t) {
  const [px, py] = worldToMap(t.x, t.y);
  const [sx, sy] = mapToScreen(px, py);
  const col = colorFor(t.cam, 0);
  const h = BAG_PX;

  ctx.fillStyle = col;
  ctx.globalAlpha = 0.85;
  ctx.fillRect(sx - h, sy - h, 2 * h, 2 * h);
  ctx.globalAlpha = 1;
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "rgba(0,0,0,.5)";
  ctx.strokeRect(sx - h, sy - h, 2 * h, 2 * h);

  ctx.fillStyle = "rgba(231,237,243,.9)";
  ctx.font = "10px monospace";
  ctx.fillText(`${t.id}`, sx + h + 2, sy - h - 2);
}

// ── MQTT ─────────────────────────────────────────────────────────────────────
function connectMqtt() {
  const m = state.cfg.mqtt;
  const host = location.hostname || m.host || "localhost";
  const url = `ws://${host}:${m.ws_port}`;
  const client = mqtt.connect(url, { reconnectPeriod: 1500 });

  client.on("connect", () => {
    statusEl.textContent = "MQTT ● live";
    statusEl.className = "on";
    client.subscribe(m.topic);
  });
  client.on("reconnect", () => { statusEl.textContent = "MQTT ◦ reconnecting"; statusEl.className = "off"; });
  client.on("close", () => { statusEl.textContent = "MQTT ◦ offline"; statusEl.className = "off"; });
  client.on("error", () => { statusEl.textContent = "MQTT ✕ error"; statusEl.className = "off"; });

  client.on("message", (_topic, payload) => {
    let msg;
    try { msg = JSON.parse(payload.toString()); } catch { return; }
    const rx = Date.now();
    for (const p of (msg.polybags || [])) {
      state.tracks.set(`${p.cam}#${p.id}`,
        { cam: p.cam, id: p.id, x: p.x_mm, y: p.y_mm, conf: p.conf, metric: p.metric, rx });
    }
  });
}

// ── legend ───────────────────────────────────────────────────────────────────
function buildLegend() {
  const rows = state.cfg.cameras.map((c, i) => {
    const col = colorFor(c.name, i);
    const tag = c.metric ? "" : " · homography-only";
    return `<div class="row"><span class="dot" style="background:${col}"></span>${c.name}${tag}</div>`;
  }).join("");
  legendEl.innerHTML = rows +
    `<div class="hint">drag to pan · wheel to zoom</div>`;
}

// ── boot ─────────────────────────────────────────────────────────────────────
async function boot() {
  state.cfg = await (await fetch("/api/config")).json();
  state.ttlMs = (state.cfg.mqtt.track_ttl_s || 1.0) * 1000;
  document.getElementById("hdr-sub").textContent =
    `conveyor · session ${state.cfg.session} · live polybag positions`;
  buildLegend();

  const img = new Image();
  img.onload = () => { state.mapImg = img; resize(); };
  img.src = state.cfg.map_url;

  window.addEventListener("resize", resize);
  resize();
  connectMqtt();
  requestAnimationFrame(render);
}
boot();
