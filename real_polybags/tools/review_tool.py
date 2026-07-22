"""
Overlay viewer — browse auto-labeled frames with zoom/pan.
Run: python3 review_tool.py  →  open http://localhost:8080
"""

from pathlib import Path
from flask import Flask, render_template_string, send_file

BASE         = Path(__file__).resolve().parent.parent
OVERLAYS_DIR = BASE / "full_dataset" / "overlays"
LABELS_AUTO  = BASE / "full_dataset" / "labels_auto"
LABELS_MANUAL= BASE / "full_dataset" / "labels_manual"

manual_stems = {p.stem for p in LABELS_MANUAL.glob("*.txt")}
auto_frames  = sorted(
    p.stem for p in LABELS_AUTO.glob("*.txt")
    if p.stem not in manual_stems
)

app = Flask(__name__)

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Overlay Viewer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #111; color: #eee;
  display: flex; flex-direction: column; align-items: center;
  min-height: 100vh; padding: 16px;
}
#header {
  width: 100%; max-width: 1100px;
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 12px;
}
#title { font-size: 1.05rem; font-weight: 600; }
#frame-info { font-size: 0.82rem; color: #888; }

#image-container {
  width: 100%; max-width: 1100px; position: relative;
  background: #000; border-radius: 6px; overflow: hidden;
  margin-bottom: 10px; border: 1px solid #2a2a2a;
  cursor: zoom-in; user-select: none;
}
#image-container.panning { cursor: grabbing; }
#image-container.zoomed  { cursor: grab; }
#overlay-img {
  width: 100%; display: block;
  transform-origin: 0 0; will-change: transform;
}
#zoom-indicator {
  position: absolute; bottom: 8px; left: 10px;
  background: rgba(0,0,0,0.55); color: #ccc;
  font-size: 0.72rem; padding: 2px 7px; border-radius: 4px; pointer-events: none;
}

#controls {
  display: flex; gap: 10px; align-items: center;
  margin-bottom: 10px;
}
button {
  padding: 8px 22px; border: none; border-radius: 6px;
  font-size: 0.95rem; font-weight: 600; cursor: pointer;
  background: #2a2a2a; color: #ccc; border: 1px solid #444;
}
button:hover   { background: #383838; }
button:disabled { opacity: 0.3; cursor: not-allowed; }
.zoom-btn { padding: 6px 14px; }

#jump-form { display: flex; gap: 6px; align-items: center; }
#jump-input {
  width: 70px; padding: 7px; border: 1px solid #444;
  background: #222; color: #eee; border-radius: 6px;
  text-align: center; font-size: 0.9rem;
}
#hint { font-size: 0.72rem; color: #444; margin-top: 4px; }
</style>
</head>
<body>

<div id="header">
  <div id="title">Overlay Viewer</div>
  <div id="frame-info">—</div>
</div>

<div id="image-container">
  <img id="overlay-img" src="" alt="overlay" draggable="false">
  <div id="zoom-indicator">100%</div>
</div>

<div id="controls">
  <button id="btn-prev" onclick="go(-1)">← Prev</button>
  <button id="btn-next" onclick="go(1)">Next →</button>
  <button class="zoom-btn" onclick="zoomBy(0.25)">＋</button>
  <button class="zoom-btn" onclick="zoomBy(-0.25)">－</button>
  <button class="zoom-btn" onclick="resetZoom()">Reset</button>
  <div id="jump-form">
    <input id="jump-input" type="number" min="1" placeholder="#">
    <button onclick="jumpTo()">Go</button>
  </div>
</div>
<div id="hint">[←/→] Navigate &nbsp;|&nbsp; scroll to zoom &nbsp;|&nbsp; drag to pan &nbsp;|&nbsp; [+/-] Zoom &nbsp;|&nbsp; [Z] Reset zoom</div>

<script>
const frames = {{ frames|tojson }};
let idx = 0;
const Z = { scale:1, x:0, y:0, dragging:false, lx:0, ly:0 };

function applyZoom() {
  const img = document.getElementById('overlay-img');
  img.style.transform = `translate(${Z.x}px,${Z.y}px) scale(${Z.scale})`;
  const pct = Math.round(Z.scale*100)+'%';
  document.getElementById('zoom-indicator').textContent = pct;
  document.getElementById('image-container').classList.toggle('zoomed', Z.scale > 1);
}
function resetZoom() { Z.scale=1; Z.x=0; Z.y=0; applyZoom(); }
function zoomBy(d) {
  const c = document.getElementById('image-container');
  const cx = c.clientWidth/2, cy = c.clientHeight/2;
  const ns = Math.max(0.5, Math.min(10, Z.scale+d));
  const ix = (cx-Z.x)/Z.scale, iy = (cy-Z.y)/Z.scale;
  Z.x = cx-ix*ns; Z.y = cy-iy*ns; Z.scale = ns; applyZoom();
}

document.getElementById('image-container').addEventListener('wheel', e => {
  e.preventDefault();
  const c = document.getElementById('image-container');
  const r = c.getBoundingClientRect();
  const mx = e.clientX-r.left, my = e.clientY-r.top;
  const f  = e.deltaY < 0 ? 1.12 : 1/1.12;
  const ns = Math.max(0.5, Math.min(10, Z.scale*f));
  const ix = (mx-Z.x)/Z.scale, iy = (my-Z.y)/Z.scale;
  Z.x = mx-ix*ns; Z.y = my-iy*ns; Z.scale = ns; applyZoom();
}, { passive: false });

document.getElementById('image-container').addEventListener('mousedown', e => {
  if (Z.scale <= 1) return;
  e.preventDefault();
  Z.dragging=true; Z.lx=e.clientX; Z.ly=e.clientY;
  document.getElementById('image-container').classList.add('panning');
});
document.addEventListener('mousemove', e => {
  if (!Z.dragging) return;
  Z.x += e.clientX-Z.lx; Z.y += e.clientY-Z.ly; Z.lx=e.clientX; Z.ly=e.clientY; applyZoom();
});
document.addEventListener('mouseup', () => {
  Z.dragging=false;
  document.getElementById('image-container').classList.remove('panning');
});

function render() {
  resetZoom();
  document.getElementById('overlay-img').src = '/overlay/'+frames[idx]+'?t='+Date.now();
  document.getElementById('frame-info').textContent =
    frames[idx] + '  —  ' + (idx+1) + ' / ' + frames.length;
  document.getElementById('btn-prev').disabled = idx === 0;
  document.getElementById('btn-next').disabled = idx === frames.length-1;
}

function go(d) { idx = Math.max(0, Math.min(frames.length-1, idx+d)); render(); }
function jumpTo() {
  const v = parseInt(document.getElementById('jump-input').value);
  if (!isNaN(v)) { idx = Math.max(0, Math.min(frames.length-1, v-1)); render(); }
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if      (e.key==='ArrowLeft')        go(-1);
  else if (e.key==='ArrowRight')       go(1);
  else if (e.key==='+'||e.key==='=')   zoomBy(0.25);
  else if (e.key==='-')                zoomBy(-0.25);
  else if (e.key==='z'||e.key==='Z')   resetZoom();
});

render();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    from flask import render_template_string
    return render_template_string(HTML, frames=auto_frames)

@app.route("/overlay/<stem>")
def overlay(stem):
    path = OVERLAYS_DIR / f"{stem}.png"
    if not path.exists():
        return "not found", 404
    return send_file(path, mimetype="image/png")

if __name__ == "__main__":
    print(f"{len(auto_frames)} frames  →  http://localhost:8080")
    app.run(debug=False, port=8080)
