/* OVGU AMS — 3D-map extrinsics.
   Point-correspondence calibration against the conveyor .glb: pick a feature in
   the 3D mesh and the same feature in the camera frame, solve image->world.
   Vanilla ES module + vendored three.js (no build step). */

import * as THREE from 'three';
import { GLTFLoader } from '/vendor/three/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from '/vendor/three/jsm/controls/OrbitControls.js';

const S = {
  dataset: null, glbName: null, camName: null, intrName: null,
  K: null, D: null,
  pairs: [],                 // {world:[X,Z]mm, pos:THREE.Vector3, obj, imgPx:[u,v], dot}
  pendingWorld: null,        // {pos, obj}
  pendingImg: null,          // {px:[u,v], dot}
  solved: false,
  valMode: false, valPending: null, val: [],   // {pred:[X,Z], err|null}
  valObjs: [], valDots: [],                     // every validation 3D marker / image dot, for Clear
  frame: { ox: 0, oz: 0, ang: 0, hasOrigin: false, hasAxis: false },
  pickMode: null,               // 'origin' | 'axis' | null
};

// Ground-plane world frame: +X across the belt, +Y along the conveyor; mm.
// Height (Z) is deliberately ignored — the belt is treated as a flat plane, so a
// single homography maps image -> belt. (The belt's slight concavity/slant is
// not modelled; the pick's height component is simply dropped.)
function worldMm(p) {
  const dx = p.x - S.frame.ox, dz = p.z - S.frame.oz, a = S.frame.ang;
  const along = Math.cos(a) * dx + Math.sin(a) * dz;
  const across = -Math.sin(a) * dx + Math.cos(a) * dz;
  return [across * 1000, along * 1000];          // [worldX, worldY]
}
function worldToPoint(mmX, mmY) {
  const across = mmX / 1000, along = mmY / 1000, a = S.frame.ang;
  const dx = Math.cos(a) * along - Math.sin(a) * across;
  const dz = Math.sin(a) * along + Math.cos(a) * across;
  return new THREE.Vector3(S.frame.ox + dx, 0, S.frame.oz + dz);
}

// ── three.js scene ──────────────────────────────────────────────────────────
let scene, camera, renderer, controls, raycaster, meshRoot = null;
const canvas = $('c3d-canvas');

function init3D() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d0f13);
  const wrap = $('c3d-canvas-wrap');
  camera = new THREE.PerspectiveCamera(50, wrap.clientWidth / wrap.clientHeight, 0.01, 100);
  camera.position.set(1.2, 2.0, 2.0);
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  raycaster = new THREE.Raycaster();

  scene.add(new THREE.AmbientLight(0xffffff, 1.1));
  const dir = new THREE.DirectionalLight(0xffffff, 1.4); dir.position.set(2, 4, 3); scene.add(dir);
  scene.add(new THREE.AxesHelper(0.3));                 // world origin (X=red, Y=green up, Z=blue)
  const grid = new THREE.GridHelper(3, 30, 0x334, 0x223); scene.add(grid);   // ground grid at Y=0

  resize3D();
  addEventListener('resize', resize3D);
  (function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();
}
function resize3D() {
  const wrap = $('c3d-canvas-wrap');
  renderer.setSize(wrap.clientWidth, wrap.clientHeight, false);
  camera.aspect = wrap.clientWidth / wrap.clientHeight; camera.updateProjectionMatrix();
}

function clearMesh() { if (meshRoot) { scene.remove(meshRoot); meshRoot = null; } }

function loadGLB(url) {
  clearMesh();
  new GLTFLoader().load(url, (gltf) => {
    meshRoot = gltf.scene; scene.add(meshRoot);
    // Show the map from either side — a flat map quad is easy to view from its back.
    meshRoot.traverse(o => {
      if (o.isMesh && o.material) {
        o.material.side = THREE.DoubleSide;
        if (o.material.map) o.material.map.needsUpdate = true;
      }
    });
    const box = new THREE.Box3().setFromObject(meshRoot);
    const c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3());
    const r = Math.max(s.x, s.z) * 0.9 + 0.5;
    controls.target.copy(new THREE.Vector3(c.x, 0, c.z));
    camera.position.set(c.x, r * 1.1, c.z + r);
    camera.near = 0.01; camera.far = r * 20; camera.updateProjectionMatrix();
    hud(`mesh loaded · ${s.x.toFixed(2)}×${s.z.toFixed(2)} m`);
  }, undefined, (err) => { toast('GLB load failed: ' + err, 5000); });
}

function ndc(e) {
  const r = canvas.getBoundingClientRect();
  return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1,
                           -((e.clientY - r.top) / r.height) * 2 + 1);
}
function marker(pos, color, rad = 0.02) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(rad, 16, 16),
                           new THREE.MeshBasicMaterial({ color }));
  m.position.copy(pos); scene.add(m); return m;
}

canvas.addEventListener('dblclick', (e) => {
  if (!meshRoot) return;
  raycaster.setFromCamera(ndc(e), camera);
  const hit = raycaster.intersectObject(meshRoot, true)[0];
  if (!hit) { hud('no surface under cursor'); return; }
  const p = hit.point;
  if (S.pickMode === 'origin') {
    S.frame.ox = p.x; S.frame.oz = p.z; S.frame.hasOrigin = true; S.pickMode = null;
    drawFrame(); recomputePairs(); frameState(); hud('origin set'); return;
  }
  if (S.pickMode === 'axis') {
    S.frame.ang = Math.atan2(p.z - S.frame.oz, p.x - S.frame.ox);
    S.frame.hasAxis = true; S.pickMode = null;
    drawFrame(); recomputePairs(); frameState(); hud('conveyor axis set'); return;
  }
  if (S.valMode) { onValidateActual(p); return; }
  if (S.pendingWorld) scene.remove(S.pendingWorld.obj);
  S.pendingWorld = { pos: p.clone(), obj: marker(p, 0x33e0ff, 0.018) };
  const w = worldMm(p);
  hud(`3D pick  worldX=${w[0].toFixed(0)}  worldY=${w[1].toFixed(0)} mm`);
  tryPair();
});

// ── camera frame ─────────────────────────────────────────────────────────────
const img = $('c3d-img'), imgWrap = $('c3d-img-wrap');
function imgPixel(e) {
  const r = img.getBoundingClientRect();
  return [ (e.clientX - r.left) * (img.naturalWidth / r.width),
           (e.clientY - r.top) * (img.naturalHeight / r.height) ];
}
function imgDot(e, cls) {
  const wr = imgWrap.getBoundingClientRect();
  const d = document.createElement('div'); d.className = 'imgdot' + (cls ? ' ' + cls : '');
  d.style.left = (e.clientX - wr.left) + 'px'; d.style.top = (e.clientY - wr.top) + 'px';
  imgWrap.appendChild(d); return d;
}
img.addEventListener('dblclick', (e) => {
  if (!img.src) return;
  const px = imgPixel(e);
  if (S.valMode) { onValidatePredict(px, e); return; }
  if (S.pendingImg) S.pendingImg.dot.remove();
  S.pendingImg = { px, dot: imgDot(e, 'pending') };
  tryPair();
});

function tryPair() {
  if (!S.pendingWorld || !S.pendingImg) return;
  const p = S.pendingWorld.pos;
  S.pairs.push({ world: worldMm(p), pos: p.clone(), obj: S.pendingWorld.obj,
                 imgPx: S.pendingImg.px, dot: S.pendingImg.dot });
  S.pendingWorld = null; S.pendingImg = null;
  renderPairs(); hud(`${S.pairs.length} pair(s)`);
}
function recomputePairs() { S.pairs.forEach(p => { p.world = worldMm(p.pos); }); renderPairs(); }

// origin marker + a line along the conveyor (+Y) drawn on the ground
let frameGroup = null;
function drawFrame() {
  if (frameGroup) scene.remove(frameGroup);
  frameGroup = new THREE.Group();
  const o = new THREE.Vector3(S.frame.ox, 0, S.frame.oz);
  frameGroup.add(marker(o, 0x00ff88, 0.025));
  const end = worldToPoint(0, 800);   // 0.8 m along +Y
  const g = new THREE.BufferGeometry().setFromPoints([o, end]);
  frameGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x00ff88 })));
  const endX = worldToPoint(400, 0);  // 0.4 m along +X
  const gx = new THREE.BufferGeometry().setFromPoints([o, endX]);
  frameGroup.add(new THREE.Line(gx, new THREE.LineBasicMaterial({ color: 0xff5c7a })));
  scene.add(frameGroup);
}
function frameState() {
  const deg = (S.frame.ang * 180 / Math.PI).toFixed(1);
  $('c3d-frame-state').innerHTML =
    `Frame: origin ${S.frame.hasOrigin ? 'set' : '(0,0)'} · +Y axis ${S.frame.hasAxis ? deg + '° on mesh' : 'mesh X'}` +
    (S.pairs.length ? ` · <span class="mono">pairs recomputed</span>` : '');
}
function renderPairs(resid) {
  const rows = S.pairs.map((p, i) => `<tr><td>${i + 1}</td>
     <td class="mono">${p.imgPx[0].toFixed(0)}, ${p.imgPx[1].toFixed(0)}</td>
     <td class="mono">${p.world[0].toFixed(0)}, ${p.world[1].toFixed(0)}</td>
     <td class="mono">${resid ? resid[i].toFixed(0) + ' mm' : '—'}</td>
     <td><button class="ghost small" data-del="${i}">✕</button></td></tr>`).join('');
  $('c3d-pair-rows').innerHTML = rows;
  $('c3d-pair-rows').querySelectorAll('[data-del]').forEach(b =>
    b.onclick = () => { const i = +b.dataset.del; scene.remove(S.pairs[i].obj); S.pairs[i].dot.remove();
                        S.pairs.splice(i, 1); renderPairs(); });
  $('c3d-solve').disabled = S.pairs.length < 4;
}

// ── solve ─────────────────────────────────────────────────────────────────────
$('c3d-solve').addEventListener('click', async () => {
  const noIntr = $('c3d-nointr').checked;
  const rectified = $('c3d-rectified').checked;
  const payload = {
    camera: S.camName, image_points: S.pairs.map(p => p.imgPx),
    world_points_mm: S.pairs.map(p => p.world),
    image_size: [img.naturalWidth, img.naturalHeight] };
  if (!noIntr) {                                   // omit K/D -> pure homography, no undistort, no pose
    payload.K = S.K;
    payload.D = rectified ? [0, 0, 0, 0, 0] : (S.D || [0, 0, 0, 0, 0]);
  }
  const r = await post('/api/calib3d/solve', payload);
  if (!r.ok) { toast(r.error, 6000); return; }
  S.solved = true; $('c3d-val-toggle').disabled = false; $('c3d-save').disabled = false;
  renderPairs(r.residuals_mm);
  const pos = r.camera_position_mm;
  const warn = (r.warnings || []).map(w => `<li>${w}</li>`).join('');
  $('c3d-result').innerHTML = `
    <div class="stat"><span class="k">RMS</span><span class="v mono">${r.rms_mm.toFixed(1)} mm</span></div>
    <div class="stat"><span class="k">max residual</span><span class="v mono">${r.max_mm.toFixed(1)} mm</span></div>
    ${pos ? `<div class="stat"><span class="k">camera height</span>
       <span class="v mono">${Math.abs(pos[2]).toFixed(0)} mm</span></div>
      <div class="stat"><span class="k">camera pos (X,Y,Z)</span>
       <span class="v mono">${pos.map(v => v.toFixed(0)).join(', ')}</span></div>` : ''}
    ${warn ? `<ul class="note" style="margin-top:8px">${warn}</ul>` : ''}`;
  // plot camera position in 3D
  if (pos) { if (S._camMk) scene.remove(S._camMk);
             const cp = worldToPoint(pos[0], pos[1]); cp.y = Math.abs(pos[2]) / 1000;
             S._camMk = marker(cp, 0xff5c7a, 0.03); }
  toast('Solved · RMS ' + r.rms_mm.toFixed(1) + ' mm');
});
$('c3d-save').addEventListener('click', async () => {
  const r = await post('/api/calib3d/save', { camera: S.camName });
  if (!r.ok) { toast(r.error, 6000); return; }
  toast('Saved extrinsics → ' + r.path, 5000);
  $('c3d-hdr').textContent = `${S.dataset} · ${S.camName} · saved`;
});
$('c3d-undo').onclick = () => { const p = S.pairs.pop(); if (p) { scene.remove(p.obj); p.dot.remove(); renderPairs(); } };
$('c3d-clear').onclick = () => { S.pairs.forEach(p => { scene.remove(p.obj); p.dot.remove(); }); S.pairs = []; renderPairs(); };

// ── validation ─────────────────────────────────────────────────────────────────
$('c3d-val-toggle').addEventListener('click', () => {
  S.valMode = !S.valMode;
  $('c3d-val-toggle').textContent = 'Validation mode: ' + (S.valMode ? 'ON' : 'OFF');
  $('c3d-val-toggle').classList.toggle('accent', S.valMode);
  hud(S.valMode ? 'validation: dbl-click a test point in the frame' : '');
});
async function onValidatePredict(px, e) {
  const r = await post('/api/calib3d/validate', { camera: S.camName, x: px[0], y: px[1] });
  if (!r.ok) { toast(r.error, 5000); return; }
  const mk = marker(worldToPoint(r.x_mm, r.y_mm), 0xffd23f, 0.02);
  const dot = imgDot(e, 'val');
  S.valObjs.push(mk); S.valDots.push(dot);
  S.valPending = { pred: [r.x_mm, r.y_mm] };
  hud(`predicted world  X=${r.x_mm.toFixed(0)}  Z=${r.y_mm.toFixed(0)} mm · now dbl-click where it should be on the map`);
}
function onValidateActual(p) {
  if (!S.valPending) { hud('predict first: dbl-click the frame'); return; }
  const w = worldMm(p);
  const err = Math.hypot(w[0] - S.valPending.pred[0], w[1] - S.valPending.pred[1]);
  S.valObjs.push(marker(p, 0x8affc1, 0.015));
  S.val.push({ pred: S.valPending.pred, err });
  S.valPending = null;
  renderKPI();
}
function clearValidationVisuals() {
  S.valObjs.forEach(o => scene.remove(o));
  S.valDots.forEach(d => d.remove());
  S.valObjs = []; S.valDots = []; S.val = []; S.valPending = null;
}
$('c3d-val-clear').onclick = () => { clearValidationVisuals(); renderKPI(); };
function renderKPI() {
  $('c3d-val-count').textContent = S.val.length + ' points';
  const errs = S.val.filter(v => v.err != null).map(v => v.err);
  if (!errs.length) { $('c3d-kpi').innerHTML = '<p class="note">No validated points yet.</p>'; return; }
  const avg = errs.reduce((a, b) => a + b, 0) / errs.length;
  const mx = Math.max(...errs);
  const sd = Math.sqrt(errs.reduce((a, b) => a + (b - avg) ** 2, 0) / errs.length);
  const rating = avg < 20 ? 'Excellent' : avg < 50 ? 'Good' : avg < 100 ? 'Fair' : 'Poor';
  $('c3d-kpi').innerHTML = `
    <div class="stat"><span class="k">avg error</span><span class="v mono">${avg.toFixed(0)} mm</span></div>
    <div class="stat"><span class="k">max error</span><span class="v mono">${mx.toFixed(0)} mm</span></div>
    <div class="stat"><span class="k">std dev</span><span class="v mono">${sd.toFixed(0)} mm</span></div>
    <div class="stat"><span class="k">points</span><span class="v mono">${errs.length}</span></div>
    <div class="stat"><span class="k">rating</span><span class="v">${rating}</span></div>`;
}

function hud(t) { $('c3d-hud').textContent = t || ''; }

// ── map-frame controls ──────────────────────────────────────────────────────────
$('c3d-set-origin').onclick = () => { S.pickMode = 'origin'; hud('dbl-click the origin board (8×11, under basler_1)'); };
$('c3d-set-axis').onclick = () => {
  if (!S.frame.hasOrigin) { toast('set the origin first'); return; }
  S.pickMode = 'axis'; hud('dbl-click a point along the conveyor — a far board');
};
$('c3d-frame-reset').onclick = () => {
  S.frame = { ox: 0, oz: 0, ang: 0, hasOrigin: false, hasAxis: false };
  if (frameGroup) { scene.remove(frameGroup); frameGroup = null; }
  recomputePairs(); frameState();
};
$('c3d-frame-save').addEventListener('click', async () => {
  const r = await post('/api/calib3d/frame', {
    dataset: S.dataset, glb: S.glbName,
    ox: S.frame.ox, oz: S.frame.oz, ang: S.frame.ang });
  if (!r.ok) { toast(r.error, 5000); return; }
  toast('Frame saved with the map');
});
async function loadSavedFrame() {
  const r = await (await fetch(`/api/calib3d/frame?dataset=${encodeURIComponent(S.dataset)}&glb=${encodeURIComponent(S.glbName)}`)).json();
  if (r.ok && r.frame) {
    S.frame = { ox: r.frame.ox, oz: r.frame.oz, ang: r.frame.ang, hasOrigin: true, hasAxis: true };
    drawFrame(); frameState(); hud('loaded saved map frame');
  }
}

// ── sources / dataset wiring ────────────────────────────────────────────────────
async function loadDatasets() {
  const r = await (await fetch('/api/datasets')).json();
  S._datasets = r.datasets || [];
  const sel = $('c3d-dataset'); sel.innerHTML = '';
  S._datasets.forEach(d => sel.add(new Option(d.name, d.name)));
  if (S._datasets.length) fillDataset(S._datasets[0].name);
  sel.onchange = () => fillDataset(sel.value);
}
function fillDataset(name) {
  const d = S._datasets.find(x => x.name === name); if (!d) return;
  const fill = (id, arr) => { const s = $(id); s.innerHTML = ''; arr.forEach(v => s.add(new Option(v, v))); };
  fill('c3d-map', d.glb); fill('c3d-cam', d.images); fill('c3d-intr', d.intrinsics);
  // auto-pick the intrinsics that matches the chosen camera name
  $('c3d-cam').onchange = () => {
    const base = $('c3d-cam').value.replace(/\.[^.]+$/, '');
    const m = d.intrinsics.find(i => i.includes(base)); if (m) $('c3d-intr').value = m;
  };
  $('c3d-cam').onchange();
}
$('c3d-load').addEventListener('click', async () => {
  S.dataset = $('c3d-dataset').value; S.glbName = $('c3d-map').value;
  S.camName = $('c3d-cam').value.replace(/\.[^.]+$/, ''); S.intrName = $('c3d-intr').value;
  const base = `/api/dataset/${S.dataset}`;
  const bust = `?t=${Date.now()}`;              // defeat browser caching of the .glb/frame
  loadGLB(`${base}/${S.glbName}${bust}`);
  img.src = `${base}/${$('c3d-cam').value}${bust}`;
  $('c3d-img-badge').textContent = 'CAMERA FRAME · ' + S.camName;
  if (S.intrName) {
    const j = await (await fetch(`${base}/${S.intrName}`)).json();
    S.K = j.intrinsics.K; S.D = (j.intrinsics.D || []).concat([0, 0, 0, 0, 0]).slice(0, 5);
  }
  $('c3d-hdr').textContent = `${S.dataset} · ${S.camName}`;
  $('c3d-set-origin').disabled = false; $('c3d-set-axis').disabled = false;
  $('c3d-frame-save').disabled = false;
  frameState();
  loadSavedFrame();
  toast('Loaded ' + S.camName);
});

init3D();
loadDatasets();
