/* OVGU AMS — validate extrinsics across the dashboard map.
   Double-click a feature in any camera frame; its saved extrinsics projects it to
   a belt position, plotted on the dashboard in that camera's colour. Same physical
   point from several cameras should coincide. */

import * as THREE from 'three';
import { GLTFLoader } from '/vendor/three/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from '/vendor/three/jsm/controls/OrbitControls.js';

const marks = [];                         // {obj, dot}
let scene, camera, renderer, controls, dashboard = null;
let showBags = false;
const canvas = $('val-canvas');

function init3D() {
  scene = new THREE.Scene(); scene.background = new THREE.Color(0x0d0f13);
  const wrap = $('val-canvas-wrap');
  camera = new THREE.PerspectiveCamera(50, wrap.clientWidth / wrap.clientHeight, 0.01, 100);
  camera.position.set(0.4, 2.4, 1.6);
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
  scene.add(new THREE.AmbientLight(0xffffff, 1.2));
  const d = new THREE.DirectionalLight(0xffffff, 1.2); d.position.set(1, 3, 2); scene.add(d);
  scene.add(new THREE.AxesHelper(0.15));
  scene.add(new THREE.GridHelper(3, 30, 0x334, 0x223));
  resize(); addEventListener('resize', resize);
  new GLTFLoader().load('/api/dataset/dashboard/conveyor_dashboard.glb?t=' + Date.now(), (g) => {
    dashboard = g.scene; scene.add(dashboard);
    dashboard.traverse(o => { if (o.isMesh && o.material) { o.material.side = THREE.DoubleSide;
      if (o.material.map) o.material.map.needsUpdate = true; } });
    const b = new THREE.Box3().setFromObject(dashboard), c = b.getCenter(new THREE.Vector3());
    controls.target.copy(c); camera.position.set(c.x + 0.3, 1.8, c.z + 1.4);
  }, undefined, (e) => toast('dashboard map load failed: ' + e, 5000));
  (function loop(){ requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();
}
function resize() {
  const w = $('val-canvas-wrap'); renderer.setSize(w.clientWidth, w.clientHeight, false);
  camera.aspect = w.clientWidth / w.clientHeight; camera.updateProjectionMatrix();
}

function plot(xMm, yMm, colorHex) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.018, 16, 16),
                           new THREE.MeshBasicMaterial({ color: colorHex }));
  m.position.set(xMm / 1000, 0.012, yMm / 1000);   // belt X across, Y along; slightly above surface
  scene.add(m); return m;
}

// ── camera frame panels ──────────────────────────────────────────────────────
async function loadCameras() {
  const r = await (await fetch('/api/validate/cameras')).json();
  const cams = r.cameras || [];
  $('val-hdr').textContent = cams.length + ' cameras';
  const box = $('val-cams'); box.innerHTML = '';
  if (!cams.length) { box.innerHTML = '<p class="note">No saved extrinsics found. Solve some in the 3D-map page first.</p>'; return; }
  cams.forEach(c => {
    const card = document.createElement('div'); card.className = 'cam-card';
    const rms = (c.rms_mm != null) ? `${c.rms_mm.toFixed(1)} mm` : '—';
    const first = (showBags && c.frame_bags) ? c.frame_bags : c.frame;
    card.innerHTML =
      `<div class="hd"><span class="swatch" style="background:${c.color}"></span>
         <span>${c.camera}</span>
         <span class="note" style="margin:0 0 0 auto">rms ${rms}${c.used_intrinsics ? '' : ' · no-intr'}</span></div>
       <div class="frame-wrap">${c.frame ? `<img alt="${c.camera}" data-board="${c.frame}" data-bags="${c.frame_bags || ''}" src="${first}?t=${Date.now()}">`
                                          : '<p class="note" style="padding:10px">frame missing</p>'}</div>`;
    box.appendChild(card);
    const img = card.querySelector('img'); if (!img) return;
    const wrap = card.querySelector('.frame-wrap');
    img.addEventListener('dblclick', async (e) => {
      const rect = img.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (img.naturalWidth / rect.width);
      const y = (e.clientY - rect.top) * (img.naturalHeight / rect.height);
      const res = await post('/api/validate/project', { camera: c.camera, x, y });
      if (!res.ok) { toast(res.error, 5000); return; }
      // dot on the frame
      const dot = document.createElement('div'); dot.className = 'cdot';
      dot.style.background = c.color;
      dot.style.left = (e.clientX - wrap.getBoundingClientRect().left) + 'px';
      dot.style.top = (e.clientY - wrap.getBoundingClientRect().top) + 'px';
      wrap.appendChild(dot);
      // marker on the map
      const obj = plot(res.x_mm, res.y_mm, c.color);
      marks.push({ obj, dot });
      $('val-count').textContent = marks.length + ' points';
      $('val-hud').textContent = `${c.camera} → belt  X=${res.x_mm.toFixed(0)}  Y=${res.y_mm.toFixed(0)} mm`;
    });
  });
}

function clearAll() {
  marks.forEach(m => { scene.remove(m.obj); m.dot.remove(); });
  marks.length = 0; $('val-count').textContent = '0 points';
}
$('val-clear').onclick = () => { clearAll(); $('val-hud').textContent = 'cleared'; };
$('val-frameset').onclick = () => {
  showBags = !showBags;
  $('val-frameset').textContent = 'Show: ' + (showBags ? 'polybag frames' : 'board frames');
  $('val-frameset').classList.toggle('accent', showBags);
  clearAll();                                        // dots belong to the old frame
  document.querySelectorAll('.cam-card img').forEach(img => {
    const url = (showBags && img.dataset.bags) ? img.dataset.bags : img.dataset.board;
    if (url) img.src = url + '?t=' + Date.now();
  });
  $('val-hud').textContent = showBags ? 'polybag frames — click a bag' : 'board frames';
};

init3D();
loadCameras();
