"""
Workspace map — OVGU AMS calibration tool.

Generalizes the tool beyond this conveyor: the world plane no longer has to be
"a belt of width x length". Any rig whose cameras watch a common plane can
upload a **top-down map of that plane** and georeference it — define where the
world origin sits and what a pixel is worth in millimetres — and every
downstream artefact (camera footprints, the mosaic, coverage and overlap)
renders against that map instead of a blank rectangle.

Two upload forms:

- **PNG/JPEG** — a floor plan, a CAD export, a stitched photo. Carries no
  units, so it must be georeferenced by hand: click the origin, then either
  type mm-per-pixel or click two points a known (tape-measured) distance
  apart. The same two-clicks-and-a-tape ethos as the rest of the tool.
- **GLB (binary glTF)** — e.g. a scan or CAD model of the cell. Parsed here
  directly (stdlib only — the container is 12 bytes of header plus a JSON and
  a BIN chunk; no dependency needed for the triangle-mesh subset) and rendered
  top-down. glTF's spec fixes the units at **metres** and +Y as up, so scale
  and origin come out of the file itself: mm-per-pixel is exact and the world
  origin is the model's own origin. No hand georeferencing required — which is
  the one real advantage of the format, and why it is worth parsing.

Conventions, chosen to match the existing belt frame so nothing downstream
changes: world X = map-image right, world Y = map-image down, Z = 0 is the
mapped plane. For GLB, the render looks down the glTF +Y (up) axis, with
world X = glTF X and world Y = glTF Z.

Stored under `results/_workspace/` — a subdirectory, like `_plan/`, because
both the belt map and the results summary enumerate camera calibrations with
`results/*.json` and must not pick up a map's metadata as a camera.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

SCHEMA_VERSION = 1
WORKSPACE_DIRNAME = "_workspace"
MAP_FILENAME = "map.png"
META_FILENAME = "meta.json"


# ── Storage paths ────────────────────────────────────────────────────────────

def workspace_dir(results_dir: Path) -> Path:
    return Path(results_dir) / WORKSPACE_DIRNAME


def map_path(results_dir: Path) -> Path:
    return workspace_dir(results_dir) / MAP_FILENAME


def meta_path(results_dir: Path) -> Path:
    return workspace_dir(results_dir) / META_FILENAME


# ── The map itself ───────────────────────────────────────────────────────────

@dataclass
class WorkspaceMap:
    """A georeferenced top-down image of the world plane.

    `origin_px` is where world (0, 0) sits in the image; `mm_per_px` is the
    scale. Either may be None until the operator sets them — an image without
    both is a picture, not a map, and `is_georeferenced` says which one you
    have. Nothing downstream will consume a non-georeferenced map: rendering a
    metric overlay against unknown units would produce confident nonsense.
    """
    image_size: tuple[int, int]                  # (w, h) px
    mm_per_px: float | None = None
    origin_px: tuple[float, float] | None = None
    source: str = "png"                          # "png" | "glb"
    notes: str = ""

    @property
    def is_georeferenced(self) -> bool:
        return self.mm_per_px is not None and self.origin_px is not None

    # World +X = image right, +Y = image down — the same handedness the belt
    # frame already uses, so a map-derived frame drops in without sign flips.
    def px_to_mm(self, pts_px) -> np.ndarray:
        self._require_georef()
        p = np.asarray(pts_px, np.float64).reshape(-1, 2)
        o = np.asarray(self.origin_px, np.float64)
        return (p - o) * self.mm_per_px

    def mm_to_px(self, pts_mm) -> np.ndarray:
        self._require_georef()
        p = np.asarray(pts_mm, np.float64).reshape(-1, 2)
        o = np.asarray(self.origin_px, np.float64)
        return p / self.mm_per_px + o

    def _require_georef(self):
        if not self.is_georeferenced:
            missing = []
            if self.mm_per_px is None:
                missing.append("scale (mm per pixel)")
            if self.origin_px is None:
                missing.append("origin")
            raise ValueError("map is not georeferenced yet — missing "
                             + " and ".join(missing))

    def frame_dict(self) -> dict:
        """The world extent this map covers, as BeltFrame constructor kwargs.

        Returned as a plain dict rather than a BeltFrame to keep this module
        importable on its own; the caller builds the BeltFrame.
        """
        self._require_georef()
        w, h = self.image_size
        ox, oy = self.origin_px
        s = self.mm_per_px
        return {"x_min_mm": -ox * s, "x_max_mm": (w - ox) * s,
                "y_min_mm": -oy * s, "y_max_mm": (h - oy) * s,
                "mm_per_px": s}

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "image_size": list(self.image_size),
            "mm_per_px": self.mm_per_px,
            "origin_px": list(self.origin_px) if self.origin_px else None,
            "source": self.source,
            "notes": self.notes,
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


def scale_from_points(p1_px, p2_px, distance_mm: float) -> float:
    """mm-per-pixel from two clicked points a tape-measured distance apart.

    The honest way to georeference a picture: no unit metadata is trusted,
    only a physical measurement — the same principle as the rest of the tool.
    """
    d_px = float(np.linalg.norm(np.asarray(p2_px, float) - np.asarray(p1_px, float)))
    if d_px < 10:
        raise ValueError(f"the two points are only {d_px:.1f} px apart — too "
                         f"close to set a scale reliably; click further apart")
    if distance_mm <= 0:
        raise ValueError("distance must be positive millimetres")
    return distance_mm / d_px


def rotate_to_axis(image: np.ndarray, origin_px, y_point_px
                   ) -> tuple[np.ndarray, tuple[float, float]]:
    """Rotate the map so the origin->y_point direction becomes world +Y (down).

    The rotation is baked into the stored image once, at set-time, so every
    later consumer stays axis-aligned — a running rotation term threaded
    through every transform would be a standing invitation for sign errors.
    Returns (rotated image, new origin_px).
    """
    o = np.asarray(origin_px, np.float64)
    p = np.asarray(y_point_px, np.float64)
    v = p - o
    if np.linalg.norm(v) < 10:
        raise ValueError("axis point is too close to the origin to define a "
                         "direction — click further away")
    # Angle from image +Y (down) to the clicked direction; rotate back by it.
    ang = np.degrees(np.arctan2(v[0], v[1]))          # 0 when already down
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D(tuple(o), -ang, 1.0)

    # Expand the canvas so no part of the map is cut off by the rotation.
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    rc = (M[:, :2] @ corners.T).T + M[:, 2]
    shift = -rc.min(axis=0)
    M[:, 2] += shift
    size = tuple(np.ceil(rc.max(axis=0) + shift).astype(int))
    rotated = cv2.warpAffine(image, M, size, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(40, 40, 40))
    new_origin = (M[:, :2] @ o) + M[:, 2]
    return rotated, (float(new_origin[0]), float(new_origin[1]))


# ── Persistence ──────────────────────────────────────────────────────────────

def save(m: WorkspaceMap, image: np.ndarray, results_dir: Path) -> Path:
    d = workspace_dir(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(map_path(results_dir)), image)
    meta_path(results_dir).write_text(json.dumps(m.to_dict(), indent=2) + "\n")
    return d


def save_meta(m: WorkspaceMap, results_dir: Path) -> None:
    meta_path(results_dir).write_text(json.dumps(m.to_dict(), indent=2) + "\n")


def load(results_dir: Path) -> tuple[WorkspaceMap, np.ndarray] | None:
    """The stored map and its image, or None if none has been uploaded."""
    mp, tp = map_path(results_dir), meta_path(results_dir)
    if not (mp.exists() and tp.exists()):
        return None
    meta = json.loads(tp.read_text())
    image = cv2.imread(str(mp))
    if image is None:
        return None
    m = WorkspaceMap(
        image_size=(image.shape[1], image.shape[0]),
        mm_per_px=meta.get("mm_per_px"),
        origin_px=tuple(meta["origin_px"]) if meta.get("origin_px") else None,
        source=meta.get("source", "png"),
        notes=meta.get("notes", ""))
    return m, image


def delete(results_dir: Path) -> bool:
    d = workspace_dir(results_dir)
    if not d.exists():
        return False
    for f in d.iterdir():
        f.unlink()
    d.rmdir()
    return True


def background_for_frame(image: np.ndarray, m: WorkspaceMap, frame) -> np.ndarray:
    """The map resampled into a BeltFrame's pixel grid, as the mosaic backdrop.

    Frame px -> world mm -> map px is affine (both grids are axis-aligned by
    construction), so a single warpAffine does it exactly.
    """
    m._require_georef()
    s_f, s_m = frame.mm_per_px, m.mm_per_px
    ox, oy = m.origin_px
    # dst(frame px) -> src(map px):  src = dst * s_f/s_m + (frame.min/s_m + o)
    M = np.array([[s_f / s_m, 0.0, frame.x_min_mm / s_m + ox],
                  [0.0, s_f / s_m, frame.y_min_mm / s_m + oy]])
    return cv2.warpAffine(image, M, (frame.width_px, frame.height_px),
                          flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(26, 26, 26))


# ── GLB (binary glTF 2.0) — minimal, dependency-free ─────────────────────────

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_COMPONENT = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2),
              5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def parse_glb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (N,3 float, glTF metres) and triangle faces (M,3 int).

    Deliberately supports only the plain triangle-mesh subset — indexed or
    non-indexed TRIANGLES with float positions, node TRS/matrix hierarchies.
    Draco-compressed or sparse geometry is refused by name rather than
    half-read: a wrong mesh silently rendered would georeference the whole
    rig against garbage.
    """
    data = Path(path).read_bytes()
    if len(data) < 12:
        raise ValueError("not a GLB file (too short)")
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise ValueError("not a GLB file (bad magic — is this a .gltf JSON "
                         "file? Export as binary .glb instead)")
    if version != 2:
        raise ValueError(f"unsupported glTF version {version}")

    gltf, binbuf, off = None, b"", 12
    while off + 8 <= len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + clen]
        if ctype == _CHUNK_JSON:
            gltf = json.loads(chunk)
        elif ctype == _CHUNK_BIN:
            binbuf = chunk
        off += 8 + clen + (-clen % 4)
    if gltf is None:
        raise ValueError("GLB has no JSON chunk")

    exts = set(gltf.get("extensionsRequired", []))
    if "KHR_draco_mesh_compression" in exts:
        raise ValueError("this GLB uses Draco compression, which needs the "
                         "Draco decoder — re-export without compression")

    def read_accessor(idx: int) -> np.ndarray:
        acc = gltf["accessors"][idx]
        if acc.get("sparse"):
            raise ValueError("sparse accessors are not supported")
        fmt, size = _COMPONENT[acc["componentType"]]
        n = _NCOMP[acc["type"]]
        bv = gltf["bufferViews"][acc["bufferView"]]
        start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = bv.get("byteStride") or size * n
        count = acc["count"]
        out = np.empty((count, n), np.float64)
        for i in range(count):
            out[i] = struct.unpack_from(f"<{n}{fmt}", binbuf, start + i * stride)
        return out

    def node_matrix(node: dict) -> np.ndarray:
        if "matrix" in node:
            return np.array(node["matrix"], float).reshape(4, 4).T  # col-major
        M = np.eye(4)
        if "scale" in node:
            M = M @ np.diag(list(node["scale"]) + [1.0])
        if "rotation" in node:                       # quaternion x,y,z,w
            x, y, z, w = node["rotation"]
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])
            M4 = np.eye(4); M4[:3, :3] = R
            M = M4 @ M
        if "translation" in node:
            T = np.eye(4); T[:3, 3] = node["translation"]
            M = T @ M
        return M

    all_verts, all_faces = [], []

    def walk(node_idx: int, parent: np.ndarray):
        node = gltf["nodes"][node_idx]
        M = parent @ node_matrix(node)
        if "mesh" in node:
            mesh = gltf["meshes"][node["mesh"]]
            for prim in mesh.get("primitives", []):
                if prim.get("mode", 4) != 4:         # TRIANGLES only
                    continue
                pos = read_accessor(prim["attributes"]["POSITION"])
                v4 = np.column_stack([pos, np.ones(len(pos))])
                v = (M @ v4.T).T[:, :3]
                base = sum(len(x) for x in all_verts)
                all_verts.append(v)
                if "indices" in prim:
                    idx = read_accessor(prim["indices"]).astype(np.int64).ravel()
                else:
                    idx = np.arange(len(pos), dtype=np.int64)
                all_faces.append(idx.reshape(-1, 3) + base)
        for child in node.get("children", []):
            walk(child, M)

    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    for root in scene.get("nodes", []):
        walk(root, np.eye(4))

    if not all_verts or not all_faces:
        raise ValueError("no triangle geometry found in the GLB")
    return np.vstack(all_verts), np.vstack(all_faces)


def render_glb_topdown(path: Path, max_px: int = 2000
                       ) -> tuple[np.ndarray, WorkspaceMap]:
    """Render a GLB straight down onto its ground plane, georeferenced.

    glTF fixes units at metres and +Y as up, so this needs no hand
    calibration: world X = glTF X, world Y = glTF Z, one metre = 1000 mm, and
    the world origin is the model's own origin. Faces are painted lowest
    first, so the visible surface at each pixel is the highest one — a plan
    view — with brightness encoding height for legibility.
    """
    verts, faces = parse_glb(path)
    xz = verts[:, [0, 2]] * 1000.0                   # metres -> mm, plan axes
    height = verts[:, 1]

    lo, hi = xz.min(axis=0), xz.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    pad = 0.02 * span.max()
    lo, hi = lo - pad, hi + pad
    mm_per_px = float(max(hi - lo) / max_px)
    size = np.maximum(np.ceil((hi - lo) / mm_per_px).astype(int), 1)

    img = np.full((size[1], size[0], 3), 26, np.uint8)
    h_lo, h_hi = float(height.min()), float(height.max())
    h_span = max(h_hi - h_lo, 1e-9)

    order = np.argsort(height[faces].mean(axis=1))   # lowest first
    px = ((xz - lo) / mm_per_px).astype(np.int32)
    for fi in order:
        tri = px[faces[fi]]
        shade = 40 + 190 * (height[faces[fi]].mean() - h_lo) / h_span
        cv2.fillPoly(img, [tri], (shade, shade, shade))

    origin_px = tuple((-lo / mm_per_px).astype(float))
    m = WorkspaceMap(image_size=(int(size[0]), int(size[1])),
                     mm_per_px=mm_per_px, origin_px=origin_px, source="glb",
                     notes=f"rendered top-down from GLB: {len(verts)} vertices, "
                           f"{len(faces)} faces; scale and origin from the "
                           f"glTF model itself (metres, model origin)")
    return img, m
