#!/usr/bin/env python3
"""
Extract a clean, leveled, origin-set conveyor map from a room scan — for CalibrationHub.

Input : a Scaniverse .glb of the conveyor + surrounding room (tilted, cluttered).
Output: a .glb containing only the conveyor, with the belt plane made horizontal,
        +X along the belt, +Z across it, and the world origin translated to the
        8x11 checkerboard (under basler_1). Texture and UVs are preserved.

CalibrationHub ingests .glb maps whose metres are the world scale and whose z=0
plane is the ground; it cannot rotate a tilted mesh, so leveling must be baked in
here. Written with the stdlib only (glTF binary = 12-byte header + JSON chunk +
BIN chunk), so it needs no mesh library.
"""
from __future__ import annotations
import struct, json, sys, argparse
import numpy as np
import cv2

CT = {5120:('b',1),5121:('B',1),5122:('h',2),5123:('H',2),5125:('I',4),5126:('f',4)}
NC = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}


def load_glb(path):
    d = bytes(open(path,'rb').read()); L = struct.unpack('<I', d[8:12])[0]
    off = 12; js = None; bina = b''
    while off < L:
        clen, ctype = struct.unpack('<II', d[off:off+8]); off += 8
        chunk = d[off:off+clen]; off += clen
        if ctype == 0x4E4F534A: js = json.loads(chunk)
        elif ctype == 0x004E4942: bina = chunk
    return js, bina


def accessor(js, bina, ai):
    a = js['accessors'][ai]; bv = js['bufferViews'][a['bufferView']]
    bo = bv.get('byteOffset',0)+a.get('byteOffset',0); fmt,sz = CT[a['componentType']]
    nc = NC[a['type']]; n = a['count']; stride = bv.get('byteStride') or sz*nc
    buf = np.frombuffer(bina, np.uint8); dt = np.dtype('<'+fmt); out = np.empty((n,nc))
    for c in range(nc):
        idx = bo + c*sz + np.arange(n)*stride
        out[:,c] = np.stack([buf[idx+k] for k in range(sz)],1).view(dt).ravel().astype(float)
    return out


def bv_bytes(js, bina, bvi):
    bv = js['bufferViews'][bvi]; o = bv.get('byteOffset',0); return bina[o:o+bv['byteLength']]


def rot_align(a, b):
    a = a/np.linalg.norm(a); b = b/np.linalg.norm(b); v = np.cross(a,b); c = a@b
    if np.linalg.norm(v) < 1e-9: return np.eye(3)
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+vx+vx@vx*(1/(1+c))


def write_glb(path, V, UV, F, img_bytes, mime):
    """Emit a minimal textured GLB: one mesh, one material, one texture."""
    V = V.astype('<f4'); UV = UV.astype('<f4'); F = F.astype('<u4')
    pos = V.tobytes(); uv = UV.tobytes(); idx = F.tobytes()
    def pad(b, fill=b'\x00'):
        r = (-len(b)) % 4
        return b + fill*r
    pos_p, uv_p, idx_p, img_p = pad(pos), pad(uv), pad(idx), pad(img_bytes)
    blob = pos_p + uv_p + idx_p + img_p
    o_pos = 0; o_uv = len(pos_p); o_idx = o_uv+len(uv_p); o_img = o_idx+len(idx_p)
    gltf = {
        "asset": {"version": "2.0", "generator": "AMS export_conveyor_glb.py"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                                    "indices": 2, "material": 0, "mode": 4}]}],
        "materials": [{"pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
        "textures": [{"sampler": 0, "source": 0}],
        "images": [{"bufferView": 3, "mimeType": mime}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [
            {"buffer":0,"byteOffset":o_pos,"byteLength":len(pos),"target":34962},
            {"buffer":0,"byteOffset":o_uv,"byteLength":len(uv),"target":34962},
            {"buffer":0,"byteOffset":o_idx,"byteLength":len(idx),"target":34963},
            {"buffer":0,"byteOffset":o_img,"byteLength":len(img_bytes)},
        ],
        "accessors": [
            {"bufferView":0,"componentType":5126,"count":len(V),"type":"VEC3",
             "min":V.min(0).tolist(),"max":V.max(0).tolist()},
            {"bufferView":1,"componentType":5126,"count":len(UV),"type":"VEC2"},
            {"bufferView":2,"componentType":5125,"count":len(F)*3,"type":"SCALAR"},
        ],
    }
    js = json.dumps(gltf, separators=(',',':')).encode()
    js = js + b' '*((-len(js)) % 4)
    total = 12 + 8+len(js) + 8+len(blob)
    with open(path,'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, total))
        f.write(struct.pack('<II', len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack('<II', len(blob), 0x004E4942)); f.write(blob)


def render_topdown(Vn, UVn, Fn, tex, mmpp=2.0):
    """Orthographic plan view of the belt: project every face onto the ground and
    paint its texture, highest-last so the top surface wins. Returns the image
    and its world extent (X0,X1,Z0,Z1)."""
    th, tw = tex.shape[:2]
    C = Vn[Fn].mean(1)
    lo = C[:, [0, 2]].min(0); hi = C[:, [0, 2]].max(0)
    W = max(1, int((hi[0]-lo[0]) * 1000 / mmpp)); H = max(1, int((hi[1]-lo[1]) * 1000 / mmpp))
    img = np.full((H, W, 3), 18, np.uint8)
    fuv = UVn[Fn].mean(1)
    tx = np.clip((fuv[:, 0]*tw).astype(int), 0, tw-1); ty = np.clip((fuv[:, 1]*th).astype(int), 0, th-1)
    col = tex[ty, tx]
    ix = (Vn[Fn][:, :, 0]-lo[0]) * 1000 / mmpp
    iz = (Vn[Fn][:, :, 2]-lo[1]) * 1000 / mmpp
    tri = np.stack([ix, iz], axis=2).astype(np.int32)
    for k in np.argsort(C[:, 1]):
        cv2.fillConvexPoly(img, tri[k], [int(x) for x in col[k]])
    return img, (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))


def write_flat_glb(path, extent, img_bgr):
    """A flat, level map: one horizontal quad at Y=0 spanning `extent`, textured
    with the top-down image. Guaranteed level — geometry has no height at all."""
    X0, X1, Z0, Z1 = extent
    V = np.array([[X0, 0, Z0], [X1, 0, Z0], [X1, 0, Z1], [X0, 0, Z1]], np.float32)
    UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    # winding chosen so the face normal points +Y (up) — the textured front faces
    # the overhead camera rather than its back.
    F = np.array([[0, 2, 1], [0, 3, 2]], np.uint32)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    write_glb(path, V, UV, F, buf.tobytes(), "image/jpeg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--origin-x", type=float, default=None,
                    help="world origin X in the leveled (pre-shift) frame; default = left board")
    ap.add_argument("--origin-z", type=float, default=None,
                    help="world origin Z in the leveled (pre-shift) frame; default = left board")
    ap.add_argument("--twist-deg", type=float, default=12.6,
                    help="extra in-plane rotation so the board line is axis-aligned "
                         "(default flattens the residual tilt; adjust by eye)")
    ap.add_argument("--ymin-mm", type=float, default=-70.0,
                    help="drop faces below this height above the belt (floor clutter)")
    ap.add_argument("--ymax-mm", type=float, default=120.0,
                    help="drop faces above this height above the belt (tall rails/frame)")
    ap.add_argument("--flat", action=argparse.BooleanOptionalAction, default=True,
                    help="export a FLAT textured plane (top-down projection at Y=0) — "
                         "guaranteed level, ideal for X-Y calibration. --no-flat keeps "
                         "the true 3D belt surface (with its trough).")
    ap.add_argument("--mmpp", type=float, default=2.0,
                    help="mm per pixel for the flat map's top-down texture")
    args = ap.parse_args()

    js, bina = load_glb(args.src)
    pr = js['meshes'][0]['primitives'][0]
    V = accessor(js, bina, pr['attributes']['POSITION'])
    UV = accessor(js, bina, pr['attributes']['TEXCOORD_0'])
    F = accessor(js, bina, pr['indices']).astype(int).reshape(-1,3)
    img_bytes = bv_bytes(js, bina, js['images'][0]['bufferView'])
    mime = js['images'][0].get('mimeType','image/jpeg')
    tex = cv2.imdecode(np.frombuffer(img_bytes,np.uint8), cv2.IMREAD_COLOR); th,tw = tex.shape[:2]
    C = V[F].mean(1)
    print(f"loaded: {len(V)} verts, {len(F)} faces, tex {tw}x{th}")

    # --- belt plane from the checkerboard seed region, then level ---
    seed = (C[:,0]>-1.6)&(C[:,0]<0.2)&(C[:,2]>0.55)&(C[:,2]<1.4)
    def fit(P): c=P.mean(0); _,_,vt=np.linalg.svd(P-c); n=vt[2]; return c,n/np.linalg.norm(n)
    c,n = fit(C[seed])
    for _ in range(4):
        d=np.abs((C-c)@n); c,n=fit(C[d<0.03])
    if n[1] < 0: n = -n
    R1 = rot_align(n, np.array([0,1.,0])); Vl = V@R1.T; Cl = Vl[F].mean(1)
    # Level the whole belt to horizontal, robust to UNEVEN scan sampling. The
    # right side is scanned densely and the left edge sparsely, so a plain fit is
    # out-voted by the right and leaves the left tilted up. Fix: bin the belt
    # surface into ground cells, take each cell's median height, and fit the
    # plane to the cell centres — equal weight per location, not per point — so
    # both ends and both edges come level regardless of point density.
    beltY0 = np.median(Cl[(np.abs((C-c)@n) < 0.025) & seed][:, 1])
    for _pass in range(2):
        bands = Cl[np.abs(Cl[:, 1] - beltY0) < 0.06]        # belt surface (no rails)
        cell = 0.05
        keys = np.round(bands[:, [0, 2]] / cell).astype(np.int64)
        acc = {}
        for k, y in zip(map(tuple, keys), bands[:, 1]):
            acc.setdefault(k, []).append(y)
        pts = np.array([[k[0]*cell, np.median(v), k[1]*cell]
                        for k, v in acc.items() if len(v) >= 3])
        A = np.c_[pts[:, 0], pts[:, 2], np.ones(len(pts))]
        coef = np.linalg.lstsq(A, pts[:, 1], rcond=None)[0]
        for _ in range(3):                                   # trim outlier cells
            r = pts[:, 1] - A @ coef; keep = np.abs(r) < 2 * r.std() + 1e-9
            A, pts = A[keep], pts[keep]
            coef = np.linalg.lstsq(A, pts[:, 1], rcond=None)[0]
        a, b = coef[0], coef[1]
        nn = np.array([-a, 1.0, -b]); nn /= np.linalg.norm(nn)
        Rlv = rot_align(nn, np.array([0, 1., 0])); Vl = Vl @ Rlv.T; Cl = Vl[F].mean(1)
        beltY0 = np.median(pts[:, 1])
    print(f"levelled (grid-binned, {len(pts)} cells): residual tilt "
          f"along {a*1000:+.1f}, across {b*1000:+.1f} mm/m")
    belt = (np.abs((C-c)@n)<0.025) & seed
    Bxz = Cl[belt][:,[0,2]]; m=Bxz.mean(0); _,_,vt=np.linalg.svd(Bxz-m); ax=vt[0]; ang=np.arctan2(ax[1],ax[0])
    ca,sa=np.cos(-ang),np.sin(-ang); R2=np.array([[ca,0,-sa],[0,1,0],[sa,0,ca]])
    Vl = Vl@R2.T; Cl = Vl[F].mean(1)
    # Extra in-plane twist (about the vertical), so the conveyor axis — the line
    # through the three checkerboards — lands exactly on a coordinate axis. The
    # PCA above leaves a residual tilt; this corrects it. Adjustable, because the
    # board line is read by eye and the operator may want to nudge it.
    tw = np.radians(args.twist_deg)
    ct, st = np.cos(tw), np.sin(tw); Rt = np.array([[ct,0,-st],[0,1,0],[st,0,ct]])
    Vl = Vl@Rt.T; Cl = Vl[F].mean(1)
    beltY = np.median(Cl[belt][:,1])
    print(f"leveled: belt normal was {np.round(n,3)}; belt Y median {beltY:.3f} "
          f"(spread {Cl[belt][:,1].std()*1000:.0f} mm)")

    # --- origin = centre of the 8x11 board (under basler_1) ---
    # Auto-detection is unreliable on this coarse-mesh texture, so the three board
    # centres were read from a leveled top-down render (leveled, pre-shift frame):
    #   left plain checker  ~ (0.26, -0.415)   <- default origin (assumed 8x11/basler_1)
    #   middle ChArUco      ~ (0.66, -0.485)
    #   right plain checker ~ (1.33, -0.655)
    # Override with --origin-x/--origin-z if basler_1's board is a different one;
    # or just set the origin by clicking the board in CalibrationHub's Map Config.
    BOARDS = {"left": (0.26, -0.415), "middle": (0.66, -0.485), "right": (1.33, -0.655)}
    # those were read in the pre-twist frame; rotate them by the same twist
    BOARDS = {k: (ct*x - st*z, st*x + ct*z) for k, (x, z) in BOARDS.items()}
    ox = args.origin_x if args.origin_x is not None else BOARDS["left"][0]
    oz = args.origin_z if args.origin_z is not None else BOARDS["left"][1]
    origin = np.array([ox, beltY, oz])
    print(f"origin (assumed 8x11/left board, leveled) = X={ox:.3f} Y={beltY:.3f} Z={oz:.3f} m")
    print("board offsets from this origin (X along belt, Z across), for Map Config:")
    for name,(bx,bz) in BOARDS.items():
        print(f"    {name:7s} ({bx-ox:+.3f}, {bz-oz:+.3f}) m")

    # translate origin to (0,0,0); keep belt at Y=0
    Vl = Vl - origin; Cl = Vl[F].mean(1)

    # --- crop to conveyor: belt block + margin (recompute belt extent post-shift) ---
    bx0,bx1 = Cl[belt][:,0].min()-args.margin, Cl[belt][:,0].max()+args.margin
    bz0,bz1 = Cl[belt][:,2].min()-args.margin, Cl[belt][:,2].max()+args.margin
    # Height band: the belt sits at Y~0 after the origin shift. Keep the belt
    # surface and the boards on it, but drop the tall side-rails/frame/rollers and
    # the floor below — that clutter is what made the map read as slanted even
    # though the belt is level.
    keep = ((Cl[:,0]>bx0)&(Cl[:,0]<bx1)&(Cl[:,2]>bz0)&(Cl[:,2]<bz1)
            & (Cl[:,1] > args.ymin_mm/1000.0) & (Cl[:,1] < args.ymax_mm/1000.0))
    Fk = F[keep]
    used = np.unique(Fk)
    remap = -np.ones(len(V),int); remap[used]=np.arange(len(used))
    Vn = Vl[used]; UVn = UV[used]; Fn = remap[Fk]
    print(f"cropped: {keep.sum()}/{len(F)} faces kept, {len(used)} verts")
    print(f"conveyor extent: X {Vn[:,0].min():.2f}..{Vn[:,0].max():.2f} "
          f"Z {Vn[:,2].min():.2f}..{Vn[:,2].max():.2f} m")

    if args.flat:
        top, extent = render_topdown(Vn, UVn, Fn, tex, mmpp=args.mmpp)
        write_flat_glb(args.out, extent, top)
        print(f"wrote FLAT map {args.out}  ({top.shape[1]}x{top.shape[0]} px @ {args.mmpp}mm/px, "
              f"level plane at Y=0)")
    else:
        write_glb(args.out, Vn, UVn, Fn, img_bytes, mime)
        print("wrote 3D-surface map", args.out)


if __name__ == "__main__":
    main()
