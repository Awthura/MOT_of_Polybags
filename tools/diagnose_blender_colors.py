"""
Diagnostic script: find how the original render assigned colors to polybags.
Run inside Blender:
  /Applications/Blender.app/Contents/MacOS/Blender \
    /Users/awthura/OVGU/AMS/convert_stl_to_animation_multi_camera.blend \
    --background --python diagnose_blender_colors.py

Prints:
  1. All materials already in the scene and their base colors
  2. For frame 100: imports STL WITHOUT overriding materials → what color do parts get?
  3. Objects already present in the scene at frame 100
"""

import bpy
import sys
from pathlib import Path
from mathutils import Vector

BASE       = Path("/Users/awthura/OVGU/AMS")
STL_FOLDER = BASE / "superquadrics_stl_files_100_2000_frames"

def material_color(mat):
    """Return (R,G,B) or None from a material's Principled BSDF."""
    if mat is None or not mat.use_nodes:
        return None
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            c = node.inputs["Base Color"].default_value
            return (round(c[0],3), round(c[1],3), round(c[2],3))
    return None

# ── 1. List existing materials in the scene ───────────────────────────────────
print("\n" + "="*65)
print("EXISTING MATERIALS IN SCENE:")
for mat in bpy.data.materials:
    col = material_color(mat)
    print(f"  {mat.name:40s}  RGB={col}")

# ── 2. List existing mesh objects and their materials ────────────────────────
print("\n" + "="*65)
print("MESH OBJECTS IN SCENE (first 30):")
for obj in list(bpy.data.objects)[:30]:
    if obj.type == "MESH":
        mats = [m.name if m else "None" for m in obj.data.materials]
        print(f"  {obj.name:40s}  mats={mats}")

# ── 3. Check particle systems ─────────────────────────────────────────────────
print("\n" + "="*65)
print("OBJECTS WITH PARTICLE SYSTEMS:")
for obj in bpy.data.objects:
    for ps in obj.particle_systems:
        print(f"  {obj.name:30s} → particle system '{ps.name}'")
        pset = ps.settings
        print(f"    render_type={pset.render_type}")
        if pset.material_slot:
            print(f"    material_slot={pset.material_slot}")

# ── 4. Import STL for frame 100 WITHOUT material override ─────────────────────
print("\n" + "="*65)
print("STL IMPORT (NO MATERIAL OVERRIDE) — FRAME 100:")

frame = 100
candidates = [
    STL_FOLDER / f"ExtractSurface1_frame_{frame:04d}.stl",
    STL_FOLDER / f"dump_plane1stl_frame_{frame:04d}.stl",
    STL_FOLDER / f"Triangulate1_frame_{frame:04d}.stl",
]
stl_path = next((p for p in candidates if p.exists()), None)
if stl_path:
    print(f"  Using: {stl_path.name}")
    bpy.ops.object.select_all(action="DESELECT")
    try:
        bpy.ops.wm.stl_import(filepath=str(stl_path))
    except AttributeError:
        bpy.ops.import_mesh.stl(filepath=str(stl_path))

    imported = bpy.context.selected_objects[0]
    imported.name = "STL_diag"
    bpy.ops.object.select_all(action="DESELECT")
    imported.select_set(True)
    bpy.context.view_layer.objects.active = imported
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")

    parts = bpy.context.selected_objects
    print(f"  Separated into {len(parts)} parts")
    for i, part in enumerate(parts):
        mats = [m.name if m else "None" for m in part.data.materials]
        colors = [material_color(m) for m in part.data.materials if m]
        vcount = len(part.data.vertices)
        mat_w = part.matrix_world
        verts_w = [mat_w @ v.co for v in part.data.vertices]
        centroid = sum((Vector(v) for v in verts_w), Vector()) / len(verts_w)
        print(f"  Part {i:2d}: verts={vcount:5d}  centroid=({centroid.x:.3f},{centroid.y:.3f},{centroid.z:.3f})"
              f"  mats={mats}  colors={colors}")
else:
    print("  STL not found for frame 100")

# ── 5. Geometry Nodes check ───────────────────────────────────────────────────
print("\n" + "="*65)
print("GEOMETRY NODES MODIFIERS:")
for obj in bpy.data.objects:
    for mod in obj.modifiers:
        if mod.type == "NODES":
            print(f"  {obj.name:30s} → GN modifier '{mod.name}'")
            if mod.node_group:
                print(f"    node_group='{mod.node_group.name}'")
                for node in mod.node_group.nodes:
                    print(f"      node: {node.type} '{node.name}'")

print("\n" + "="*65)
print("DONE. Check above for color assignment clues.")
