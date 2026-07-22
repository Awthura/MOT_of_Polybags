"""
Diagnose original blend file: list all objects, materials, and their Base Colors.
Run inside Blender with the original blend file.
"""
import bpy, sys

print("\n" + "="*60)
print("  ORIGINAL BLEND FILE DIAGNOSTIC")
print("="*60)

# List all materials and their Base Colors
print("\n--- Materials ---")
for mat in bpy.data.materials:
    if mat.use_nodes and mat.node_tree:
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            col = bsdf.inputs["Base Color"].default_value
            print(f"  {mat.name:40s}  RGBA=({col[0]:.3f}, {col[1]:.3f}, {col[2]:.3f}, {col[3]:.3f})")
        else:
            print(f"  {mat.name:40s}  (no Principled BSDF)")
    else:
        print(f"  {mat.name:40s}  (no nodes)")

# List all objects
print("\n--- Objects (first 30) ---")
for i, obj in enumerate(bpy.data.objects[:30]):
    mats = [m.name if m else "None" for m in obj.material_slots]
    print(f"  {obj.name:40s}  type={obj.type:8s}  mats={mats}")

# List particle systems
print("\n--- Objects with Particle Systems ---")
for obj in bpy.data.objects:
    for ps in obj.particle_systems:
        print(f"  {obj.name}.{ps.name}  type={ps.settings.type}  count={ps.settings.count}")

# List collections
print("\n--- Collections ---")
for col in bpy.data.collections:
    print(f"  {col.name}  objects={[o.name for o in col.objects][:10]}")

# Check frame range
scene = bpy.context.scene
print(f"\n--- Scene ---")
print(f"  Frame range: {scene.frame_start} – {scene.frame_end}")
print(f"  Render resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")

# Check for any mesh objects and their materials at frame 100
print("\n--- Mesh objects at frame 100 ---")
scene.frame_set(100)
bpy.context.view_layer.update()
mesh_count = 0
for obj in bpy.data.objects:
    if obj.type == "MESH" and obj.visible_get():
        mats = [m.name if m else "None" for m in obj.material_slots]
        loc = obj.matrix_world.translation
        print(f"  {obj.name:40s}  loc=({loc.x:.3f},{loc.y:.3f},{loc.z:.3f})  mats={mats}")
        mesh_count += 1
        if mesh_count >= 20:
            print("  ... (truncated)")
            break

print("\n" + "="*60)
