from bpy import data, context, ops

def delete_scene():
    # Will collect meshes from delete objects
    meshes, curves = set(), set()
    # Iterate over all collections
    for collection in data.collections:
        # Get objects in the collection if they are meshes
        for obj in [o for o in collection.objects if o.type == 'MESH']:
            # Store the internal mesh
            meshes.add(obj.data)
            # Delete the object
            data.objects.remove(obj)
        # Look at meshes that are orphean after objects removal
        for mesh in [m for m in meshes if m.users == 0]:
            # Delete the meshes
            data.meshes.remove(mesh)
        # Get objects in the collection if they are meshes
        for obj in [o for o in collection.objects if o.type == 'CURVE']:
            # Store the internal mesh
            curves.add(obj.data)
            # Delete the object
            data.objects.remove(obj)
        # Look at meshes that are orphean after objects removal
        for curve in [c for c in curves if c.users == 0]:
            # Delete the meshes
            data.curves.remove(curve)

    for m in data.materials:
        m.user_clear()
        data.materials.remove(m)

    # Select all objects
    for o in context.scene.objects:
        o.select_set(True)

    # Delete all objects in scene
    ops.object.delete()