from bpy import context, data, ops
from mathutils import Euler, Matrix, Quaternion, Vector
import csv
import numpy as np
import os, sys
import numpy as np

file_path = os.path.realpath(__file__).replace('trajectories.blend/plot_separatrix.py','')

def create_trajectory(csvfile):
    # Create a bezier circle and enter edit mode.
    ops.curve.primitive_bezier_curve_add(radius=1,
                                        location=(0.0, 0.0, 0.0),
                                        enter_editmode=True)

    # Import the trajectory coordinates
    with open(csvfile, newline='') as inputfile:
        coordinates = list(csv.reader(inputfile))
    header = coordinates[0]
    coordinates = np.asfarray(coordinates[1:])
    print(header)
    print(len(coordinates))

    # Subdivide the trajectory by a number of cuts, giving the
    # random vertex function more points to work with.
    ops.curve.subdivide(number_cuts=len(coordinates)-3)
    trajectory = context.active_object
    trajectory.name = 'trajectory'

    # Locate the array of bezier points.
    bez_pts = trajectory.data.splines[0].bezier_points
    for (id, b) in enumerate(bez_pts):
        d0 = coordinates[id+1]
        dl = coordinates[id]
        # dr = coordinates[id+2]
        # print(d0)
        b.co = Vector(d0)
        b.handle_left = Vector(d0+0.25*(dl-d0))
        b.handle_right  = Vector(d0-0.25*(dl-d0))

    # Scale the curve while in edit mode.
    #ops.transform.resize(value=(2.0, 2.0, 3.0))

    # Return to object mode.
    ops.object.mode_set(mode='OBJECT')

    # Store a shortcut to the curve object's data.
    obj_data = context.active_object.data

    # Which parts of the curve to extrude ['HALF', 'FRONT', 'BACK', 'FULL'].
    obj_data.fill_mode = 'FULL'

    # Breadth of extrusion.
    obj_data.extrude = 0.0

    # Depth of extrusion.
    obj_data.bevel_depth = 0.03

    # Smoothness of the segments on the curve.
    obj_data.resolution_u = 64
    obj_data.render_resolution_u = 64
    
    return trajectory

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

# Set black background
data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
# Set transparent
context.scene.render.film_transparent = True

trajectories = []
trajectories.append(create_trajectory(f'{file_path}/../codes/photon_cylinder_orbit_1.csv'))
trajectories.append(create_trajectory(f'{file_path}/../codes/photon_cylinder_orbit_2.csv'))

r0, x0, y0, z0 = 1, 0, 0, 0
# Create a bezier circle and enter edit mode.
ops.mesh.primitive_cylinder_add(vertices=128, radius=1.5*r0, location=(x0, y0, z0), scale=(1,1,3))
photon_cylinder = context.active_object
photon_cylinder.name = 'photon_cylinder'
ops.object.shade_smooth()

ops.mesh.primitive_cylinder_add(vertices=128, radius=1*r0, location=(x0, y0, z0), scale=(1,1,3.01))
ops.object.shade_smooth()
horizon = context.active_object
horizon.name = 'horizon'

emission_strength_trajectories = 0
# Create trajectory material
#mat_t1 = data.materials.new(name="t1")
#trajectories[1].data.materials.append(mat_t1)
#mat_t1.use_nodes = True
#bsdf = mat_t1.node_tree.nodes["Principled BSDF"]
#bsdf.inputs[0].default_value = (1, 0, 1, 1)
#bsdf.inputs[7].default_value = 0
#bsdf.inputs[19].default_value = (1, 0, 1, 1)
#bsdf.inputs[20].default_value = emission_strength_trajectories

# Create trajectory material
mat_t1 = data.materials.new(name="t1")
trajectories[0].data.materials.append(mat_t1)
mat_t1.use_nodes = True
bsdf = mat_t1.node_tree.nodes["Principled BSDF"]
#bsdf.inputs[0].default_value = (0, 1, 1, 1)
#bsdf.inputs[7].default_value = 0
#bsdf.inputs[19].default_value = (0, 1, 1, 1)
bsdf.inputs[0].default_value = (1, 0, 1, 1)
bsdf.inputs[7].default_value = 0
bsdf.inputs[19].default_value = (1, 0, 1, 1)
bsdf.inputs[20].default_value = emission_strength_trajectories

# Create trajectory material
mat_t2 = data.materials.new(name="t2")
trajectories[1].data.materials.append(mat_t2)
mat_t2.use_nodes = True
bsdf = mat_t2.node_tree.nodes["Principled BSDF"]
bsdf.inputs[0].default_value = (0., 1, 0.04, 1)
bsdf.inputs[7].default_value = 0
bsdf.inputs[19].default_value = (0., 1, 0.04, 1)
bsdf.inputs[20].default_value = emission_strength_trajectories

# Create photon sphere material
mat = data.materials.new(name="photon_cylinder")
photon_cylinder.data.materials.append(mat)
mat.use_nodes = True
mat_out = mat.node_tree.nodes["Material Output"]  # Output node
bsdf = mat.node_tree.nodes["Principled BSDF"]  # Standard surface node
cv = 0.004
bsdf.inputs[0].default_value = (cv,cv,cv, 1)  # Base color
bsdf.inputs[6].default_value = 0.0  # Metallic
bsdf.inputs[7].default_value = 0.0  # Specular
bsdf.inputs[9].default_value = 0  # Roughness
bsdf.inputs[16].default_value = 1.1  # IOR
bsdf.inputs[17].default_value = 1  # Transmission
bsdf.inputs[18].default_value = 0.2  # Transmission roughness
bsdf.inputs[21].default_value = 0  # Alpha
volume = mat.node_tree.nodes.new("ShaderNodeVolumePrincipled")  # Add volume node
volume.location = (-500,200)
volume.inputs[2].default_value = 0.1
volume.inputs[2].default_value = 0.5
volume.inputs[0].default_value = (0, 0, 0, 1)
mat.node_tree.links.new(volume.outputs["Volume"], mat_out.inputs["Volume"])

# Create horizon material
mat = data.materials.new(name="horizon")
horizon.data.materials.append(mat)
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
mat_out = mat.node_tree.nodes["Material Output"]  # Output node
cv = 0.001
bsdf.inputs[0].default_value = (cv,cv,cv, 1)  # Base color
bsdf.inputs[6].default_value = 0.3  # Metallic
bsdf.inputs[7].default_value = 0.2  # Specular
bsdf.inputs[9].default_value = 0.2  # Roughness
bsdf.inputs[16].default_value = 0  # IOR
bsdf.inputs[17].default_value = 0  # Transmission
bsdf.inputs[18].default_value = 0.0  # Transmission roughness
bsdf.inputs[21].default_value = 0  # Alpha
volume = mat.node_tree.nodes.new("ShaderNodeVolumePrincipled")  # Add volume node
volume.location = (-500,200)
volume.inputs[2].default_value = 0.2
volume.inputs[2].default_value = 1
volume.inputs[0].default_value = (0, 0, 0, 1)

bsdf.inputs[0].default_value = (cv,cv,cv, 1)  # Base color

mat.node_tree.links.new(volume.outputs["Volume"], mat_out.inputs["Volume"])

#bsdf.inputs[19].default_value = (0, 0, 0, 1)

#bsdf.inputs[19].default_value = (0.00975761, 0.647787, 1, 1)
#bsdf.inputs[20].default_value = 0

ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(x0, y0, z0), scale=(1, 1, 1))


cam_loc = (9,0,0)
#cam_loc = (0,0,9)
#cam_loc = (0,0,-9)
ops.object.camera_add(enter_editmode=False, align='VIEW', location=cam_loc, rotation=(0, 0, 0), scale=(1, 1, 1))
cam = data.objects["Camera"]
cam.constraints.new(type='TRACK_TO')
cam.constraints['Track To'].target = data.objects["Empty"]
cam.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
cam.constraints["Track To"].up_axis = 'UP_Y'
cam.data.lens = 42
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 6.7
#context.object.data.dof.use_dof = True
#context.object.data.dof.focus_object = data.objects["Empty"]
#context.object.data.dof.aperture_fstop = 4

i=0
cones_up = []
cones_dn = []
for i in range(3):
    ops.mesh.primitive_cone_add(radius1=0.1, radius2=0, depth=0.4, enter_editmode=False, align='WORLD', location=(1.5, 0, (i+1)*0.627854), scale=(1, 1, 1))
    context.object.rotation_euler[0] = -1.48311
    cones_up.append(context.active_object)
    ops.mesh.primitive_cone_add(radius1=0.1, radius2=0, depth=0.4, enter_editmode=False, align='WORLD', location=(1.5, 0, -(i+1)*0.627854), scale=(1, 1, 1))
    context.object.rotation_euler[0] = 1.48311 - np.pi
    cones_dn.append(context.active_object)
for (cu,cd) in zip(cones_up,cones_dn):
    cu.data.materials.append(mat_t2)
    cd.data.materials.append(mat_t1)

#light_intensity = 200
#light_loc = Vector(np.asarray(cam_loc) + np.asarray([0, -5, 2]))
#ops.object.light_add(type='SPOT', align='WORLD', location=light_loc, scale=(1,1,1))
#spot = data.objects["Spot"]
#spot.data.spot_size = np.pi/2
#spot.data.energy = light_intensity
#spot.data.color = (1, 1, 1)
#spot.constraints.new(type='TRACK_TO')
#spot.constraints['Track To'].target = data.objects["Empty"]
#spot.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
#spot.constraints["Track To"].up_axis = 'UP_Y'

#light_loc = Vector(np.asarray(cam_loc) + np.asarray([0, 5, 2]))
#ops.object.light_add(type='SPOT', align='WORLD', location=light_loc, scale=(1,1,1))
#spot = context.active_object
#spot.data.spot_size = np.pi/2
#spot.data.energy = light_intensity
#spot.data.color = (1, 1, 1)
#spot.constraints.new(type='TRACK_TO')
#spot.constraints['Track To'].target = data.objects["Empty"]
#spot.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
#spot.constraints["Track To"].up_axis = 'UP_Y'

#light_loc = Vector(0*np.asarray(cam_loc) + np.asarray([0, 0, 10]))
#ops.object.light_add(type='SPOT', align='WORLD', location=light_loc, scale=(1,1,1))
#spot = context.active_object
#spot.data.spot_size = np.pi/2
#spot.data.energy = light_intensity
#spot.data.color = (1, 1, 1)
#spot.constraints.new(type='TRACK_TO')
#spot.constraints['Track To'].target = data.objects["Empty"]
#spot.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
#spot.constraints["Track To"].up_axis = 'UP_Y'
#ops.object.light_add(type='SPOT', align='WORLD', location=(x0+10, y0, z0+10*r0), scale=(1,1,1))
#context.object.data.spot_size = np.pi/2
#context.object.data.energy = light_intensity
#context.object.data.color = (1, 0, 0)
##context.object.rotation_euler[0] = np.pi
#context.object.rotation_euler[1] = 0.313423

#ops.object.light_add(type='SPOT', align='WORLD', location=(x0, y0+10, z0+10*r0), scale=(1,1,1))
#context.object.data.spot_size = np.pi/2
#context.object.data.energy = light_intensity
#context.object.data.color = (0, 0, 1)
##context.object.rotation_euler[0] = np.pi
#context.object.rotation_euler[0] = -0.313423


#ops.object.light_add(type='SPOT', align='WORLD', location=(x0, y0+10, z0-10*r0), scale=(1,1,1))
#context.object.data.spot_size = np.pi/2
#context.object.data.energy = light_intensity
#context.object.data.color = (0.9, 1, 0)
#context.object.rotation_euler[0] = np.pi+0.313423
##context.object.rotation_euler[0] = -0.313423

#ops.object.light_add(type='SPOT', align='WORLD', location=(-3, -1.45, -30), scale=(1,1,1))
#context.object.data.spot_size = np.pi/2
#context.object.data.energy = light_intensity
#context.object.data.color = (0.9, 1, 0.4)
#context.object.rotation_euler[0] = 2.95
#context.object.rotation_euler[1] = 0.32
#context.object.rotation_euler[2] = 0.44
