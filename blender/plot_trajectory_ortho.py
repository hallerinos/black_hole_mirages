from bpy import context, data, ops
from mathutils import Vector
import pandas as pd
import numpy as np
import argparse

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

    # for m in data.materials:
    #     m.user_clear()
    #     data.materials.remove(m)

    # Select all objects
    for o in context.scene.objects:
        o.select_set(True)

    # Delete all objects in scene
    ops.object.delete()


def create_trajectory(df, args):
    # Create a bezier circle and enter edit mode.
    ops.curve.primitive_bezier_curve_add(radius=1,
                                         location=(0.0, 0.0, 0.0),
                                         enter_editmode=True)

    # Subdivide the trajectory by a number of cuts, giving the
    # random vertex function more points to work with.
    ops.curve.subdivide(number_cuts=len(df)-3)
    trajectory = context.active_object
    trajectory.name = 'trajectory'

    # Locate the array of bezier points.
    bez_pts = trajectory.data.splines[0].bezier_points
    for (id, b) in enumerate(bez_pts):
        d0 = df.iloc[id+1][['x','y','z_s']]
        dl = df.iloc[id][['x','y','z_s']]
        
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
    obj_data.bevel_depth = 0.04

    # Smoothness of the segments on the curve.
    obj_data.resolution_u = 64
    obj_data.render_resolution_u = 64
    
    return (trajectory, df)

def get_args():
    parser = argparse.ArgumentParser()

    # get all script args
    _, all_arguments = parser.parse_known_args()
    double_dash_index = all_arguments.index('--')
    script_args = all_arguments[double_dash_index + 1: ]

    # add parser rules
    parser.add_argument('-csvfile', help="name of csv file")
    parser.add_argument('-save', help="output file")
    parser.add_argument('-scale', help="resolution rezie")
    parser.add_argument('-zscale', help="rescale z axis")
    parser.add_argument('-zview', help="rescale z axis")
    parsed_script_args, _ = parser.parse_known_args(script_args)
    return parsed_script_args

args = get_args()
print(args)
# exit()
delete_scene()

# Set black background
data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
# Set transparent
scene = context.scene
scene.render.film_transparent = True
scene.render.engine = 'CYCLES'
standard_res = 1080
scene.render.resolution_x = standard_res
scene.render.resolution_y = standard_res
scene.cycles.samples = 64

if args.scale:
    scene.render.resolution_x = int(standard_res*float(args.scale))
    scene.render.resolution_y = int(standard_res*float(args.scale))

df = pd.read_csv(args.csvfile)
# eventually scale coordinates
if args.zscale:
    df['z_s'] = df['z']*float(args.zscale)
else:
    df['z_s'] = df['z']
chis = np.unique(df['\\chi'])
trajectories = [create_trajectory(df[df['\\chi']==chi], args) for chi in chis]

r0, x0, y0, z0 = 1, 0, 0, 0
# Create a bezier circle and enter edit mode.
ops.mesh.primitive_cylinder_add(vertices=128, radius=1.5*r0, location=(x0, y0, z0), scale=(1,1,4))
photon_cylinder = context.active_object
photon_cylinder.name = 'photon_cylinder'
ops.object.shade_smooth()

ops.mesh.primitive_cylinder_add(vertices=128, radius=1*r0, location=(x0, y0, z0), scale=(1,1,4.01))
ops.object.shade_smooth()
horizon = context.active_object
horizon.name = 'horizon'

emission_strength_trajectories = 0

# Create trajectory material
mat_t1 = data.materials.new(name="t1")
trajectories[0][0].data.materials.append(mat_t1)
mat_t1.use_nodes = True
bsdf = mat_t1.node_tree.nodes["Principled BSDF"]
bsdf.inputs[0].default_value = (1, 0, 1, 1)
bsdf.inputs[7].default_value = 0
bsdf.inputs[19].default_value = (1, 0, 1, 1)
bsdf.inputs[20].default_value = emission_strength_trajectories
trajectory_materials = [mat_t1]*len(trajectories)

## Create trajectory material
mat_t2 = data.materials.new(name="t2")
trajectories[1][0].data.materials.append(mat_t2)
mat_t2.use_nodes = True
bsdf = mat_t2.node_tree.nodes["Principled BSDF"]
bsdf.inputs[0].default_value = (0., 1, 0.04, 1)
bsdf.inputs[7].default_value = 0
bsdf.inputs[19].default_value = (0., 1, 0.04, 1)
bsdf.inputs[20].default_value = emission_strength_trajectories
trajectory_materials[1] = mat_t2

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
volume.inputs[2].default_value = 0.3
volume.inputs[0].default_value = (0, 0, 0, 1)

bsdf.inputs[0].default_value = (cv,cv,cv, 1)  # Base color

mat.node_tree.links.new(volume.outputs["Volume"], mat_out.inputs["Volume"])

ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(x0, y0, z0), scale=(1, 1, 1))

cam_loc = (0,0,int(args.zview))
ops.object.camera_add(enter_editmode=False, align='VIEW', location=cam_loc, rotation=(0, 0, 0), scale=(1, 1, 1))
cam = data.objects["Camera"]
cam.constraints.new(type='TRACK_TO')
cam.constraints['Track To'].target = data.objects["Empty"]
cam.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
cam.constraints["Track To"].up_axis = 'UP_Y'
cam.data.type = 'ORTHO'
scene.camera = context.object

s = 130  # for the lensing
s = 112  # for the unstable orbit
cone_locations = [t[1].iloc[s][['x', 'y', 'z_s']] for t in trajectories]
cone_directions = [t[1].iloc[s+1][['x', 'y', 'z_s']]-t[1].iloc[s][['x', 'y', 'z_s']] for t in trajectories]
for (cl, cd, mat) in zip(cone_locations, cone_directions, trajectory_materials):
    ops.mesh.primitive_cone_add(radius1=0.1, radius2=0, depth=0.4, enter_editmode=False, align='WORLD', location=cl, scale=(1, 1, 1))
    cone = context.active_object
    #cone.rotation_euler[0] = -1.48311
    cone.data.materials.append(mat)
    cone.rotation_mode = 'QUATERNION'
    cone.rotation_quaternion = Vector(cd).to_track_quat('Z','Y')

if args.save:
    # export scene
    frame = 1
    context.scene.frame_set(frame)
    context.scene.render.filepath = args.save
    ops.render.render(write_still=True)
