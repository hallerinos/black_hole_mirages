from unittest import case
import kwant
import kwant.continuum

def discretize_model(continuum_hamiltonian):
    tb_matrix_elements, coords = kwant.continuum.discretize_symbolic(continuum_hamiltonian)
    weyl_model = kwant.continuum.build_discretized(tb_matrix_elements, coords)
    return weyl_model

def cylinder_shape(site, R=10):
    x, y = site.pos  # position in integer lattice coordinates
    x = x-R
    return ((x**2 + y**2) < R**2)  # finite cylinder height

def cuboid_shape(site, R=10, Ry=10):
    x, y = abs(site.pos)  # position in integer lattice coordinates
    return ((x < R) and (y < Ry)) 

def tilt(x, y, tilt_frac=1, R=10, v_abs=0):
    if abs(x) < tilt_frac*R and abs(y) < tilt_frac*R:
        return v_abs
    else:
        return 0

def lead_shape(site, lead_frac=1, R=10, Ry=10, direction='ex'):
    x, y = abs(site.pos)
    if direction=='ex':
        return (y < lead_frac*Ry)
    elif direction=='ey':
        return (x < lead_frac*R)
    else: 
        return False

def create_leads(system, model, lead_shape, lead_fracs=[1, 1, 1, 1], R=10, Ry=10):
    # lead order: left right bottom top
    trans_symm = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    # directions = ['ex', 'ex', 'ey', 'ey']
    directions = ['ex', 'ex', 'ey', 'ey']
    s_point = [(-R+1,0), (Ry-1,0), (0,-Ry+1), (0,R-1)]

    electrodes = [kwant.Builder(kwant.TranslationalSymmetry(ts)) for ts in trans_symm]
    for (e,lf,dir,sp) in zip(electrodes, lead_fracs, directions,s_point):
        e.fill(model, lambda site: lead_shape(site, lead_frac=lf, R=R, Ry=Ry, direction=dir), sp)
        system.attach_lead(e)