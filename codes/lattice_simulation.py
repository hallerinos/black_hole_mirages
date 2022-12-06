import os, shutil, sys
from turtle import position
import pandas as pd
import numpy as np
import json
import h5py
import uuid
from copy import copy, deepcopy
from mystreamplot import mycurrent

from scipy import interpolate

import kwant

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = (0.5*(3+3/8),(3+3/8))
plt.rc('text.latex', preamble=r'\usepackage{bm}')

sim_id = str(uuid.uuid1())
path_out = f'weyl_plots/{sim_id}'
if os.path.exists(path_out):
    shutil.rmtree(path_out)
os.makedirs(path_out)

# if no argsys is given, create a config
if len(sys.argv)<2:
    # system parameters
    L = 300
    params = {'Ls': [L, L], 'k0': 1, 'energy': 0.8, 'r0': L//5, 'pos0': [-L//2, int(0.5*L)], 'alpha': 1, 'fd': -0.3, 'fu': -0.9}
    params = {'Ls': [L, L], 'k0': 1, 'energy': 0.8, 'r0': L//5, 'pos0': [-L//2, int(1*L)], 'alpha': 1, 'fd': 0.8, 'fu': 0.4}
    params = {'Ls': [L, L], 'k0': 1, 'energy': 0.8, 'r0': L//5, 'pos0': [-L//2, int(-1000000*L)], 'alpha': 1, 'fd': 0.19, 'fu': -0.21}
else:
    with open(sys.argv[1]) as file:
        params = json.load(file)

# useful definitions for later
sigma_0 = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
sigmas = [sigma_0, sigma_x, sigma_y, sigma_z]

# tilt profile
def tilt(pos, **kwargs):
    pos0 = np.asarray(kwargs.get('pos0', [0,0]))
    # length scale of the tilt
    r0 = kwargs.get('r0', 5)
    # exponent of the power-law
    alpha = kwargs.get('alpha', 1.0)
    
    if r0 == 0:
        return np.asarray([0,0])
    
    rvec = pos - pos0
    rabs = np.linalg.norm(rvec)
    return (r0/rabs)**alpha * rvec/rabs

# true if rvec is inside a Lx * Ly rectangle centered at (0,0)
def is_in_rectangle(pos, **kwargs):
        Ls = (kwargs.get('Ls', (10,5)))
        rvec = pos - np.asarray(kwargs.get('center_site', [0,0]))
        return all([-r < p < r for (p, r) in zip(rvec, Ls)])

# constructor for the scattering region and the leads
def make_system(sigmas, **kwargs):
    # default lattice: simple cubic
    primitive_vectors = kwargs.get('primitive_vectors', [np.asarray([1,0]), np.asarray([0,1])])
    # region function to populate lattice
    def is_in_region(pos, start=[0,0]):
        return is_in_rectangle(pos, **kwargs)
    # starting position to fill the lattice
    center_site = kwargs.get('center_site', [0,0])
    # mass gap for the superficial Weyl nodes at the boundary BZ
    k0 = kwargs.get('k0', 1)
    # if sites are passed, hoppings are only set on the sites
    sites = kwargs.get('sites', [])

    tilt_function = lambda pos : tilt(pos, **kwargs)
    
    # create the lattice
    lat = kwant.lattice.Monatomic(primitive_vectors, norbs=2)

    # sym specifies the semi-infinite direction of a lead
    sym = kwargs.get('sym', [])
    if sym != []:
        sym_lead = kwant.TranslationalSymmetry(sym)
        syst = kwant.Builder(sym_lead)
    else:
        syst = kwant.Builder()

    # on-site hopping elements
    if sites == []:
        syst[lat.shape(is_in_region, center_site)] = -2/k0*sigmas[3]
    else:
        for s in sites:
            syst[lat(*s.pos)] = -2/k0*sigmas[3]
            # quickfix create leads
            for pv in primitive_vectors:
                syst[lat(*(s.pos+sym))] = -2/k0*sigmas[3]

    # nearest neighbor terms
    nn = [0.5/k0*sigmas[3] - 0.5j*1*sigmas[1], 0.5/k0*sigmas[3] - 0.5j*1*sigmas[2]]
    # space-dependent tilt, taken in the center between sites
    def f(site1, site2, i):
        s1p, s2p = site1.pos, site2.pos
        (x,y) = 0.5*(s1p+s2p)
        return nn[i] - 0.5j*sigmas[0]*tilt_function((x,y))[i]
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = lambda s1,s2 : f(s1, s2, 0)
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = lambda s1,s2 : f(s1, s2, 1)
    return syst

syst = kwant.Builder()
syst = make_system(sigmas, **params)

# temporary finalized system, will be overwritten later
fsyst = syst.finalized()

sites = fsyst.sites  # get all sites
positions = np.asarray([s.pos for s in sites])
xmin, xmax = np.min(positions[:,0]), np.max(positions[:,0])
ymin, ymax = np.min(positions[:,1]), np.max(positions[:,1])

# select all edge sites of the system
xmin_sites = np.where(positions[:,0] == xmin)[0]
xmax_sites = np.where(positions[:,0] == xmax)[0]
ymin_sites = np.where(positions[:,1] == ymin)[0]
ymax_sites = np.where(positions[:,1] == ymax)[0]
bottom_sites_1 = np.where(positions[:,1] < int(ymin*params['fu']))[0]
bottom_sites_2 = np.where(positions[:,1] < int(ymin*params['fd']))[0]
bottom_sites_intersection = np.intersect1d(bottom_sites_1, bottom_sites_2)
bottom_sites_difference = np.setdiff1d(bottom_sites_1, bottom_sites_2)

lead_sites, lead_syms = [], []
tp0 = np.intersect1d(xmin_sites, bottom_sites_difference)
if len(tp0)>0:
    lead_sites.append(tp0)
    lead_syms.append([-1, 0])
tp1 = np.intersect1d(xmin_sites, bottom_sites_intersection)
if len(tp1)>0:
    lead_sites.append(tp1)
    lead_syms.append([-1, 0])
tp2 = np.setdiff1d(np.setdiff1d(xmin_sites, tp0), tp1)
if len(tp2)>0:
    lead_sites.append(tp2)
    lead_syms.append([-1, 0])

lead_sites.append(xmax_sites)
lead_syms.append([1, 0])
lead_sites.append(ymin_sites)
lead_syms.append([0, -1])
lead_sites.append(ymax_sites)
lead_syms.append([0, 1])

# this order is determined by the order of lead_sites
lead_centers = [(np.mean(positions[ls,0]), np.mean(positions[ls,1])) for ls in lead_sites]

# # visualize it, to verify
# fig, ax = plt.subplots(1,1)
# ax.scatter(positions[:,0], positions[:,1], color='gray', s=1)
# for ls in lead_sites:
#     ax.scatter(positions[ls,0], positions[ls,1], s=1)
# for (lc,ls) in zip(lead_centers,lead_syms):
#     ax.scatter(lc[0], lc[1], marker='x', s=1, color='black')
#     ax.quiver(lc[0], lc[1], ls[0], ls[1])
# ax.set_xlim(1.15*np.asarray((xmin,xmax)))
# ax.set_ylim(1.3*np.asarray((ymin,ymax)));
# ax.set_aspect('equal')
# ax.set_xlabel('$x/a$')
# ax.set_ylabel('$y/a$')
# plt.savefig(f'{path_out}/system_leads.png', dpi=1200, bbox_inches='tight', pad_inches=0.1)
# plt.close()

syst_tp = [deepcopy(syst) for _ in range(len(lead_syms))]
for (id, (ls,lc,s)) in enumerate(zip(lead_syms, lead_centers, lead_sites)):
    tp = [sites[id] for id in s]
    lead = make_system(sigmas, sym=ls, center_site=lc, sites=tp, r0=0)
    syst.attach_lead(lead)
    syst_tp[id].attach_lead(lead)
    # kwant.plotter.bands(lead.finalized())
fsyst = syst.finalized()
fsyst_tps = [stp.finalized() for stp in syst_tp]

fig, ax = plt.subplots(1,1)

cmap = plt.get_cmap('viridis')
# kwant.plot(fsyst, ax=ax, lead_color=cmap(0), site_color='black', lead_site_symbol=('p', 3, 0))
kwant.plot(fsyst_tps[0], ax=ax, lead_color=cmap(0.5), site_color='black', lead_site_symbol=('p', 3, -90))
kwant.plot(fsyst_tps[1], ax=ax, lead_color=cmap(0.0), site_color='black', lead_site_symbol=('p', 3, 90))
kwant.plot(fsyst_tps[2], ax=ax, lead_color=cmap(0.0), site_color='black', lead_site_symbol=('p', 3, 90))
kwant.plot(fsyst_tps[3], ax=ax, lead_color=cmap(0.0), site_color='black', lead_site_symbol=('p', 3, -90))
kwant.plot(fsyst_tps[4], ax=ax, lead_color=cmap(0.0), site_color='black', lead_site_symbol=('p', 3, 180))
kwant.plot(fsyst_tps[5], ax=ax, lead_color=cmap(0.0), site_color='black', lead_site_symbol=('p', 3, 0))
ax.set_aspect('equal')
ax.set_xlabel('$x/a$')
ax.set_ylabel('$y/a$')
ax.axis('off')
plt.savefig(f'{path_out}/system.png', dpi=2400, bbox_inches='tight', pad_inches=0.1)
plt.close()

print('-'*80, '\nsystem finalized.')
# kwant.plot(fsyst);
sites = fsyst.sites
positions = np.asarray([s.pos for s in sites])

# get the scattering states
print('-'*80, '\ncompute scattering states')
# propagating_modes, _ = fsyst.leads[0].modes(energy=energy)
scattering_states = kwant.wave_function(fsyst, energy=params['energy'])
ss = [scattering_states(i) for i in range(len(fsyst.leads))]

rho_0 = kwant.operator.Density(fsyst).bind()
rho_x = kwant.operator.Density(fsyst, sigma_x).bind()
rho_y = kwant.operator.Density(fsyst, sigma_y).bind()
rho_z = kwant.operator.Density(fsyst, sigma_z).bind()

J_0 = kwant.operator.Current(fsyst).bind()
J_x = kwant.operator.Current(fsyst, sigma_x).bind()
J_y = kwant.operator.Current(fsyst, sigma_y).bind()
J_z = kwant.operator.Current(fsyst, sigma_z).bind()
print(len(ss[0]))
psi = ss[0][0]

currents = []
for (idf,f) in enumerate([J_0,J_x,J_y,J_z]):
    current = f(psi)
    currents.append(current)

    fig, ax = plt.subplots(1,1)
    imag = mycurrent(fsyst, current, ax=ax, cmap=cmap, max_linewidth=1, min_linewidth=0.25, arrowsize=0.5)
    circle = plt.Circle(params['pos0'], 3/2*params['r0'], color='black', alpha=0.5, zorder=10)
    ax.add_patch(circle)
    circle = plt.Circle(params['pos0'], params['r0'], color='black', alpha=0.75, zorder=10)
    ax.add_patch(circle)
    ax.set_xlim(params['Ls'][0]*np.asarray([-1,1]))
    ax.set_xlim(params['Ls'][1]*np.asarray([-1,1])+np.asarray([1,-1]))
    ax.set_aspect('equal')
    ax.set_xlabel('$x/a$')
    ax.set_ylabel('$y/a$')
    # divider = make_axes_locatable(ax)
    # norm = mpl.colors.Normalize(vmin=0, vmax=np.max(np.abs(current)))
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical', label='$n_{0,\\bm r}$')
    plt.savefig(f'{path_out}/kwant_current_{idf}.png', dpi=1200, bbox_inches='tight', pad_inches=0.1)
    plt.close()

densities = []
for (idf,f) in enumerate([rho_0,rho_x,rho_y,rho_z]):
    density = f(psi)
    densities.append(density)

    fig, ax = plt.subplots(1,1)
    imag = kwant.plotter.density(fsyst, density, ax=ax, cmap=cmap)
    circle = plt.Circle(params['pos0'], 3/2*params['r0'], color='black', alpha=0.5, zorder=10)
    ax.add_patch(circle)
    circle = plt.Circle(params['pos0'], params['r0'], color='black', alpha=0.75, zorder=10)
    ax.add_patch(circle)
    ax.set_xlim(params['Ls'][0]*np.asarray([-1,1]))
    ax.set_xlim(params['Ls'][1]*np.asarray([-1,1]))
    ax.set_aspect('equal')
    ax.set_xlabel('$x/a$')
    ax.set_ylabel('$y/a$')
    # divider = make_axes_locatable(ax)
    # norm = mpl.colors.Normalize(vmin=0, vmax=np.max(np.abs(current)))
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')
    plt.savefig(f'{path_out}/kwant_density_{idf}.png', dpi=1200, bbox_inches='tight', pad_inches=0.1)
    plt.close()

print('-'*80, '\ncomputation finished.')

print('-'*80, '\nwrite results to files')
all_data = {'params': params, 'psi': psi}

json_string = json.dumps(params)
# save parameter file
with open(f'{path_out}/params.json', 'w') as outfile:
    outfile.write(json_string)
# save data
with h5py.File(f'{path_out}/data.hdf5', "w") as f:
    f.create_dataset('psi', data=psi)
    f.create_dataset('densities', data=densities)
    f.create_dataset('currents', data=currents)
    f.create_dataset('positions', data=positions)

print('-'*80, '\nfinished')