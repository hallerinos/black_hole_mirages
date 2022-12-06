import os, shutil, sys, glob
import pandas as pd
import numpy as np
import json
import h5py
import uuid
from copy import copy, deepcopy

from scipy import interpolate

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = ((3+3/8),(3+3/8))
mpl.rcParams['figure.figsize'] = (5.927/3,5.927/3)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

def find_files(find_str, pwd_path):
    message = "searching for " + find_str + ": "
    sys.stdout.write(message)
    filenames = []
    for filename in os.popen("find " + str(pwd_path) + " -path "
                            + '"' + str(find_str) + '"').read().split('\n')[0:-1]:
        filenames.append(filename)
    message = str(len(filenames)) + " files found.\n"
    sys.stdout.write(message)
    return filenames

path_in = "weyl_plots"
fns = find_files('*.hdf5', f'./{path_in}')

rts = [1.5, 1.4]
vfs = [0.9, 0.8]

cmap = mpl.colormaps['viridis']
# exit()
col = cmap(1.0)
ctr = 0
for fn in fns:
    print(f'Compute average trajectory... {(100*ctr)//len(fns)}%')
    with h5py.File(fn, "r+") as f:
        psi = np.asarray(f['psi'])
        positions = np.asarray(f['positions'])
        currents = np.asarray(f['currents'])
        densities = np.asarray(f['densities'])

    json_fn = fn.replace('data.hdf5','params.json')
    with open(json_fn) as file:
        params = json.load(file)

    df = pd.DataFrame()
    df['X'] = positions[:,0]
    df['Y'] = positions[:,1]
    df['rho_0'] = densities[0]

    fig, ax = plt.subplots(1,1)
    pivotted = df.pivot('Y', 'X', 'rho_0')
    extent = [np.min(df['X']),np.max(df['X']),np.min(df['Y']),np.max(df['Y'])]

    cmap.set_over('white')

    asel = int(2*params['alpha'])-1
    imag = ax.imshow(pivotted,extent=extent,origin='lower',cmap=cmap,vmin=0,vmax=0.01)
    circle = plt.Circle(params['pos0'], rts[asel]*params['r0'], facecolor='white', alpha=0.2, zorder=10, edgecolor='none')
    ax.add_patch(circle)
    circle = plt.Circle(params['pos0'], params['r0'], facecolor='white', alpha=0.3, zorder=10, edgecolor='none')
    ax.add_patch(circle)
    # ax.scatter(params['pos0'][0], params['pos0'][1], marker='x', color='black')
    ax.set_xlabel('$x/a$')
    ax.set_ylabel('$y/a$')
    ax.set_xlim(np.min(df['X']),np.max(df['X']))
    ax.set_ylim(np.min(df['Y']),np.max(df['Y']))
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(imag, cax=cax, orientation='vertical', extend='max')
    fig_fn = fn.replace('data.hdf5',f'lensing_device_eom.jpg')

    eom_fn = fn.replace('data.hdf5','data_semiclassics.csv')
    df_eom = pd.read_csv(eom_fn)
    ints = np.sort(np.unique([int(k[1:]) for k in df_eom.keys()]))
    # ints = ints[0:len(ints):5]

    for i in ints:
        line = ax.plot(df_eom[f'X{i}'],df_eom[f'Y{i}'],color=col,linewidth=0.5,linestyle='dashed')
        sze, col = 15, line[0].get_color()

        for n in range(0,len(df_eom[f'X{i}'])-1,520):
            x,y = df_eom[f'X{i}'].iloc[n],df_eom[f'Y{i}'].iloc[n]
            dx,dy = df_eom[f'X{i}'].iloc[n+1]-df_eom[f'X{i}'].iloc[n],df_eom[f'Y{i}'].iloc[n+1]-df_eom[f'Y{i}'].iloc[n]
            arrow = patches.FancyArrow(x, y, dx, dy, width=0, head_width=sze, head_length=3*sze, label='EOM',length_includes_head=True, overhang=0.7, zorder=10)
            arrow.set(capstyle='round', color=col, linewidth=0.5)
            ax.add_patch(arrow)
    
    math = 'r_{0,y}'
    txt = params['pos0'][1]
    props = dict(boxstyle='round', facecolor='black', alpha=0.5, edgecolor='none')
    # ax.text(-0.39,-0.25,f'${math}={txt}$', transform=ax.transAxes)
    print(fig_fn)
    plt.savefig(fig_fn, dpi=1200, bbox_inches='tight', pad_inches=0.01)
    # plt.show()
    plt.close()
    ctr += 1
    # if ctr==2:
    #     exit()