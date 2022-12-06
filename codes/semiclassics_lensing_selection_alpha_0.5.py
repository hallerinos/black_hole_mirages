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
mpl.rcParams['figure.figsize'] = ((3+3/8), (3+3/8))
mpl.rcParams['figure.figsize'] = (5.927, 2.0/3*5.927)
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

df = pd.DataFrame()
for fn in fns:
    json_fn = fn.replace('data.hdf5','params.json')
    with open(json_fn) as file:
        params = json.load(file)

    tp = pd.DataFrame()
    tp['fn'] = [fn]
    tp['pos00'] = [params['pos0'][0]]
    tp['pos01'] = [params['pos0'][1]]
    tp['alpha'] = [params['alpha']]
    tp['Ls0'] = [params['Ls'][0]]
    tp['Ls1'] = [params['Ls'][1]]
    tp['r0'] = [params['r0']]
    tp['fd'] = [params['fd']]
    tp['fu'] = [params['fu']]
    tp['k0'] = [params['k0']]
    tp['energy'] = [params['energy']]
    df = pd.concat([df, tp])

df_sel = df[df['alpha']==0.5]
# df_sel = df_sel[df_sel['fd']==0.15]
df_sel = df_sel.sort_values('pos01', ascending=False)
# print(df_sel)
# exit()

fig = plt.figure()
gs = fig.add_gridspec(2, 3, hspace=0.05, wspace=0.05)
axs = gs.subplots(sharex='col', sharey='row')
axs = axs.ravel()

sel = np.asarray([False]*len(df_sel))
# sel[0] = True
# sel[1] = True
# sel[2] = True
# sel[5] = True
# sel[6] = True
# sel[7] = True
# sel[8] = True
# sel[9] = True
# sel[10] = True
# sel[13] = True
# sel[14] = True
# sel[15] = True

ind_pos = [0,4,8,12,16,20]
sel[ind_pos] = [True]*len(ind_pos)

lbls = [f'$({l})$' for l in 'abcdefghijklmnopqrs']
for (idf, fn) in enumerate(df_sel['fn'][sel]):
    with h5py.File(fn, "r+") as f:
        psi = np.asarray(f['psi'])
        positions = np.asarray(f['positions'])
        densities = np.asarray(f['densities'])

    json_fn = fn.replace('data.hdf5','params.json')
    with open(json_fn) as file:
        params = json.load(file)
    
    df = pd.DataFrame()
    df['X'] = positions[:,0]
    df['Y'] = positions[:,1]
    df['rho_0'] = densities[0]

    pivotted = df.pivot('Y', 'X', 'rho_0')
    extent = [np.min(df['X']),np.max(df['X']),np.min(df['Y']),np.max(df['Y'])]

    cmap.set_over('white')

    ax = axs[idf]

    asel = int(2*params['alpha'])-1
    imag = ax.imshow(pivotted,extent=extent,origin='lower',cmap=cmap,vmin=0,vmax=0.01)
    circle = plt.Circle(params['pos0'], rts[asel]*params['r0'], facecolor='white', alpha=0.2, zorder=10, edgecolor='none')
    ax.add_patch(circle)
    circle = plt.Circle(params['pos0'], params['r0'], facecolor='white', alpha=0.3, zorder=10, edgecolor='none')
    ax.add_patch(circle)

    ax.set_xlim(np.min(df['X']),np.max(df['X']))
    ax.set_ylim(np.min(df['Y']),np.max(df['Y']))

    eom_fn = fn.replace('data.hdf5','data_semiclassics.csv')
    
    df_eom = pd.read_csv(eom_fn)
    ints = np.sort(np.unique([int(k[1:]) for k in df_eom.keys()]))

    for i in ints:
        sze, col = 10, 'white'  # line[0].get_color()
        line = ax.plot(df_eom[f'X{i}'], df_eom[f'Y{i}'], color=col, linewidth=0.5, alpha=0.3)

        for n in range(0,len(df_eom[f'X{i}'])-1,520):
            x,y = df_eom[f'X{i}'].iloc[n],df_eom[f'Y{i}'].iloc[n]
            dx,dy = df_eom[f'X{i}'].iloc[n+1]-df_eom[f'X{i}'].iloc[n],df_eom[f'Y{i}'].iloc[n+1]-df_eom[f'Y{i}'].iloc[n]
            arrow = patches.FancyArrow(x, y, dx, dy, width=0, head_width=sze, head_length=3*sze, label='EOM',length_includes_head=True, overhang=0.7, zorder=10)
            arrow.set(capstyle='round', color=col, linewidth=0.5, alpha=0.3)
            ax.add_patch(arrow)
    
    math = 'r_{0,y}'
    txt = params['pos0'][1]
    props = dict(boxstyle='round', facecolor='black', alpha=0.5, edgecolor='none')
    ax.text(0.05, 0.9, lbls[idf], transform=ax.transAxes, color='white')
axs[0].set_ylabel('$y/a$')
axs[3].set_ylabel('$y/a$')
# axs[6].set_ylabel('$y/a$')
# axs[9].set_ylabel('$y/a$')
axs[-3].set_xlabel('$x/a$')
axs[-2].set_xlabel('$x/a$')
axs[-1].set_xlabel('$x/a$')

plt.savefig('fig/trajectories_alpha_0.5.jpg', dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.close()