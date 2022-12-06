import os, shutil, sys, glob
import pandas as pd
import numpy as np
import json
import h5py
import uuid
from copy import copy, deepcopy

from scipy import interpolate

import kwant

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = ((3+3/8),(3+3/8))
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

df_av_all = pd.DataFrame()
df_all = pd.DataFrame()
path_in = "weyl_plots_alpha_0.5_2"
fns = find_files('*.hdf5', f'./{path_in}')
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
    df['Y*rho_0'] = df['Y']*df['rho_0']

    df_av = df.groupby('X').mean().reset_index()
    df_av['<Y*rho_0>_x/<rho_0>_x'] = df_av['Y*rho_0']/df_av['rho_0']

    # fig, ax = plt.subplots(1,1)
    # pivotted = df.pivot('Y', 'X', 'rho_0')
    # extent = [np.min(df['X']),np.max(df['X']),np.min(df['Y']),np.max(df['Y'])]
    # imag = ax.imshow(pivotted,extent=extent,origin='lower',cmap='Reds')
    # ax.plot(df_av['X'], df_av['<Y*rho_0>_x/<rho_0>_x'], color='black')
    # circle = plt.Circle(params['pos0'], 3/2*params['r0'], color='black', alpha=0.5, zorder=10)
    # ax.add_patch(circle)
    # circle = plt.Circle(params['pos0'], params['r0'], color='black', alpha=0.75, zorder=10)
    # ax.add_patch(circle)
    # ax.set_xlabel('$x/a$')
    # ax.set_ylabel('$y/a$')
    # ax.set_xlim(np.min(df['X']),np.max(df['X']))
    # ax.set_ylim(np.min(df['Y']),np.max(df['Y']))
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(imag, cax=cax, orientation='vertical')
    # fig_fn = fn.replace('data.hdf5','average_trajectory.png')
    
    # plt.savefig(fig_fn, dpi=300, bbox_inches='tight', pad_inches=0)
    # # plt.show()
    # plt.close()

    df_av['fn'] = fn
    df_av['r0'] = params['r0']
    df_av['pos0_0'] = params['pos0'][0]
    df_av['pos0_1'] = params['pos0'][1]
    df_av_all = pd.concat([df_av_all, df_av], ignore_index=True)

    df['fn'] = fn
    df['r0'] = params['r0']
    df['pos0_0'] = params['pos0'][0]
    df['pos0_1'] = params['pos0'][1]
    # keep only the visible densities
    df = df[np.abs(df['rho_0']) > 1e-6*np.max(np.abs(df['rho_0']))]
    df_all = pd.concat([df_all, df], ignore_index=True)
    ctr += 1

# dump data
df_av_all.to_csv(f'all_average_trajectories_{path_in}.csv')
df_all.to_csv(f'all_densities_{path_in}.csv')