import os, shutil
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = ((3+3/8),(3+3/8))
plt.rc('text.latex', preamble=r'\usepackage{bm}')

path_out = f'trajs'
if os.path.exists(path_out):
    shutil.rmtree(path_out)
os.makedirs(path_out)

df_av = pd.read_csv('all_average_trajectories_alpha_0.5_2.csv')
df_av = df_av.sort_values('pos0_1').reset_index()

df = pd.read_csv('all_densities_alpha_0.5_2.csv')
df = df.sort_values('pos0_1').reset_index()
fns = np.unique(df['fn'])
ps = np.unique(df['pos0_1'])

for (idp,p) in enumerate(ps):
    print(f'Plot average trajectory... {(100*idp)//len(fns)}%')
    df_av_sel = df_av[df_av['pos0_1'] == p]
    df_sel = df[df['pos0_1'] == p]

    pivotted = df_sel.pivot('Y', 'X', 'rho_0')
    extent = [np.min(df_sel['X']),np.max(df_sel['X']),np.min(df_sel['Y']),np.max(df_sel['Y'])]
    print(extent)
    fig, ax = plt.subplots(1,1)
    imag = ax.imshow(pivotted, extent=extent, origin='lower', cmap='Reds')
    
    df_av_sel = df_av_sel.sort_values('X').reset_index()
    ax.plot(df_av_sel['X'], df_av_sel['<Y*rho_0>_x/<rho_0>_x'], color='black')
    
    pos0 = [df_av_sel['pos0_0'].iloc[0], df_av_sel['pos0_1'].iloc[0]]
    r0 = df_av_sel['r0'].iloc[0]
    circle = plt.Circle(pos0, 3/2*r0, color='black', alpha=0.5, zorder=10)
    ax.add_patch(circle)
    circle = plt.Circle(pos0, r0, color='black', alpha=0.75, zorder=10)
    ax.add_patch(circle)
    ax.set_xlabel('$x/a$')
    ax.set_ylabel('$y/a$')
    L = 800
    ax.set_xlim(-L,L)
    ax.set_ylim(-L,L)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(imag, cax=cax, orientation='vertical')
    
    plt.savefig(f'{path_out}/'+ f'{idp}'.zfill(4), dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()