import os, shutil, sys, glob
import pandas as pd
import numpy as np
from copy import copy
import json
import h5py
import uuid
from copy import copy, deepcopy
from fractions import Fraction

from scipy import interpolate

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = (6.2,4)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

chis = ['-1.', '1.']
csvfiles = [f'codes/{s}.csv' for s in [f'lensed_trajectory_chi_{chi}_alpha_0.5' for chi in chis]]
# csvfiles = [f'codes/{s}.csv' for s in [f'unstable_orbit_chi_{chi}_alpha_0.5' for chi in chis]]
print(csvfiles)

fig = plt.figure()
gs = fig.add_gridspec(2, 2, hspace=0.05, wspace=0.29)
axs = gs.subplots(sharex='col')
axs = axs.ravel()

cols = ['red', 'green']
lsts_offset = [0, 1]
opacities = [1, 1]
for (csvfile, col, lso, opcty) in zip(csvfiles, cols, lsts_offset, opacities):

    data = pd.read_csv(csvfile)
    chi = np.unique(data['\\chi'])[0]
    alpha = float(Fraction(np.unique(data['\\alpha'])[0]))

    keys_sel = [['t', 'x', 'y', 'z'], ['t', 'k_x', 'k_y', 'k_z'], ['t', 's_x', 's_y', 's_z'], ['t', 'L_x', 'L_y', 'L_z']]
    icond = data[data['t']==0]
    kt0 = np.sqrt((icond['k_x']**2 + icond['k_y']**2 + icond['k_z']**2)).iloc[0]
    Lt0 = np.sqrt((icond['L_x']**2 + icond['L_y']**2 + icond['L_z']**2)).iloc[0]
    pns = [f'$({s})$' for s in 'abcd']
    lsts = [(lso, (1,1)), (lso, (1,1)), (lso, (1,1))]
    xlbls = ['i/\\rho_t', 'k_i/k_{\\rm initial}', 's_i', 'L_i/L_{\\rm initial}']
    norm = np.sqrt((data['x']**2 + data['y']**2 + data['z']**2))
    for (ax, ks, pn, xlbl) in zip(axs, keys_sel, pns, xlbls):
        Xs = data['t']/kt0
        for (k, lst) in zip(ks[1:], lsts):
            Ys = data[k]
            if k in keys_sel[0]:
                lbl = f'${k}/\\rho_t,\\ \\chi={{{chi}}}$'
            if k in keys_sel[1]:
                tp = "k_{\\rm initial}"
                lbl = f'${k}/{tp},\\ \\chi={{{chi}}}$'
                Ys = Ys/kt0
            if k in keys_sel[2]:
                lbl = f'${k},\\ \\chi={{{chi}}}$'
            if k in keys_sel[3]:
                lbl = f'${k[2]},\\ \\chi={{{chi}}}$'
                Ys = Ys/Lt0
            line, = ax.plot(Xs, Ys, alpha=opcty, linestyle=lst, label=lbl)
        tpaps = data['t'].iloc[np.argsort(norm).iloc[0]]/kt0
        # ax.plot([tpaps, tpaps], [np.min(Ys),np.max(Ys)], color='gray', linewidth=0.5)
        ax.set_ylabel(f'${xlbl}$')
        ax.text(0.02, 0.86, pn, transform = ax.transAxes)
        # ax.legend(fancybox=False, frameon=False)
axs[-2].set_xlabel('$t\\,v_Fk_{\\rm initial}$')
axs[-1].set_xlabel('$t\\,v_Fk_{\\rm initial}$')
# leg = axs[0].legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False, handlelength=0.9)

out_fn = csvfiles[-1].replace(f'chi_{chis[-1]}_','').replace('.csv','').replace('codes/', '')
plt.savefig(f'fig/{out_fn}.pdf', bbox_inches='tight', pad_inches=0.01)
plt.savefig(f'fig/{out_fn}.png', dpi=1200, bbox_inches='tight', pad_inches=0.01)

# then create a new image
# adjust the figure size as necessary
figsize = (6.2, 0.2)
fig_leg = plt.figure(figsize=figsize)
ax_leg = fig_leg.add_subplot(111)
# add the legend from the previous axes
ax_leg.legend(*axs[0].get_legend_handles_labels(), loc='center', ncol=3, frameon=False)
# hide the axes frame and the x/y labels
ax_leg.axis('off')
fig_leg.savefig(f'fig/{out_fn}_legend.pdf', bbox_inches='tight', pad_inches=0.01)