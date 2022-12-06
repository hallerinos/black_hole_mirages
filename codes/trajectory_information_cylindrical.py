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
mpl.rcParams['figure.figsize'] = (6.2,6.2//2)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

# chis = ["1","-1"]
# csvfiles = [f'codes/{s}.csv' for s in [f'lensing_example_chi_{chi}' for chi in chis]]

csvfiles = ['codes/lensing_chi_1.csv', 'codes/lensing_chi_-1.csv']
# csvfiles = ['codes/unstable_orbit_example_chi_-1.csv', 'codes/unstable_orbit_example_chi_1.csv']

fig = plt.figure()
gs = fig.add_gridspec(1, 2, hspace=0, wspace=0.3)
axs = gs.subplots()
axs = axs.ravel()

cols = ['red', 'green']
lsts_offset = [0, 1]
opacities = [1, 1]
for csvfile in csvfiles:

    data = pd.read_csv(csvfile)
    chi = np.unique(data['\\chi'])[0]
    alpha = float(Fraction(np.unique(data['\\alpha'])[0]))
    rhot = np.unique(data['\\rho_t'])[0]
    vF = np.unique(data['v_F'])[0]

    print(chi,alpha,rhot,vF)

    kistr = "k_{\\rm initial}"

    keys_sel = [['t', '\\rho', '\\phi', 'z'], ['t', 'k_\\rho^{(-)}', 'k_\\rho^{(+)}', 'k_\\rho']]
    # keys_sel = [['t', '\\rho', '\\phi', 'z'], ['t', 'k_x', 'k_y', 'k_z']]
    icond = data[np.abs(data['t']) < 1e-12]
    kt0 = vF*np.sqrt((icond['k_x']**2 + icond['k_y']**2 + icond['k_z']**2)).iloc[0]
    Lt0 = np.sqrt((icond['L_x']**2 + icond['L_y']**2 + icond['L_z']**2)).iloc[0]
    pns = [f'$({s})$' for s in 'abcd']
    xlbls = [' ', f'k_\\rho^i/{kistr}']
    norm = np.sqrt((data['x']**2 + data['y']**2 + data['z']**2))
    tp = np.argsort(data['\\rho'])[0]

    for (ax, ks, pn, xlbl) in zip(axs, keys_sel, pns, xlbls):
        Xs = data['t']/kt0
        col=None
        lwd=None
        lst=None
        for k in ks[1:]:
            Ys = data[k]
            ax.plot([Xs[tp]]*2, [-100,100], color='black', linewidth=0.5)
            if k in keys_sel[0]:
                lbl = f'${k}$'
                if chi == -1 and k != 'z':
                    continue
                if k == 'z':
                    lbl += f"$/\\rho_t,\\ \\chi={chi}$"
                if k == '\\rho':
                    lbl += f"$/\\rho_t$"
                ax.set_ylim(-2,3)
            if k in keys_sel[1]:
                lbl = f'${k}$'
                Ys = Ys/kt0
                ax.set_ylim(-1.1,2.1)
                if k == 'k_\\rho':
                    lst='dotted'
                    col='black'
                else:
                    lst=None
                    col=None
                if chi == -1:
                    continue

            line, = ax.plot(Xs, Ys, label=lbl, color=col, linewidth=lwd, linestyle=lst)
        #     ax.plot([Xs[periapsis_time],Xs[periapsis_time]],[-100,100], color='black', linestyle='dotted', linewidth=0.5)
        # ax.set_ylim(ymin, ymax)
        tpaps = data['t'].iloc[np.argsort(norm).iloc[0]]/kt0
        # ax.plot([tpaps, tpaps], [np.min(Ys),np.max(Ys)], color='gray', linewidth=0.5)
        ax.set_ylabel(f'${xlbl}$')
        ax.text(0.01, 0.947, pn, transform = ax.transAxes)
axs[0].legend(fancybox=False, frameon=False, loc='upper left', handlelength=1.2)
axs[1].legend(fancybox=False, frameon=False, loc='center left', handlelength=1.75)
axs[-2].set_xlabel(f'$t\\, v_F{kistr}$')
axs[-1].set_xlabel(f'$t\\, v_F{kistr}$')
# leg = axs[0].legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False, handlelength=0.9)

out_fn = csvfiles[-1].replace('.csv','').replace('codes/', '').replace(f'_chi_{chi}', '')
plt.savefig(f'SciPost_resubmission/fig/{out_fn}_cylinder.pdf', bbox_inches='tight', pad_inches=0.01)
plt.savefig(f'SciPost_resubmission/fig/{out_fn}_cylinder.png', dpi=1200, bbox_inches='tight', pad_inches=0.01)

# then create a new image
# adjust the figure size as necessary
figsize = (6.2, 0.2)
fig_leg = plt.figure(figsize=figsize)
ax_leg = fig_leg.add_subplot(111)
# add the legend from the previous axes
ax_leg.legend(*axs[0].get_legend_handles_labels(), loc='center', ncol=3, frameon=False)
# hide the axes frame and the x/y labels
ax_leg.axis('off')
fig_leg.savefig(f'SciPost_resubmission/fig/{out_fn}_cylinder_legend.pdf', bbox_inches='tight', pad_inches=0.01)