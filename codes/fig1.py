import os, shutil, sys, glob
import pandas as pd
import numpy as np
from copy import copy
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
mpl.rcParams['figure.figsize'] = (6.2,6.2)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

csvfiles = [f'codes/Fig1_{s}.csv' for s in ['orbits','momenta','velocities','acceleration']]

r0 = 3

fig = plt.figure()
gs = fig.add_gridspec(2, 2, hspace=0.05, wspace=0.5)
axs = gs.subplots()
axs = axs.ravel()

lbls = 'abcdefghijklmnopqrstuvwxyz'
pranges = [3, 4, 1.3, 0.5]
szes = 0.04*np.asfarray(pranges)
arrowoffs=[[0,60,160,33],[300,20,600,93],[400,30,900,35]]
arrowoffs=[[0,180,130,12],[0,250,400,98],[0,30,150,77],[0,20,400,65]]

cols = []
lgds = ['$\\rm separatrix$','$\\rm capture$','$\\rm lensing$','$\\rm radial\\ escape$']
for (id, (csvfile, sze, lbl, pr)) in enumerate(zip(csvfiles,szes,lbls,pranges)):
    ax = axs[id]
    data = pd.read_csv(csvfile)

    if csvfile == 'codes/Fig1_orbits.csv':
        circle = plt.Circle([0,0], 3.0/2, facecolor='black', alpha=0.2, edgecolor='none')
        ax.add_patch(circle)
        circle = plt.Circle([0,0], 1, facecolor='black', alpha=0.2, edgecolor='none')
        ax.add_patch(circle)
        circle = plt.Circle([0,0], 0.12, facecolor='black', edgecolor='none', zorder=10)
        ax.add_patch(circle)
    # if csvfile == 'codes/Fig1_velocities.csv':
    #     circle = plt.Circle([0,0], 1, facecolor='black', alpha=0.2, edgecolor='none')
    #     ax.add_patch(circle)
    #     circle = plt.Circle([0,0], 1/np.sqrt(3), facecolor='black', alpha=0.2, edgecolor='none')
    #     ax.add_patch(circle)
    # if csvfile == 'codes/Fig1_momenta.csv':
    #     circle = plt.Circle([0,0], 3, facecolor='black', alpha=0.2, edgecolor='none')
    #     ax.add_patch(circle)
    # if csvfile == 'codes/Fig1_acceleration.csv':
    #     circle = plt.Circle([0,0], 1/3/np.sqrt(2), facecolor='black', alpha=0.2, edgecolor='none')
    #     ax.add_patch(circle)
    ax.set_aspect('equal')

    ax.plot([0,0],[-10,10], color='black', linewidth=0.5, zorder=0)
    ax.plot([-10,10],[0,0], color='black', linewidth=0.5, zorder=0)

    nkeys = np.asarray(range(len(data.keys())//3))+1
    for (idn,(n,lgd)) in enumerate(zip(nkeys, lgds)):
        Xs, Ys = data[f'X{n}'], data[f'Y{n}']
        p = ax.plot(Xs, Ys, linewidth=1, label=f'{lgd}')
        col = p[-1].get_color()
        cols.append(col)

        aos = arrowoffs[id]
        imax = np.flip(Xs.notna()).idxmax()
        for i in [imax-aos[idn]-1]:
            # dx, dy = np.mean(Xs[i:i+4]-Xs[i]), np.mean(Ys[i:i+4]-Ys[i])
            # x,y = np.mean(Xs[i:i+4]), np.mean(Ys[i:i+4])
            x,y = Xs[i], Ys[i]
            dx, dy = Xs[i+1]-Xs[i], Ys[i+1]-Ys[i]
            arrow = patches.FancyArrow(x, y, dx, dy, width=0., head_width=sze, head_length=1.62*sze,head_starts_at_zero=False, length_includes_head=True, overhang=0.01, linewidth=1, zorder=100)
            arrow.set(capstyle='round', color=col)
            ax.add_patch(arrow)
        
        # ax.arrow(data[f'X{n}'].iloc[al], data[f'Y{n}'].iloc[al], data[f'X{n}'].iloc[al+1]-data[f'X{n}'].iloc[al], data[f'Y{n}'].iloc[al+1]-data[f'Y{n}'].iloc[al], lw=0, length_includes_head=True, head_width=0.2, head_length=0.5, color=col)
    # l1 = ax.legend(bbox_to_anchor=(1, 1), borderaxespad=0)
    # l2 = plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    # ax.legend(bbox_to_anchor=(1, 0.5), mode='expand', borderaxespad=0, loc="center left",frameon=False)
    # l4 = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
    #                 mode="expand", borderaxespad=0, ncol=3)
    # l5 = plt.legend(bbox_to_anchor=(1, 0), loc="lower right",
    #                 bbox_transform=fig.transFigure, ncol=3)
    # l6 = plt.legend(bbox_to_anchor=(0.4, 0.8), loc="upper right")
    ax.text(0.0125, 0.93, f"$({lbl})$", transform = ax.transAxes)
    ax.set_xlim(-pr,+pr)
    ax.set_ylim(-pr,+pr)
# axs[3].set_xlim(-0.085,+0.085)
# axs[3].set_ylim(-0.085,+0.085)
axs[0].set_xlabel('$x/\\rho_t$')
axs[0].set_ylabel('$y/\\rho_t$')
axs[1].set_xlabel('$k_x/k_{\\rm initial}$')
axs[1].set_ylabel('$k_y/k_{\\rm initial}$')
axs[2].set_xlabel('$\\dot x/v_F$')
axs[2].set_ylabel('$\\dot y/v_F$')
axs[3].set_xlabel('$\\ddot x\\rho_t/v_F^2$')
axs[3].set_ylabel('$\\ddot y\\rho_t/v_F^2$')

# plt.tight_layout()
# plt.show()
# exit()
plt.savefig('fig/fig1.pdf', bbox_inches='tight', pad_inches=0.01)
plt.savefig('fig/fig1.png', dpi=1200, bbox_inches='tight', pad_inches=0.01)

# then create a new image
# adjust the figure size as necessary
figsize = (6.2, 0.2)
fig_leg = plt.figure(figsize=figsize)
ax_leg = fig_leg.add_subplot(111)
# add the legend from the previous axes
ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=4, frameon=False)
# hide the axes frame and the x/y labels
ax_leg.axis('off')
fig_leg.savefig('fig/fig1_legend.pdf', bbox_inches='tight', pad_inches=0.01)