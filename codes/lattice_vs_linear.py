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
mpl.rcParams['figure.figsize'] = (6.2,6.2/2)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

csvfiles = ['codes/lattice_vs_linear_capture.csv', 'codes/lattice_vs_linear_lensing.csv']

fig, axs = plt.subplots(1, 2, sharey=True)

for (id, csvfile) in enumerate(csvfiles):
    ax = axs[id]
    data = pd.read_csv(csvfile)
    with open(csvfile.replace('.csv','.json')) as file:
            p = json.load(file)

    circle = plt.Circle([p['x0']/p['r0'],p['y0']/p['r0']], 3.0/2, facecolor='black', alpha=0.2, edgecolor='none')
    ax.add_patch(circle)
    circle = plt.Circle([p['x0']/p['r0'],p['y0']/p['r0']], 1, facecolor='black', alpha=0.2, edgecolor='none')
    ax.add_patch(circle)
    circle = plt.Circle([p['x0']/p['r0'],p['y0']/p['r0']], 0.12, facecolor='black', edgecolor='none', zorder=10)
    ax.add_patch(circle)
    ax.set_aspect('equal')

    ax.plot([0,0],[-10,10], color='black', linewidth=0.5, zorder=0)
    ax.plot([-10,10],[0,0], color='black', linewidth=0.5, zorder=0)

    # ax.plot([p['x0']/p['r0'],p['x0']/p['r0']],[p['y0']/p['r0']-10,p['y0']/p['r0']+10], color='black', linestyle='dotted', linewidth=0.5, zorder=0)
    ax.plot([p['x0']/p['r0']-10,p['x0']/p['r0']+10],[p['y0']/p['r0'],p['y0']/p['r0']], color='black', linestyle='dotted', linewidth=0.5, zorder=0)

    nkeys = np.asarray(range(len(data.keys())//2))+1
    nkeys = [1, 2, 4, 6, 8, 10, 12, 14]
    sizes = [1.5]*len(nkeys)
    styles = ['dashed', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid']
    labels = ['${\\rm linearized}$']
    for es in p['energies']:
        labels.append(f'$Em^*a^2={es}$')
    zorders = [2, 1, 1, 1, 1, 1, 1, 1]
    for (n,si,st,lbl,zord) in zip(nkeys,sizes,styles,labels,zorders):
        ax.plot(data[f'X{n}']/p['r0'],data[f'Y{n}']/p['r0'],linewidth=si,linestyle=st, label=lbl, zorder=zord)
    # l1 = ax.legend(bbox_to_anchor=(1, 1), borderaxespad=0)
    # l2 = plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    # ax.legend(bbox_to_anchor=(1, 0.5), mode='expand', borderaxespad=0, loc="center left",frameon=False)
    # l4 = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
    #                 mode="expand", borderaxespad=0, ncol=3)
    # l5 = plt.legend(bbox_to_anchor=(1, 0), loc="lower right",
    #                 bbox_transform=fig.transFigure, ncol=3)
    # l6 = plt.legend(bbox_to_anchor=(0.4, 0.8), loc="upper right")
    ax.set_xlim(p['x0']/p['r0']-4,p['x0']/p['r0']+4)
    ax.set_ylim(p['y0']/p['r0']-4,p['y0']/p['r0']+4)
    ax.set_xlabel('$x/\\rho_t$')
axs[0].text(0.01, 0.95, "$(a)$", transform = axs[0].transAxes)
axs[1].text(0.01, 0.95, "$(b)$", transform = axs[1].transAxes)
axs[0].set_ylabel('$y/\\rho_t$')
# axs[0].legend(loc="upper center", handlelength=1.3, frameon=False, ncol=2)
plt.tight_layout()
# plt.show()
plt.savefig('fig/linearized_vs_lattice.pdf', bbox_inches='tight', pad_inches=0.01)
plt.savefig('fig/linearized_vs_lattice.png', dpi=1200, bbox_inches='tight', pad_inches=0.01)

# then create a new image
# adjust the figure size as necessary
figsize = (6.2, 0.2)
fig_leg = plt.figure(figsize=figsize)
ax_leg = fig_leg.add_subplot(111)
# add the legend from the previous axes
ax_leg.legend(*axs[0].get_legend_handles_labels(), loc='center', ncol=4, frameon=False)
# hide the axes frame and the x/y labels
ax_leg.axis('off')
fig_leg.savefig('fig/linearized_vs_lattice_legend.pdf', bbox_inches='tight', pad_inches=0.01)