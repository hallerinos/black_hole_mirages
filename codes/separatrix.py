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
mpl.rcParams['figure.figsize'] = (6.2,6.2/2.5)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

csvfiles = ['codes/lattice_vs_linear_capture.csv', 'codes/lattice_vs_linear_lensing.csv']

fig, axs = plt.subplots(1, 2)
axs = axs.ravel()
alphas = [0.5, 1, 2, 4]
vF = 1
lbls = ['$\\alpha=1/2$', '$\\alpha=1$', '$\\alpha=2$', '$\\alpha=4$']
for (ida, (alpha, lbl)) in enumerate(zip(alphas,lbls)):
    rt = lambda alpha: ((1 + alpha)/vF**2)**((1/2)/alpha)
    u = lambda rho: (1/rho)**alpha
    veff = lambda rho: vF/rho**2*(1-(u(rho)**2/vF**2))
    Xs = np.linspace(1,2,1001)
    Ys = veff(Xs)
    p = axs[0].plot(Xs, Ys, label=lbl)
    rtpos = rt(alpha)
    col = p[-1].get_color()
    axs[0].plot([rtpos,rtpos], [-10,10], color= col, linestyle='dotted')
# axs[0].legend(frameon=False, loc=(0.55,-0.02))
axs[0].set_xlim(1, np.max(Xs))
axs[0].set_ylim(0, 0.575)
axs[0].set_xlabel('$\\rho/\\rho_t$')
axs[0].set_ylabel('$\\varepsilon_{p,\\rm eff}\\rho_t^2/(v_F\, L_z)^2$')
axs[0].text(0.0125, 0.93, "$(a)$", transform = axs[0].transAxes)

vFs = [1]
rt = lambda alpha: (1 + alpha)**((1/2)/alpha)
Xs = np.linspace(0.5,1e2,1001)
Ys = rt(Xs)
axs[1].plot(Xs, Ys, color='black')
# axs[1].legend(frameon=False, loc='upper right')
axs[1].set_xlim(np.min(Xs), np.max(Xs))
axs[1].set_ylim(1, 1.5)
axs[1].set_xlabel('$\\alpha$')
axs[1].set_ylabel('$(1+\\alpha)^{1/2\\alpha}$')
axs[1].set_xscale('log')
axs[1].text(0.075, 0.93, "$(b)$", transform = axs[1].transAxes)


# axs[0].set_xscale('log')
# ax.set_xscale('log')
# ax.set_yscale('log')
plt.tight_layout()
# plt.show()
plt.savefig('fig/effective_potential.pdf', bbox_inches='tight', pad_inches=0.01)
plt.savefig('fig/effective_potential.png', dpi=1200, bbox_inches='tight', pad_inches=0.01)

# then create a new image
# adjust the figure size as necessary
figsize = (6.2, 0.2)
fig_leg = plt.figure(figsize=figsize)
ax_leg = fig_leg.add_subplot(111)
# add the legend from the previous axes
ax_leg.legend(*axs[0].get_legend_handles_labels(), loc='center', ncol=4, frameon=False)
# hide the axes frame and the x/y labels
ax_leg.axis('off')
fig_leg.savefig('fig/effective_potential_legend.pdf', bbox_inches='tight', pad_inches=0.01)