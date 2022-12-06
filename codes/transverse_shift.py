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
import matplotlib.gridspec as gridspec

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = (6.2/2,6.2/2)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

csvfile = f'codes/transverse_shift_chi_1_rhot_1_vF_1.csv'

fig = plt.figure()
gs = fig.add_gridspec(1, 1)
ax = gs.subplots()

data = pd.read_csv(csvfile)

keys = data.keys()
alphas = np.unique(data['\\alpha'])
for a in alphas:
    data_sel = data[data['\\alpha']==a]
    p = ax.plot(data_sel[keys[2]], -data_sel[keys[4]], alpha=0.5)
    Xs, Ys = data_sel[keys[2]], -data_sel[keys[3]]
    ax.scatter(Xs[::4], Ys[::4], color=p[-1].get_color(), label=f'$\\alpha={a}$')
    ax.plot(data_sel[keys[2]], -data_sel[keys[5]], color=p[-1].get_color(), linestyle='dotted')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$\\rho_p/\\rho_t$')
ax.set_ylabel('$-\\chi\\Delta z/\\rho_t$')
ax.set_ylim(1e-7,0.8)
ax.text(0.2, 0.94, "$(c)$", transform = ax.transAxes)
# plt.show()

out_fn = csvfile.replace('.csv','').replace('codes/', '')
ax.plot([], [], color='black', zorder=1, label='$\\rm Eq.\\ (39)$')
ax.plot([], [], linestyle='dotted', color='black', zorder=1, label='$\\rm Eq.\\ (21)$')
ax.legend(fancybox=False, frameon=False, fontsize=8)
# plt.tight_layout()

print(out_fn)
plt.savefig(f'fig/{out_fn}.pdf', bbox_inches='tight', pad_inches=0.01)
plt.savefig(f'fig/{out_fn}.png', dpi=1200, bbox_inches='tight', pad_inches=0.01)

# then create a new image
# adjust the figure size as necessary
figsize = (6.2, 0.2)
fig_leg = plt.figure(figsize=figsize)
ax_leg = fig_leg.add_subplot(111)
# add the legend from the previous axes
ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=6, frameon=False)
# hide the axes frame and the x/y labels
ax_leg.axis('off')
fig_leg.savefig(f'fig/{out_fn}_legend.pdf', bbox_inches='tight', pad_inches=0.01)