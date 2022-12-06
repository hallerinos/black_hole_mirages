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
from matplotlib.patches import ArrowStyle

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = (6.2/2,6.2/2)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

csvfile = f'codes/transverse_shift_example_chi_1.csv'

r0 = 3

fig = plt.figure()
gs = fig.add_gridspec(2, 1, hspace=0.5)
axs = gs.subplots()
axs = axs.ravel()

lbls = 'abcd'
pranges = [3, 4, 1.3, 0.5]
szes = 0.04*np.asfarray(pranges)
arrowoffs=[[0,60,160,33],[300,20,600,93],[400,30,900,35]]
arrowoffs=[[0,180,130,12],[0,250,400,98],[0,30,150,77],[0,20,400,65]]

cols = []

data = pd.read_csv(csvfile)

ax = axs[0]
circle = plt.Circle([0,0], 3.0/2, facecolor='black', alpha=0.2, edgecolor='none')
ax.add_patch(circle)
circle = plt.Circle([0,0], 1, facecolor='black', alpha=0.2, edgecolor='none')
ax.add_patch(circle)
circle = plt.Circle([0,0], 0.12, facecolor='black', edgecolor='none', zorder=10)
ax.add_patch(circle)

ax.plot([0,0],[-100,100], color='gray', linewidth=0.5, zorder=0)
ax.plot([-100,100],[0,0], color='gray', linewidth=0.5, zorder=0)

Xs, Ys = data['x'], data['y']
ax.plot(Xs, Ys, color='black')
sze = 0.5
for i in [90]:
    x,y = Xs[i], Ys[i]
    dx, dy = Xs[i+1]-Xs[i], Ys[i+1]-Ys[i]
    arrow = patches.FancyArrow(x, y, dx, dy, width=0., head_width=sze, head_length=1.62*sze,head_starts_at_zero=False, length_includes_head=True, overhang=0.01, linewidth=1, zorder=100)
    arrow.set(capstyle='round', color='black')
    ax.add_patch(arrow)
distsxy = np.sqrt(data['x']**2 + data['y']**2)
data_sel = data[np.sqrt(data['x']**2 + data['y']**2)==np.min(distsxy)]
print(data_sel)
ax.annotate("", xy=(-0.01, -0.01), xytext=(data_sel['x'], data_sel['y']), textcoords=ax.transData, arrowprops=dict(arrowstyle=ArrowStyle("|-|", widthA=0.2, angleA=0, widthB=0.2, angleB=0)))
ax.text(0.7, 0.6, "$\\rho_p$", transform=ax.transData)
x0 = 0; xr = 10
ax.set_xlim(x0-xr,x0+xr)
y0 = 0; yr = 4
ax.set_ylim(y0-yr,y0+yr)
ax.text(0.0125, 0.88, "$(a)$", transform = ax.transAxes)
ax.set_xlabel('$x/\\rho_t$')
ax.set_ylabel('$y/\\rho_t$')


ax = axs[1]
Xs, Ys = data['y'], data['z']
ax.plot(Xs, Ys, color='black')
sze = 0.06
for i in [40, 250]:
    x,y = Xs[i], Ys[i]
    dx, dy = Xs[i+1]-Xs[i], Ys[i+1]-Ys[i]
    arrow = patches.FancyArrow(x, y, dx, dy, width=0., head_width=sze, head_length=30*sze,head_starts_at_zero=False, length_includes_head=True, overhang=0.01, linewidth=1, zorder=100)
    arrow.set(capstyle='round', color='black')
    ax.add_patch(arrow)
x0 = -14; xr = 20
ax.set_xlim(x0-xr,x0+xr)
y0 = -0.45; yr = 0.5
ax.set_ylim(y0-yr,y0+yr)
ax.set_xlabel('$y/\\rho_t$')
ax.set_ylabel('$z/\\rho_t$')

ax.annotate("", xy=(-31, data['z'].iloc[0]), xytext=(-31, data['z'].iloc[-1]), textcoords=ax.transData, arrowprops=dict(arrowstyle=ArrowStyle("|-|", widthA=0.2, angleA=0, widthB=0.2, angleB=0)))
ax.text(0.0125, 0.88, "$(b)$", transform = ax.transAxes)
ax.text(-30, -0.6, "$\\Delta z$", transform=ax.transData)
plt.savefig('fig/transverse_shift_ab.pdf', bbox_inches='tight', pad_inches=0.01)
plt.savefig('fig/transverse_shift_ab.png', dpi=1200, bbox_inches='tight', pad_inches=0.01)

# # then create a new image
# # adjust the figure size as necessary
# figsize = (6.2, 0.2)
# fig_leg = plt.figure(figsize=figsize)
# ax_leg = fig_leg.add_subplot(111)
# # add the legend from the previous axes
# ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=4, frameon=False)
# # hide the axes frame and the x/y labels
# ax_leg.axis('off')
# fig_leg.savefig('fig/fig1_legend.pdf', bbox_inches='tight', pad_inches=0.01)