from cmath import nan
import os, shutil, sys, glob
import pandas as pd
import numpy as np
from copy import copy
import json
import h5py
import uuid
from copy import copy, deepcopy
from scipy.optimize import curve_fit

from scipy import interpolate

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

def filled_arc(center, radius, theta1, theta2, ax, color):

    circ = patches.Wedge(center, radius, theta1, theta2, facecolor=color, alpha=0.6, edgecolor='none')
    pt1 = (radius * (np.cos(theta1*np.pi/180.)) + center[0],
           radius * (np.sin(theta1*np.pi/180.)) + center[1])
    pt2 = (radius * (np.cos(theta2*np.pi/180.)) + center[0],
           radius * (np.sin(theta2*np.pi/180.)) + center[1])
    pt3 = center
    ax.add_patch(circ)


mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = (6.2,6.2)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

csvfile = 'codes/capture_high_energies.csv'

r0 = 3

fig, axs = plt.subplots(2, 2)
axs = axs.ravel()

lbls = 'abcdefghijklmnopqrstuvwxyz'
pranges = [20, 4, 1.3, 0.5]
szes = 0.04*np.asfarray(pranges)
arrowoffs=[[0,60,160,33],[300,20,600,93],[400,30,900,35]]
arrowoffs=[[0,180,130,12],[0,250,400,98],[0,30,150,77],[0,20,400,65]]

data = pd.read_csv(csvfile)
ax = axs[0]
ax.set_aspect('equal')
ax.plot([0,0],[-100,100], color='black', linewidth=0.5)
ax.plot([-100,100],[0,0], color='black', linewidth=0.5)
cmap = mpl.cm.get_cmap('viridis')

nkeys = np.unique(data[data['t']==0]['E'])
cols = [cmap(x) for x in np.linspace(0,1,len(nkeys))]
for (idn,n) in enumerate(nkeys):
    data_sel = data[abs(data['E']-n)<1e-3]
    idxs = data_sel[abs(data_sel['E']-n)>1e-5].index
    if not idxs.empty:
        maxi = idxs[0]
        if abs(n - 0.01) < 1e-7:
            maxi = 152
        data_sel = data_sel.loc[:maxi]
    Xs, Ys = np.asfarray(data_sel['x']), np.asfarray(data_sel['y'])
    Xs /= data_sel['\\rho_t']
    Ys /= data_sel['\\rho_t']

    lbl = n.round(2)
    if n==2:
        p = ax.plot(Xs, Ys, zorder=999, label=f'$\\rm linearized$', linestyle='dashed', color='black')
    else:
        p = ax.plot(Xs, Ys, zorder=len(nkeys)-idn-1, label=f'$Em^*a^2={lbl}$', color=cols[idn])
circle = plt.Circle([0,0], 3.0/2, facecolor='black', alpha=0.2, edgecolor='none')
ax.add_patch(circle)
circle = plt.Circle([0,0], 1, facecolor='black', alpha=0.2, edgecolor='none')
ax.add_patch(circle)
circle = plt.Circle([0,0], 0.12, facecolor='black', edgecolor='none', zorder=10)
ax.add_patch(circle)
ax.text(0.0125, 0.92, f"$(a)$", transform = ax.transAxes)
ax.set_xlim(-4,+4)
ax.set_ylim(-4,+4)
ax.set_xlabel('$x/\\rho_t$')
ax.set_ylabel('$y/\\rho_t$')

csvfile = 'codes/weak_lensing_high_energies.csv'
data = pd.read_csv(csvfile)
ax = axs[1]
ax.set_aspect('equal')
ax.plot([0,0],[-100,100], color='black', linewidth=0.5)
ax.plot([-100,100],[0,0], color='black', linewidth=0.5)

nkeys = np.unique(data[data['t']==0]['E'])
for (idn,n) in enumerate(nkeys):
    data_sel = data[abs(data['E']-n)<1e-3]
    idxs = data_sel[abs(data_sel['E']-n)>1e-5].index
    if not idxs.empty:
        maxi = idxs[0]
        if abs(n - 0.01) < 1e-7:
            maxi = 152
        data_sel = data_sel.loc[:maxi]
    Xs, Ys = np.asfarray(data_sel['x']), np.asfarray(data_sel['y'])
    Xs /= data_sel['\\rho_t']
    Ys /= data_sel['\\rho_t']

    lbl = n.round(2)
    if n==2:
        p = ax.plot(Xs, Ys, zorder=999, label=f'$Em^*a^2={lbl}$', linestyle='dashed', color='black')
    else:
        p = ax.plot(Xs, Ys, zorder=len(nkeys)-idn-1, label=f'$Em^*a^2={lbl}$', color=cols[idn])
circle = plt.Circle([0,0], 3.0/2, facecolor='black', alpha=0.2, edgecolor='none')
ax.add_patch(circle)
circle = plt.Circle([0,0], 1, facecolor='black', alpha=0.2, edgecolor='none')
ax.add_patch(circle)
circle = plt.Circle([0,0], 0.12, facecolor='black', edgecolor='none', zorder=10)
ax.add_patch(circle)
ax.text(0.0125, 0.92, f"$(b)$", transform = ax.transAxes)
ax.set_xlim(-10,+10)
ax.set_ylim(-10,+10)
ax.set_xlabel('$x/\\rho_t$')
ax.set_ylabel('$y/\\rho_t$')

ax = axs[2]
csvfile = 'codes/lensing_angle_high_energies_diag_chi_1_rhot_110_vF_1.csv'
data_numeric = pd.read_csv(csvfile)
enes = np.unique(data_numeric['E'])
for (ide,e) in enumerate(enes):
    data_slice = data_numeric[data_numeric['E']==e]
    Xs, Ys = np.asarray(data_slice[f'\\rho_p/\\rho_t']), np.asarray(data_slice[f'\\varphi'])
    order = np.argsort(Xs)
    Xs, Ys = Xs[order], Ys[order]
    enth = 1
    if e == 2:
        ax.plot(Xs[::enth], Ys[::enth], zorder=999, color='black', linestyle='dashed')
    else:
        ax.plot(Xs[::enth], Ys[::enth], zorder=len(nkeys)-idn-1, color=cols[ide])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(3,1e5)
ax.set_ylim(1e-5,2)
ax.set_xlabel('$\\rho_p/\\rho_t$')
ax.set_ylabel('$\\varphi$')
ax.text(0.0125, 0.92, f"$(c)$", transform = ax.transAxes)

ax = axs[3]
csvfile = 'codes/lensing_angle_high_energies_KX_chi_1_rhot_110_vF_1.csv'
data_numeric = pd.read_csv(csvfile)
enes = np.unique(data_numeric['E'])
for (ide,e) in enumerate(enes):
    data_slice = data_numeric[data_numeric['E']==e]
    Xs, Ys = np.asarray(data_slice[f'\\rho_p/\\rho_t']), np.asarray(data_slice[f'\\varphi'])
    order = np.argsort(Xs)
    Xs, Ys = Xs[order], Ys[order]
    enth = 1
    if e == 2:
        ax.plot(Xs[::enth], Ys[::enth], zorder=999, color='black', linestyle='dashed')
    else:
        ax.plot(Xs[::enth], Ys[::enth], zorder=len(nkeys)-idn-1, color=cols[ide])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(3,1e5)
ax.set_ylim(1e-5,2)
ax.set_xlabel('$\\rho_p/\\rho_t$')
ax.set_ylabel('$\\varphi$')
ax.text(0.0125, 0.92, f"$(d)$", transform = ax.transAxes)
plt.tight_layout()
plt.savefig('fig/deflection_angle_high_energy.pdf', bbox_inches='tight', pad_inches=0.01)
plt.savefig('fig/deflection_angle_high_energy.png', dpi=1200, bbox_inches='tight', pad_inches=0.01)

# then create a new image
# adjust the figure size as necessary
figsize = (6.2, 0.2)
fig_leg = plt.figure(figsize=figsize)
ax_leg = fig_leg.add_subplot(111)
# add the legend from the previous axes
ax_leg.legend(*axs[0].get_legend_handles_labels(), loc='center', ncol=4, frameon=False)
# hide the axes frame and the x/y labels
ax_leg.axis('off')
fig_leg.savefig('fig/deflection_angle_high_energy_legend.pdf', bbox_inches='tight', pad_inches=0.01)