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
from matplotlib.patches import ArrowStyle

def filled_arc(center, radius, theta1, theta2, ax, color):

    circ = patches.Wedge(center, radius, theta1, theta2, facecolor=color, alpha=0.6, edgecolor='none')
    pt1 = (radius * (np.cos(theta1*np.pi/180.)) + center[0],
           radius * (np.sin(theta1*np.pi/180.)) + center[1])
    pt2 = (radius * (np.cos(theta2*np.pi/180.)) + center[0],
           radius * (np.sin(theta2*np.pi/180.)) + center[1])
    pt3 = center
    ax.add_patch(circ)


mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = (6.2,6.2/2)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

csvfile = 'codes/deflection_angle_orbits.csv'

r0 = 3

fig, axs = plt.subplots(1, 2)
axs = axs.ravel()

lbls = 'abcdefghijklmnopqrstuvwxyz'
pranges = [20, 4, 1.3, 0.5]
szes = 0.04*np.asfarray(pranges)
arrowoffs=[[0,60,160,33],[300,20,600,93],[400,30,900,35]]
arrowoffs=[[0,180,130,12],[0,250,400,98],[0,30,150,77],[0,20,400,65]]

cols = []
data = pd.read_csv(csvfile)
circle = plt.Circle([0,0], 3.0/2, facecolor='black', alpha=0.2, edgecolor='none')
axs[0].add_patch(circle)
circle = plt.Circle([0,0], 1, facecolor='black', alpha=0.2, edgecolor='none')
axs[0].add_patch(circle)
circle = plt.Circle([0,0], 0.12, facecolor='black', edgecolor='none', zorder=10)
axs[0].add_patch(circle)
axs[0].set_aspect('equal')

axs[0].plot([0,0],[-100,100], color='black', linewidth=0.5, zorder=0)
axs[0].plot([-100,100],[0,0], color='black', linewidth=0.5, zorder=0)

nkeys = np.asarray(range(len(data.keys())//3))+1
for (idn,n) in enumerate(nkeys):
    data_sel = data[data[f'X{n}'].notna()]
    Xs, Ys = np.asfarray(data_sel[f'X{n}']), np.asfarray(data_sel[f'Y{n}'])

    p = axs[0].plot(Xs, Ys, zorder=10, color='black')
    col = p[-1].get_color()
    cols.append(col)

    func = lambda x, a, b: a + b*x

    s2 = -200
    Xs_fit, Ys_fit = Xs[s2:], Ys[s2:]

    best_fit_ab, covar = curve_fit(func, Xs_fit, Ys_fit)
    
    sigma_ab = np.sqrt(np.diagonal(covar))
    a = best_fit_ab[0]
    b = best_fit_ab[1]
    
    p = axs[0].plot([-100,100], a + b*np.asfarray([-100,100]), linestyle='dashed', zorder=10)
    phi1 = 53.9
    filled_arc([2.93,-4.1], 6.5, 0, phi1, axs[0], p[-1].get_color())
    axs[0].text(5, 1, "$\\phi_2$", color=p[-1].get_color())

    s1 = 200
    Xs_fit, Ys_fit = Xs[:s1], Ys[:s1]
    
    best_fit_ab, covar = curve_fit(func, Xs_fit, Ys_fit)
    
    sigma_ab = np.sqrt(np.diagonal(covar))
    a = best_fit_ab[0]
    b = best_fit_ab[1]
    
    p = axs[0].plot([-100,100], a + b*np.asfarray([-100,100]), linestyle='dashed', zorder=10)
    phi2 = 17.5
    filled_arc([2.93,-4.1], 5, 0, phi2, axs[0], p[-1].get_color())
    filled_arc([2.93,-4.1], 2.9, 180+phi2, 180+phi1, axs[0], 'black')
    axs[0].text(2, -6.5, "$\\varphi$", color='black')
    axs[0].text(6, -5, "$\\phi_1$", color=p[-1].get_color())
distsxy = np.sqrt(data['X1']**2 + data['Y1']**2)
data_sel = data[np.sqrt(data['X1']**2 + data['Y1']**2)==np.min(distsxy)]
axs[0].annotate("", xy=(-0.01, -0.01), xytext=(data_sel['X1'], data_sel['Y1']), textcoords=axs[0].transData, arrowprops=dict(arrowstyle=ArrowStyle("|-|", widthA=0.2, angleA=0, widthB=0.2, angleB=0)))
axs[0].text(0.0125, 0.94, f"$(a)$", transform = axs[0].transAxes)
axs[0].text(1.3, -1.3, "$\\rho_p$", transform=axs[0].transData)
axs[0].set_xlim(-4,+10)
axs[0].set_ylim(-10,+6)
axs[0].set_xlabel('$x/\\rho_t$')
axs[0].set_ylabel('$y/\\rho_t$')

csvfile = 'codes/lensing_angle_analytic_nint.csv'
data_analytic_nint = pd.read_csv(csvfile)
csvfile = 'codes/lensing_angle_analytic_approx.csv'
data_analytic_approx = pd.read_csv(csvfile)
keys =np.unique([s[-1] for s in np.asarray(data_analytic_nint.keys())])
print(keys)
for k in keys:
    Xs, Ys = np.asarray(data_analytic_nint[f'\\rho_m{k}']), np.asarray(data_analytic_nint[f'\\varphi{k}'])
    order = np.argsort(Xs)
    Xs, Ys = Xs[order], Ys[order]
    p = axs[1].plot(Xs, np.real(Ys), alpha=0.5, zorder=1)
    col = p[-1].get_color()

    Xs, Ys = np.asarray(data_analytic_approx[f'\\rho_m{k}']), np.asarray(data_analytic_approx[f'\\varphi{k}'])
    order = np.argsort(Xs)
    Xs, Ys = Xs[order], Ys[order]
    axs[1].plot(Xs, np.real(Ys), linestyle='dotted', color=col, zorder=1)

csvfile = 'codes/lensing_angle_numeric.csv'
data_numeric = pd.read_csv(csvfile)
keys =np.unique([s[-1] for s in np.asarray(data_numeric.keys())])
for (idk,k) in enumerate(keys):
    Xs, Ys = np.asarray(data_numeric[f'\\rho_m{k}']), np.asarray(data_numeric[f'\\varphi{k}'])
    order = np.argsort(Xs)
    Xs, Ys = Xs[order], Ys[order]
    enth = 20
    axs[1].scatter(Xs[::enth], Ys[::enth], zorder=1, label=f'$\\alpha={(idk+1)/2}$')

axs[1].plot([], [], color='black', zorder=1, label='$\\rm Eq.\\ (11)$')
axs[1].plot([], [], linestyle='dotted', color='black', zorder=1, label='$\\rm Eq.\\ (12)$')

axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_ylim(1e-6, 2*np.pi)
axs[1].set_xlabel('$\\rho_p/\\rho_t$')
axs[1].set_ylabel('$\\varphi$')
axs[1].text(0.12, 0.94, f"$(b)$", transform = axs[1].transAxes)
axs[1].legend(fancybox=False, frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig('fig/deflection_angle.pdf', bbox_inches='tight', pad_inches=0.01)
plt.savefig('fig/deflection_angle.png', dpi=1200, bbox_inches='tight', pad_inches=0.01)