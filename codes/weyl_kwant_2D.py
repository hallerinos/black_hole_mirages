import os, shutil
import pandas as pd

import kwant
from kwant_aux import *

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['figure.figsize'] = ((3+3/8), 0.5*(3+3/8))
plt.rc('text.latex', preamble=r'\usepackage{bm,braket}')

# useful definitions for later
sigma_0 = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
sigmas = [sigma_0, sigma_x, sigma_y, sigma_z]
fs = (3+3/8)

# tilt profile
def tilt(pos, t0=1, x0=np.asarray([1e-12,0.]), alpha=0.5):
    r = np.linalg.norm(pos-x0)
    return (t0/r)**alpha * np.asarray(pos-x0)/r

def make_system(sigmas, R=[10,10], k_0=1, v=[0,0], a=1, sym=[], tilt_function=[], t0=1, x0=(0,0), shift_center=(0,0), alpha=0.5):
        a = 1
        primitive_vectors = [(a, 0), (0, a)]
        lat = kwant.lattice.Monatomic(primitive_vectors, norbs=2)
        if sym != []:
            sym_lead = kwant.TranslationalSymmetry(sym*a)
            syst = kwant.Builder(sym_lead)
        else:
            syst = kwant.Builder()
        def square(pos, pos_0=shift_center):
            return all([-r < p < r for (p, r) in zip(pos-pos_0, R)])
        syst[lat.shape(square, shift_center)] = -2/k_0*sigmas[3]
        nnx = 0.5/k_0*sigmas[3] - 0.5j*1*sigmas[1]
        nny = 0.5/k_0*sigmas[3] - 0.5j*1*sigmas[2]
        # space-dependent tilt
        if tilt_function != []:
            def fx(site1, site2):
                s1p, s2p = site1.pos, site2.pos
                (x,y) = 0.5*(s1p+s2p)
                return nnx - 0.5j*sigmas[0]*tilt_function((x,y), t0=t0, x0=x0, alpha=alpha)[0]
            def fy(site1, site2):
                s1p, s2p = site1.pos, site2.pos
                (x,y) = 0.5*(s1p+s2p)
                return nny - 0.5j*sigmas[0]*tilt_function((x,y), t0=t0, x0=x0, alpha=alpha)[1]
            syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = fx
            syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = fy
        # constant tilt
        else:
            syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = nnx
            syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = nny
        return syst

# set the amplitude of the tilt profile
L, k_0, energy, alpha = 20, 1, 0.8, 1
t0s = np.flip(np.asarray([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 10])*L/20)
# t0s = [0]
for t0 in t0s:
    print('-'*100)
    print('Starting simulation')
    print('-'*100)
    # set the parameters
    Rx = L
    Ry = int(Rx/2)
    x0 = np.asarray([int(-0.5*Rx)+0.5,int(0.*Ry)+0.5])
    R = np.asarray([Rx, Ry])
    path_out = f'weyl_plots_alpha_{alpha}/k0_{k_0}_t0_{t0}_E_{energy}_Lx_{2*Rx}_Ly_{2*Ry}'
    if os.path.exists(path_out):
        shutil.rmtree(path_out)
    os.makedirs(path_out)

    # create scattering region
    syst = kwant.Builder()
    syst = make_system(sigmas, R=R, v=[0,0], k_0=k_0, tilt_function=tilt, t0=t0, x0=x0, alpha=alpha)
    # create and attach the leads
    syms = [(-1,0), (1,0), (0,-1), (0,1)]
    shift_centers = [(0,int(-3*Ry/4)), (0,0), (0,0), (0,0)]
    lfracs = [0.2, 1, 1, 1]
    for (id, (s,lf,shift_center)) in enumerate(zip(syms, lfracs,shift_centers)):
        lead = make_system(sigmas, R=lf*R, v=[0,0], k_0=k_0, sym=s, shift_center=shift_center, alpha=alpha)
        syst.attach_lead(lead)
        # fig, ax = plt.subplots(1,1)
        # kwant.plotter.bands(lead.finalized(), show=True, ax=ax)
        # ax.axhline(y=energy, color='black', linestyle='dashdot')
        # plt.savefig(f'{path_out}/lead_band_{id}.png', dpi=300)
        # plt.close()
    fsyst = syst.finalized()

    # plot the tilt profile
    if t0>0:
        fig, ax = plt.subplots(1,1)
        sites = fsyst.sites
        positions = np.asarray([site.pos for site in sites])
        X, Y, U, V, Z = [], [], [], [], []
        df = pd.DataFrame()
        for site in sites:
            position = site.pos
            X.append(position[0])
            Y.append(position[1])
            vxy = tilt(position, t0=t0, x0=x0, alpha=alpha)
            U.append(vxy[0])
            V.append(vxy[1])
            Z.append(np.linalg.norm(vxy))
        df['X'] = X
        df['X'] = df['X']-0.25
        df['Y'] = Y
        df['U'] = U
        df['V'] = V
        df['Z'] = Z
        pivotted = np.asarray(df.pivot('Y','X','Z'))
        extent = [np.min(df['X']),np.max(df['X']),np.min(df['Y']),np.max(df['Y'])]
        imag = ax.imshow(pivotted, extent=extent, cmap='RdBu_r', interpolation='bicubic', vmin=0, vmax=np.max(np.sqrt(t0s)), aspect='auto', origin='lower')
        ax.quiver(X, Y, U, V, units='xy', width=0.07, scale=np.max(df['Z']), pivot='middle', color='white')
        circle = plt.Circle(x0, t0, fill=False, edgecolor='black')
        ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_xlabel('$x/a$')
        ax.set_ylabel('$y/a$')
        ax.set_xlim(-Rx+1,Rx-1.25)
        ax.set_ylim(-Ry+1,Ry-1.25)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(imag, extend='both', pad=0.05, label=f'$|\\bm v|$', cax=cax)
        plt.tight_layout()
        plt.savefig(f'{path_out}/tilt.png', dpi=300)
        plt.close()

    print('System finalized')
    print('-'*100)

    # plot the system
    fig, ax = plt.subplots(1,1)
    kwant.plot(fsyst, ax=ax)
    ax.set_aspect('equal')
    ax.set_xlabel('$x/a$')
    ax.set_ylabel('$y/a$')
    plt.savefig(f'{path_out}/syst.png', dpi=300)
    plt.tight_layout()
    plt.close()

    print('Compute scattering states')
    print('-'*100)

    # get the scattering states
    # propagating_modes, _ = fsyst.leads[0].modes(energy=energy)
    scattering_states = kwant.wave_function(fsyst, energy=energy)
    ss = [scattering_states(i) for i in range(len(fsyst.leads))]

    print('Compute Current')
    print('-'*100)

    # plot the current
    plot_current(fsyst, ss, f"{path_out}/current", num_plots=1)

    print('Simulation done')
    print('-'*100)