import os, shutil
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = ((3+3/8),(3+3/8))
plt.rc('text.latex', preamble=r'\usepackage{bm}')

path_out = f'trajs'
if os.path.exists(path_out):
    shutil.rmtree(path_out)
os.makedirs(path_out)

path_in = "all_average_trajectories_weyl_plots_alpha_1_1"
df_av = pd.read_csv(f'{path_in}.csv')
df_av = df_av.sort_values('pos0_1').reset_index()

fns = np.unique(df_av['fn'])
ps = np.flip(np.unique(df_av['pos0_1']))

df_av['X-pos0_0']  = df_av['X'] - df_av['pos0_0']
df_av['Y-pos0_1']  = df_av['<Y*rho_0>_x/<rho_0>_x'] - df_av['pos0_1']
df_av['dist'] = np.sqrt(df_av['X-pos0_0']**2 + df_av['Y-pos0_1']**2)

minx, maxx = np.min(df_av['X']), np.max(df_av['X'])

Xs, Ys, Zs = [], [], []
for (idp,p) in enumerate(ps):
    print(f'Compute periapsis... {(100*idp)//len(fns)}%')
    # if (100*idp)//len(fns) > 80:
    #     break
    df_av_sel = df_av[df_av['pos0_1'] == p]
    df_av_sel = df_av_sel.sort_values('X').reset_index()

    tp0 = df_av_sel[df_av_sel['X']==minx]
    tp1 = df_av_sel[df_av_sel['X']==minx+5]
    tp2 = df_av_sel[df_av_sel['X']==maxx-5]
    tp3 = df_av_sel[df_av_sel['X']==maxx]
    dy_l = tp1['<Y*rho_0>_x/<rho_0>_x'].iloc[0]-tp0['<Y*rho_0>_x/<rho_0>_x'].iloc[0]
    dx_l = tp1['X'].iloc[0]-tp0['X'].iloc[0]
    dy_r = tp3['<Y*rho_0>_x/<rho_0>_x'].iloc[0]-tp2['<Y*rho_0>_x/<rho_0>_x'].iloc[0]
    dx_r = tp3['X'].iloc[0]-tp2['X'].iloc[0]
    v_l = np.asarray([dx_l, dy_l])
    v_l = v_l/np.linalg.norm(v_l)
    v_r = np.asarray([dx_r, dy_r])
    v_r = v_r/np.linalg.norm(v_r)
    theta = np.arccos(v_l.dot(v_r))
    Xs.append(p)
    Ys.append(np.min(df_av_sel['dist']))
    Zs.append(theta)
    # plt.plot(df_av_sel['X'],df_av_sel['<Y*rho_0>_x/<rho_0>_x'])
    # plt.savefig('testfig.png')
    # exit()
plt.close()
plt.plot(Ys,Zs)
plt.savefig(f'periapsis_deflection_angle_{path_in}.png')
df = pd.DataFrame()
df['X'] = Xs
df['Y'] = Ys
df['Z'] = Zs
df.to_csv(f'periapsis_deflection_angle_{path_in}.csv')