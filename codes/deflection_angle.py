import os, shutil
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = ((3+3/8),3/4*(3+3/8))
plt.rc('text.latex', preamble=r'\usepackage{bm}')

path_out = f'trajs'
if os.path.exists(path_out):
    shutil.rmtree(path_out)
os.makedirs(path_out)

df = pd.read_csv('periapsis_deflection_angle_all_average_trajectories_weyl_plots_alpha_1_1.csv')
fig, ax = plt.subplots(1,1)
df = df.sort_values('Y').reset_index()
df_cut = df[160**2/df['Y']**2 >= 0.14]
df = df[160**2/df['Y']**2 < 0.14]
ax.plot(160**2/df['Y']**2, df['Z']/np.pi, label='$\\alpha=1$')
ax.scatter(160**2/df['Y']**2, df['Z']/np.pi, marker='x')

ax.plot(160**2/df_cut['Y']**2, df_cut['Z']/np.pi, color='gray')
ax.scatter(160**2/df_cut['Y']**2, df_cut['Z']/np.pi, marker='x', color='gray')

df = pd.read_csv('periapsis_deflection_angle_all_average_trajectories_weyl_plots_alpha_0.5_1.csv')
# fig, ax = plt.subplots(1,1)
# df = df.sort_values('Y').reset_index()
# ax.plot((80/df['Y']), df['Z']/np.pi, label='$\\alpha=0.5$')
# ax.scatter((80/df['Y']), df['Z']/np.pi, marker='x')

df2 = pd.read_csv('periapsis_deflection_angle_all_average_trajectories_weyl_plots_alpha_0.5_2.csv')

df = pd.concat((df, df2)).reset_index()
# fig, ax = plt.subplots(1,1)
df = df.sort_values('Y').reset_index()
df_cut = df[160**2/df['Y']**2 >= 0.08]
df = df[160**2/df['Y']**2 < 0.08]
ax.plot((80/df['Y']), df['Z']/np.pi, label='$\\alpha=0.5$')
ax.scatter((80/df['Y']), df['Z']/np.pi, marker='x')
ax.plot(80/df_cut['Y'], df_cut['Z']/np.pi, color='gray')
ax.scatter(80/df_cut['Y'], df_cut['Z']/np.pi, marker='x', color='gray')
ax.set_xlim(0,0.38)
ax.set_ylim(0,0.2)
ax.set_xlabel('$(r_H/r_m)^{2\\alpha}$')
ax.set_ylabel('$\\theta/\\pi$')

ax.plot([0.006,0.4006], [0,0.55], 'k:')
ax.plot([0.03,0.2], [0,0.2], 'k:')
ax.legend()
plt.savefig('deflection_angle.png', dpi=300, bbox_inches='tight', pad_inches=0.02)