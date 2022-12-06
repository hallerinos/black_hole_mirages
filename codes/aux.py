fig, ax = plt.subplots(1,1)
current_one_way = np.empty(fsyst.graph.num_edges // 2)
dim = len(fsyst.sites[0].pos)
hops = np.empty((fsyst.graph.num_edges // 2, 2, dim))
seen_hoppings = dict()
kprime = 0
for k, (i, j) in enumerate(fsyst.graph):
    if (j, i) in seen_hoppings:
        current_one_way[seen_hoppings[j, i]] -= current[k]
    else:
        current_one_way[kprime] = current[k]
        hops[kprime][0] = fsyst.sites[j].pos
        hops[kprime][1] = fsyst.sites[i].pos
        seen_hoppings[i, j] = kprime
        kprime += 1
current = current_one_way / 2
hops, lead_hops_slcs = kwant.plotter.sys_leads_hoppings(fsyst, 0)
sites, lead_sites_slcs = kwant.plotter.sys_leads_sites(fsyst, 0)
sites_pos = kwant.plotter.sys_leads_pos(fsyst, sites)
cm = plt.get_cmap('RdBu') 
cabs = np.max(np.abs(current))
cNorm  = colors.Normalize(vmin=-cabs, vmax=cabs)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
current_colors = [list(scalarMap.to_rgba(c)) for c in current]

df = pd.DataFrame()
X, Y, Z1, Z2 = [], [], [], []
for (idh,h) in enumerate(hops):
    XY1 = np.asarray(sites_pos[h[0][0]])
    XY2 = np.asarray(sites_pos[h[0][1]])
    Rvec = XY2 - XY1

    X.append(XY1[0]); Y.append(XY1[1]);
    if np.linalg.norm(Rvec - [1, 0]) < 1e-8:
        Z1.append(-current[idh]); Z2.append(None)
    elif np.linalg.norm(Rvec - [0, 1]) < 1e-8:
        Z1.append(None); Z2.append(-current[idh]) 
    else:
        print('ups')
df['X']=X; df['Y']=Y; df['Z1']=Z1; df['Z2']=Z2;
df = df.groupby(['X', 'Y']).agg('sum').reset_index()
df['Zabs'] = np.sqrt(df['Z1']**2 + df['Z2']**2)
pivotted = df.pivot('Y', 'X', 'Zabs')
extent = [np.min(df['X']), np.max(df['X']), np.min(df['Y']), np.max(df['Y'])]
vabs = np.max(df['Zabs'])
imag = ax.imshow(pivotted, extent=extent, origin='lower', cmap='Greys', interpolation='nearest')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(imag, cax=cax)

xi = np.linspace(df['X'].min(), df['X'].max(), 200)
yi = np.linspace(df['Y'].min(), df['Y'].max(), 200)
X, Y = np.meshgrid(xi, yi)
U = interpolate.griddata((df['X'], df['Y']), df['Z1'], (X, Y), method='nearest')
V = interpolate.griddata((df['X'], df['Y']), df['Z2'], (X, Y), method='nearest')
UVabs = np.sqrt(U**2+V**2)
df_max = df[df['Zabs']==df['Zabs'].max()].iloc[0]
# ax.streamplot(X, Y, U, V, color='white', start_points=[[df_max['X'],df_max['Y']]])
circle = plt.Circle(params['pos0'], params['r0'], color='black', fill=False)
ax.add_patch(circle)
ax.streamplot(X, Y, U, V, color='white')
ax.set_xlim(-params['Ls'][0],params['Ls'][0])
ax.set_ylim(-params['Ls'][1],params['Ls'][1])
# vxs, vys = [], []
# for (x,y) in zip(df['X'], df['Y']):
#     pos = np.asarray([x,y])
#     vvec = tilt(pos, **params)
#     vxs.append(vvec[0])
#     vys.append(vvec[1])
# df['vx'] = vxs; df['vy'] = vys; df['vabs'] = np.sqrt(df['vx']**2+df['vy']**2)
# pivotted = df.pivot('Y', 'X', 'vabs')
# ax.imshow(pivotted, extent=extent, origin='lower', cmap='Greys', interpolation='bicubic')
# ax.scatter(df['X'],df['Y'],c=df['vabs'])
# levels = [0.1, 1, 10, 100]
# CS = ax.contour(pivotted, levels, extent=extent, origin='lower')
# ax.clabel(CS)
plt.savefig(f'{path_out}/my_current.png',dpi=300)
plt.close()