import numpy as np
import json
import os, shutil, sys

Ls = [900]
sf =''
if len(sys.argv)>1:
    sf = sys.argv[1]
path_out = f'codes/configs{sf}/'
if os.path.exists(path_out):
    shutil.rmtree(path_out)
os.makedirs(path_out)
cfg_num = 0
for L in Ls:
    for frac in np.linspace(-1,1,32):
        params = {
                'Ls': [L, L], 'r0': L//10, 'pos0': [-L//2, int(1.5*L+frac*L)], 'alpha': 0.5, 'fd': 0.9, 'fu': 0.6, 'k0': 1, 'energy': 0.8
                }
        json_string = json.dumps(params)
        # Using a JSON string
        with open(f'{path_out}/cfg_'+str(cfg_num).zfill(4)+'.json', 'w') as outfile:
            outfile.write(json_string)
        cfg_num += 1