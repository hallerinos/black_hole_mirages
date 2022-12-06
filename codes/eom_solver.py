import numpy as np
from sympy import

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = ((3+3/8),(3+3/8))
plt.rc('text.latex', preamble=r'\usepackage{bm}')

def u(r, r0=1, alpha=1, vF=1):
    r_norm = np.linalg.norm(r)
    r_hat = r/r_norm
    return vF*(r0/r_norm)**alpha*r_hat


