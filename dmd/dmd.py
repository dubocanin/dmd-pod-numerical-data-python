# -*- coding: utf-8 -*-

"""USEFUL LIBRARIES"""
# import scipy.io
import matplotlib.pyplot as plt
import matplotlib as mpl

import modred as mr
import numpy as np
import scipy.io

from plotting import plot_modes_grided2 as plmod
from plotting import plot_fdom_cpln as plfdcp
from plotting import plot_relabs_rate as plrea
from plotting import plot_mode_osc as plmos
from plotting import plot_visual_inspect as plvin
from mytools import dmd_recon
# from mytools import extract_snaps
# from matplotlib import cm
# from scipy.interpolate import griddata


#........................... LOAD CUSTOM COLORMAPS ............................
colormap1 = scipy.io.loadmat('C:/snowflake/colormaps/CCnash.mat')
C1 = colormap1['CCnash']
CM1 = mpl.colors.ListedColormap(C1)
# custom colormap 2 - for complex plane i.e. spectrum plot
colormap2 = scipy.io.loadmat('C:/snowflake/colormaps/CCnash2.mat')
C2 = colormap2['CCnash2']
CM2 = mpl.colors.ListedColormap(C2)

#.......................... LOAD THE NECESSARY DATA ...........................
# load the data into data_array...data array columns hold vorticity values as 
# extracted from .csv files

x = np.load('C:/snowflake/data/book_data.npy')

ROW = 199
COL = 449
MPL = 10
SIM_DT = 0.02
DT = MPL*SIM_DT

# define matrix X, and time advanced X using the raw data
# x = extract_snaps(vort, MPL)

LENGTH = np.size(x,axis=1)
X1 = x[:,0:LENGTH-1]
X2 = x[:,1:LENGTH]

#.............................. THE DMD ANALYSIS ..............................
# by setting the atol = 3, computation will be truncated to rank 21
# this can be checked by examining the correlation array

DMD_res = mr.compute_DMD_arrays_direct_method(X1, X2, atol=3)
R = np.size(DMD_res.correlation_array_eigvals) # truncation

#.................................. DMD MODES .................................
# modes are flipped since they are ordred by modred, the last one is average
phi = DMD_res.exact_modes

# initial conditions and amplitudes
xo = x[:,0]
res = np.linalg.lstsq(phi,xo,rcond=None)
b = res[0]
B = np.abs(b)       # modulus of amplitudes

# since modes are computed as complex-conjugated pairs, every odd or even
# is ploted!
ID = np.array([20,19,17,15,13,11,9,7,5,3,1])

plmod(phi.real, ROW, COL, CM1, ID[1:9])
# plmod(phi.imag, ROW, COL, CC, mod_id)

# save results
# plt.savefig('C:/Users/Nebojsa/path/figure.png')

#............................... DMD EIGENVALUES ..............................
eigs = DMD_res.eigvals          # eigenvalues
exp_eigs = np.log(eigs)/DT      # exponential eigenvalues

#..................... OTHER INTERPRETATION PARAMETERS.........................
omega = exp_eigs.imag
freq = omega/(2*np.pi)
sigma = exp_eigs.real               # absolute damping rate
zeta = np.abs(sigma/omega)          # relative damping rate, the modulus of it
# is unnecessary, it is done as suggested by Robin De S.

#........................... RESULTS INTERPRETATION ...........................
# plot complex plane and frequency amplitude plot
plfdcp(freq, b, eigs, ID, ID[0:5], CM2)

# plot absolute and relative damping rate
plrea(ID[1:11], ID[1:11], sigma, zeta)

# mode oscilations over time
plmos(sigma, omega, ID[1:9])

#............................... RECONSTRUCTION ...............................

[xdmd, dynamics] = dmd_recon(x, phi, exp_eigs, b, R, DT)
xdmd = xdmd.real
# visual inspection of data and reconstructed data
plvin(x, xdmd, ROW, COL, CM1)


#........................... MAPE ERROR ESSTIMATION ...........................
MAPE = np.mean(abs((x-xdmd)/x))
print('[MAPE] is: ' + str(MAPE) + '[%]')