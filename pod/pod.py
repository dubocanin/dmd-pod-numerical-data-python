# -*- coding: utf-8 -*-

# USEFUL LIBRARIES
import matplotlib as mpl
import modred as mr
import numpy as np
import scipy.io

from plotting import plot_modes_grided as plmod
from plotting import plot_variance_pod as plvar
from plotting import plot_visual_inspect as plvin
from mytools import pod_recon

#........................... LOAD CUSTOM COLORMAPS ............................
custom_colormap = scipy.io.loadmat('C:/snowflake/colormaps/CCnash.mat')
C = custom_colormap['CCnash']
CC = mpl.colors.ListedColormap(C)


#............................... LOAD THE DATA ................................
x = np.load('C:/snowflake/data/book_data.npy')
ROW = 199
COL = 449

#.................. COMPUTING AND SUBTRACTIONG THE AVERAGE ....................
# compute avarage (of each row), and subtract it, this is done to mean center
# the data!
avg = np.mean(x, axis=1)
xavg = np.zeros((np.size(x, axis=0),np.size(x, axis=1)))
for i in range(np.size(x,axis=1)):
    xavg[:,i] = avg
xt = x-xavg

#............................. THE POD ANALYSIS ...............................
NUM_MODES = 25
POD_res = mr.compute_POD_arrays_direct_method(
    xt, list(mr.range(NUM_MODES)))
modes = POD_res.modes
eigvals = POD_res.eigvals

#............................... VIZUALIZATION ................................
plvar(eigvals, NUM_MODES)

plmod(modes, ROW, COL, CC, 8)
# plt.savefig('C:/Users/Nebojsa/path/figure.png')

#............................... RECONSTRUCTION ...............................
[xpod, aj] = pod_recon(x, xt, modes, NUM_MODES)
xpod = xpod + xavg
plvin(x, xpod, ROW, COL, CC)

#........................... MAPE ERROR ESSTIMATION ...........................
MAPE = np.mean(abs((x-xpod)/x))
print('The value of mean absolute percentage error [MAPE] is: ' + str(MAPE) + '[%]')
