from os import listdir
import numpy as np
import pandas as pd

# use a function find_csv to find and sort .csv files containing flow data 
def find_csv(path, suffix=".csv"):
    files = listdir(path)
    snaps = []
    for name in files:
        if name.endswith(suffix):
            snaps.append(name)
    return sorted(snaps, key=(len))

def xmatrix(snaps):
    # extract coordinates, so the mesh can be created and values mapped
    X = np.array(pd.read_csv(snaps[1])["Points:0"])
    Y = np.array(pd.read_csv(snaps[1])["Points:1"])
    Z = np.array(pd.read_csv(snaps[1])["Points:2"])
    coord = np.array([X, Y, Z]).T
    
    # determine the number of rows and columns to prealocate velocity matrices
    col = len(snaps)
    row = len(pd.read_csv(snaps[1])["U:0"])
    velx = np.zeros([row,col]) # velocity in X-direction
    vely = np.zeros([row,col]) # velocity in Y-direction
    velz = np.zeros([row,col]) # velocity in Z-direction
    vorz = np.zeros([row,col]) # velocity in Z-direction
    
    # extract the velocities
    for i in range(len(snaps)):
        velx[:,i] = pd.read_csv(snaps[i])["U:0"]
        vely[:,i] = pd.read_csv(snaps[i])["U:1"]
        velz[:,i] = pd.read_csv(snaps[i])["U:2"]
        vorz[:,i] = pd.read_csv(snaps[i])["Vorticity:2"]
    
    return coord, velx, vely, velz, vorz

def extract_snaps(array, mpl):
    snaps = np.arange(0, np.size(array, axis = 1),mpl)
    X = np.zeros((np.size(array, axis=0),len(snaps)))
    for i in range(len(snaps)):
        X[:,i]=array[:,snaps[i]]
    return X


# the reconstruction is done according to the expression:
# Xdmd = PHI*b*e^(lambda*t)

# the number of snapshots to be reconstructed equals number of columns of X
# keep in mind that, prediction is possible by extending the snapshots 
# vector! e.g. instead of using below expression to determine the
# number of snapshots in provided dataset, we can simply define a number 
# that is larger than the number of snapshots in our dataset, hence
# obtaining the future state of considered system.

def dmd_recon(X, exact_modes, exp_eigs, ampl, R, DT):
    snaps = np.size(X,axis=1)
    # - matrix that will hold the expression b*e^(lambda*t) is defined below
    # - first dimension is 'r', becasue the lambda also has 'r' dimension!
    dynamics = np.zeros((R,snaps),dtype = 'complex_')
    # time vector, since it starts from zero, last snapshot is subtracted
    # the number of snapshots is multiplied by dt. the get the actual time that
    # passed between first and the last snapshot
    t = np.arange(0,snaps,1)*DT
    for i in range(snaps):
        dynamics[:,i] = ampl*np.exp(exp_eigs*t[i])
        
    Xdmd = exact_modes @ dynamics
    return Xdmd, dynamics
    