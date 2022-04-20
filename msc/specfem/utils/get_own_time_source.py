import os
import numpy as np
from msc.conv.conv_model import ricker_own

def get_own_ricker(f0, tmax, n_tsteps, fname='time_source_own.txt', dest_dir='./'):
    """ 
    Writes down in a file the Ricker wavelet in an amenable way
    for SPECFEM to read it. 
    
    Args:
        f0       (float): dominant frequency. 
        tmax     (float): maximum simulation time. 
        n_tsteps (int)  : number of time steps in the simulation.
    """
    time = np.linspace(0, tmax, n_tsteps)
    rwav, t_w = ricker_own(time, f0)
    
    fname = os.path.join(dest_dir, fname)
    with open(fname, 'w') as f:
        for i, (ti, wi) in enumerate(zip(t_w, rwav)):
            if i != len(rwav) - 1:
                f.write(str(ti) + ' ' + str(wi) + '\n')
            else:
                f.write(str(ti) + ' ' + str(wi))