"""
NMO correction on multilayer using DTW (Dynamic Time Warping)

- https://gist.github.com/rowanc1/8338665
- https://github.com/rtavenar/blog/blob/main/py/dtw_path.py
"""
import os
import glob
import numpy as np
import pandas as pd
import obspy
import time as t
from tqdm import *

from msc.specfem.utils.read_su_seismograms import read_su_seismogram
from msc.specfem.utils.dtw_matching import dtw_path


def fetch_data(path2output_files: str, verbose: bool):
    """
    Fetch data from forward simulations.

    Args:
        path2output_files (str)     : path to OUTPUT_FILES folder
        verbose           (bool)    : print out the information fetched
    Return:
        data              (2D array): seismic traces
        time              (1D array): simulation time
        dt                (float)   : time step
        offsets           (1D array): offsets positions
    """
    s_traces = obspy.read(
        glob.glob(os.path.join(path2output_files, 'Uz_*.su'))[0])
    time, data = read_su_seismogram(s_traces)
    dt = s_traces[0].stats.delta
    n_samples = data.shape[0]
    n_offsets = data.shape[1]

    stations = pd.read_csv(os.path.join(
        path2output_files, 'STATIONS'), header=None, delim_whitespace=True)
    offsets = stations[2]  # Offsets along X only (Z cte.)

    if verbose:
        print("\nTRACES INFO:")
        print(f"  dt = {dt} s")
        print(f"  N samples = {n_samples}")
        print(f"  Simul time = {dt * n_samples} s")
        print(f"  N offsets = {n_offsets}")
        print(f"  (min, max) offset pos = {min(offsets)}, {max(offsets)} m")
        print(" ")

    return data, time, dt, offsets


def run_nmo_dtw(path2output_files: str, verbose=True):
    """
    Runs NMO correction on the multilayer using DTW (Dynamic Time Warping).
    """
    data, time, dt, offsets = fetch_data(path2output_files, verbose)
    n_offsets = data.shape[1]
    
    # Normal coordinates: receiver right on top of the source
    source_fname = os.path.join(path2output_files, 'SOURCE')
    with open(source_fname, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if l[:2] == 'xs':
                xs = float(l.split('=')[1].split('#')[0].strip())
            if l[:2] == 'zs':
                zs = float(l.split('=')[1].split('#')[0].strip())

    x_offsets = offsets - xs
    mid = np.argmin(np.abs(x_offsets))
    mid_trace = data[:, mid]
    
    # NMO correction
    start_t = t.time()
    print('\nRUNNING DTW-NMO CORRECTION')
    data_dtw = data.copy()
    for k in tqdm(range(mid+1, n_offsets)):
        path_R, = dtw_path(data_dtw[:, k-1], data_dtw[:, k])
        path_L, = dtw_path(data_dtw[:, 2*mid-(k-1)], data_dtw[:, 2*mid-k])
        for i, j in path_R:
            data_dtw[i, k] = data[j, k]
        for i, j in path_L:
            data_dtw[i, 2*mid-k] = data[j, 2*mid-k]
    
    elapsed_time = t.time() - start_t
    print(f"Elapsed time: {elapsed_time/60:.3f} mins")
    
    collect_results = {
        'cmp'      : data,
        'nmo'      : data_dtw, 
        'time'     : time, 
        'dt'       : dt, 
        'x_offsets': x_offsets,
        'mid_trace': mid_trace
    }
    
    return collect_results
    
