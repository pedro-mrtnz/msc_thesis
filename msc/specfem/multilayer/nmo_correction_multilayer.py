import os
import glob
import numpy as np
import obspy
import pandas as pd
import time as t

from msc.specfem.multilayer.create_tomography_file import read_material_file
from msc.specfem.utils.read_su_seismograms import read_su_seismogram
from msc.specfem.utils.nmo_correction import nmo_correction


def fetch_data(path2output_files: str, verbose: bool):
    """
    Fetch data from forward time simulations.
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
    

def run_nmo(path2output_files: str, path2mesh: str, zmin_max: tuple, uneven_dict: dict, verbose=True):
    """
    Run the NMO correction on the multilayer data. 
    """
    data, time, dt, offsets = fetch_data(path2output_files, verbose)
    n_samples = data.shape[0]
        
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
    
    zmin, zmax = zmin_max
    if uneven_dict is not None:
        L_mult = uneven_dict['L_mult']
    else:
        L_mult = zmax - zmin
    nz = n_samples
    dz = (zmax - zmin)/(nz - 1)
    zi = np.linspace(zmax, zmin, nz)

    d2v = read_material_file(path2mesh)[0]
    N = len(d2v) - 2 if uneven_dict is not None else len(d2v)
    dom_size = L_mult/N
    dom_intervals = [0.0]
    for dom_id in d2v.keys():
        size_ = dom_size
        if dom_id in uneven_dict:
            size_ = uneven_dict[dom_id]
        dom_intervals += [dom_intervals[-1] - size_]
    
    dom_in_zi = np.zeros_like(zi).astype('int32')
    for i, (sup_lim, inf_lim) in enumerate(zip(dom_intervals[:-1], dom_intervals[1:])):
        mask = (zi <= sup_lim) & (zi >= inf_lim)
        dom_in_zi[mask] = i + 1

    # Time-depth relationship (we only need the velocities tho)
    nmo_times = []
    nmo_vels = []
    for i, dom in enumerate(dom_in_zi):
        vel = d2v[dom]['vp']
        if i == 0:
            t1 = 2*(zmax - zi[i])/vel
            nmo_times.append(t1)
        else:
            t2 = t1 + 2*(zi[i-1] - zi[i])/vel
            nmo_times.append(t2)
            t1 = t2
        nmo_vels.append(vel)

    # NMO CORRECTION
    start = t.time()
    nmo = nmo_correction(data, dt, x_offsets, nmo_vels)
    elapsed_time = t.time() - start
    print(f"NMO-correction took {elapsed_time/60:.3f} mins")
    
    collect_results = {
        'cmp'      : data,
        'nmo'      : nmo,
        'time'     : time,
        'nmo_times': np.array(nmo_times),
        'nmo_vels' : np.array(nmo_vels),
        'x_offsets': x_offsets
    }
    
    return collect_results