import os
import glob
import numpy as np
import obspy
import pandas as pd
import time as t

from msc.specfem.multilayer.create_tomography_file import read_material_file
from msc.specfem.utils.read_su_seismograms import read_su_seismogram
from msc.specfem.utils.nmo_correction import nmo_correction
from msc.conv.conv_model import *


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
    x_offsets = offsets - xs

    d2v, rho, vp, vs = read_material_file(path2mesh)
    rho, vp, vs = rho.values, vp.values, vs.values
    n_layers = len(vp)
    
    zbot, ztop = zmin_max
    nz = n_samples
    dz = (ztop - zbot)/nz
    z = np.linspace(ztop, zbot, nz)
    
    # Multilayer 
    if uneven_dict is not None:
        L_mult = uneven_dict['L_mult']
        N_mult = n_layers - 2  # substract top and bottom layers
    else:
        L_mult = ztop - zbot
        N_mult = n_layers
    layer_size = L_mult/N_mult
    
    interfaces = [0.0]
    for dom_id in d2v:
        size_ = layer_size
        if (uneven_dict != None) and (dom_id in uneven_dict):
            size_ = uneven_dict[dom_id]
        interfaces += [interfaces[-1] - size_]
    interfaces[-1] = zbot
    
    # Depth domain
    vp_z = np.zeros(nz)
    for i, (sup_lim, inf_lim) in enumerate(zip(interfaces[:-1], interfaces[1:])):
        mask = (inf_lim <= z) & (z <= sup_lim)
        vp_z[mask] = vp[i]

    # Time domain
    vp_t, twt_t = depth2time(vp_z, vp_z, dt, dz, npts=n_samples, return_t=True)
    
    tdiff = np.diff(twt_t)
    vp2_t = []
    for i in range(1, n_samples):
        dt_i = twt_t[i] - twt_t[i-1]
        vp2_t.append(vp_t[i]**2 * dt_i)
    vp_rms = np.sqrt(np.cumsum(vp2_t)/np.cumsum(tdiff))
    vp_rms = np.concatenate(([vp_t[0]], vp_rms))
    
    # Muting
    interface1 = -uneven_dict[1]
    t0 = 2*interface1/vp[0]
    mask = lambda t0, x: time < np.sqrt(t0**2 + (x/vp[0])**2)
    data_m = data.copy()
    for i, x in enumerate(x_offsets):
        mask_ = mask(t0, x)
        data_m[mask_,i] = 0.0

    # NMO CORRECTION
    start = t.time()
    nmo = nmo_correction(data_m, dt, x_offsets, vp_rms)
    elapsed_time = t.time() - start
    print(f"NMO-correction took {elapsed_time/60:.3f} mins")
    
    collect_results = {
        'cmp'      : data_m,
        'nmo'      : nmo,
        'sim_time' : time,
        'twt_t'    : np.array(twt_t),
        'vp_t'     : np.array(vp_t),
        'vp_rms'   : vp_rms,
        'x_offsets': x_offsets
    }
    
    return collect_results