import os
import glob
import argparse
import numpy as np
import pandas as pd
import meshio
import matplotlib.pyplot as plt

from msc.specfem.utils.material_file import read_material_file

def tilted_boundary(x, angle_deg, iface, ztop_bot):
    x_dist = np.max(x) - np.min(x)
    z_top, z_bot = ztop_bot
    if iface == z_top:
        return z_top * np.ones(len(x))
    elif iface == z_bot:
        return z_bot * np.ones(len(x))
    else:
        return iface - (x - 0.5*x_dist) * np.tan(angle_deg * np.pi/180)
    

def create_tomo_tilted_2Dfile(xmin_max: tuple, mesh_res: tuple, uneven: dict, angle_deg: float, path2mesh='./MESH', dest_dir='./DATA', invertz=False, save_xyz=True):
    """
    Creates the tomography file .xyz of the tilted multilayer model. 

    Args:
        xmin_max  (tuple)  : size of the mesh in the format (xmin, xmax).
        mesh_res  (tuple)  : mesh resolution in the format (nx, nz)
        uneven    (dict)   : dictionary with the uneven layers.
        angle_deg (float)  : angle to be tilted in degrees.
        path2mesh (str)    : path to the MESH folder. Defaults to './MESH'.
        dest_dir  (str)    : path to the destination folder for the tomo file. Defaults to './DATA'.
        invertz   (bool)   : if True invert z so that it goes from ztop to zbot. Defaults to False.
        
    Returns:
        vp_2d  (np.ndarray): 2D array with the vp model
        rho_2d (np.ndarray): 2D array with the density model
    """
    zbot = -sum(uneven.values())
    ztop = 0.0
    xmin, xmax = min(xmin_max), max(xmin_max)
    
    nx, nz = mesh_res
    dx = (xmax - xmin)/nx
    dz = (ztop - zbot)/nz
    
    z = np.linspace(zbot, ztop, nz) if not(invertz) else np.linspace(ztop, zbot, nz)
    x = np.linspace(xmin, xmax, nx)
    
    d2v, rho, vp, vs = read_material_file(path2mesh)
    rho, vp, vs = rho.values, vp.values, vs.values
    n_layers = len(vp)  
    
    # Multilayer
    L_mult = uneven['L_mult']
    N_mult = n_layers - 2
    l_size = L_mult/N_mult
    
    # Intefraces from bottom to top
    interfaces = [ztop]
    for dom_id in d2v:
        size_ = l_size
        if dom_id in uneven:
            size_ = uneven[dom_id]
        interfaces += [interfaces[-1] - size_]
    interfaces[-1] = zbot
    
    # 2D models
    vp_2d = np.zeros((nz, nx))
    vs_2d = np.zeros((nz, nx))
    rho_2d = np.zeros((nz, nx))
    for i, (sup_lim, inf_lim) in enumerate(zip(interfaces[:-1], interfaces[1:])):
        zsup_arr = tilted_boundary(x, angle_deg, sup_lim, (ztop, zbot))
        zinf_arr = tilted_boundary(x, angle_deg, inf_lim, (ztop, zbot))
        for j, (zsup, zinf) in enumerate(zip(zsup_arr, zinf_arr)):
            mask_ = (zinf <= z) & (z <= zsup)
            vp_2d[mask_, j] = vp[i]
            rho_2d[mask_, j] = rho[i]
            
    # Collect data in the correct format
    xcoords = []
    zcoords = []
    collect_fields = {'vp': [], 'vs': [], 'rho': []}
    for i, zval in enumerate(z):
        vp_i = vp_2d[i,:]
        vs_i = vs_2d[i,:]
        rho_i = rho_2d[i,:]
        for j, xval in enumerate(x):
            xcoords.append(xval)
            zcoords.append(zval)
            collect_fields['vp'].append(vp_i[j])
            collect_fields['rho'].append(rho_i[j])
            collect_fields['vs'].append(vs_i[j])
    
    assert len(xcoords) == len(zcoords), 'Mismatch in sizes!'
    
    if save_xyz:
        xyz_fname = 'profile.xyz'
        with open(os.path.join(dest_dir, xyz_fname), 'w') as f:
            print(f'Name of the file: {f.name}')
            f.write(f'{xmin} {zbot} {xmax} {ztop}\n')
            f.write(f'{dx} {dz}\n')
            f.write(f'{nx} {nz}\n')
            f.write(f'{min(vp)} {max(vp)} {min(vs)} {max(vs)} {min(rho)} {max(rho)}\n')
            for j in range(len(xcoords)):
                f.write(f"{xcoords[j]} {zcoords[j]} {collect_fields['vp'][j]} {collect_fields['vs'][j]} {collect_fields['rho'][j]}\n")
    
    return vp_2d, rho_2d