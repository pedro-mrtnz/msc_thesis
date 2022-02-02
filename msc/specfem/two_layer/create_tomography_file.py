import os
import numpy as np
from msc.specfem.multilayer.create_tomography_file import plot_field

def create_tomo_1Dfile(mesh_size, mesh_res, two_layers_dict, dest_dir='./DATA'):
    """
    Creates 1D tomography file for the simple 2 layer model. 
    
    Args:
        mesh_size       (list) : mesh dimesnions as [(xmin, xmax), (zmin, zmax)]
        mesh_res        (tuple): mesh resolution (nx, nz)
        two_layers_dict (dict) : contains rho and vp for the two layers. 
        dest_dir        (str)  : destination folder where to save .xyz tomo file.
    """
    xmin_max, zmin_max = mesh_size[0], mesh_size[1]
    xmin, xmax = min(xmin_max), max(xmin_max)
    zmin, zmax = min(zmin_max), max(zmin_max)
    
    nx, nz = mesh_res 
    dx = (xmax - xmin)/(nx - 1)
    dz = (zmax - zmin)/(nz - 1)
    xi = np.linspace(xmin, xmax, nx)
    zi = np.linspace(zmin, zmax, nz)
    
    interface = zmax - 0.5*(zmax - zmin)
    rho = np.zeros(nz)
    vp  = np.zeros(nz)
    for i in range(nz):
        if zi[i] <= interface:
            rho[i] = two_layers_dict[1][0]
            vp[i] = two_layers_dict[1][1]
        else:
            rho[i] = two_layers_dict[2][0]
            vp[i] = two_layers_dict[2][1]
    
    # Create tomography file: profile.xyz
    xyz_fname = 'profile.xyz'
    with open(os.path.join(dest_dir, xyz_fname), 'w') as f:
        f.write(f'{xmin} {zmin} {xmax} {zmax}\n')
        f.write(f'{dx} {dz}\n')
        f.write(f'{nx} {nz}\n')
        f.write(f'{min(vp)} {max(vp)} 0.0 0.0 {min(rho)} {max(rho)}\n')
        for i in range(nz):
            for j in range(nx):
                f.write(f"{xi[j]} {zi[i]} {vp[i]} 0.0 {rho[i]}\n")