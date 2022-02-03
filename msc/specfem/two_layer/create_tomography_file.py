import os
import numpy as np
from msc.specfem.multilayer.create_tomography_file import plot_field

def create_tomo_1Dfile(mesh_size, mesh_res, two_layers_dict, dest_dir='./DATA', plot=None):
    """
    Creates 1D tomography file for the simple 2 layer model. 
    
    Args:
        mesh_size       (list) : mesh dimesnions as [(xmin, xmax), (zmin, zmax)]
        mesh_res        (tuple): mesh resolution (nx, nz)
        two_layers_dict (dict) : contains rho and vp for the two layers. 
        dest_dir        (str)  : destination folder where to save .xyz tomo file.
        plot            (str)  : plot field 'vp' / 'rho'. Defaults to None. 
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
    xcoords = []
    zcoords = []
    rho = []
    vp  = []
    for i in range(nz):
        for j in range(nx):
            zcoords.append(zi[i])
            xcoords.append(xi[j])
            if zi[i] <= interface:
                rho.append(two_layers_dict[1][0])
                vp.append(two_layers_dict[1][1])
            else:
                rho.append(two_layers_dict[2][0])
                vp.append(two_layers_dict[2][1])
            
    if plot is not None:
        assert plot in ['vp', 'rho'], "Field not recognised!"
        if plot == 'vp':
            plot_field(xcoords, zcoords, vp, plot)
        elif plot == 'rho':
            plot_field(xcoords, zcoords, rho, plot)
    
    # Create tomography file: profile.xyz
    xyz_fname = 'profile.xyz'
    with open(os.path.join(dest_dir, xyz_fname), 'w') as f:
        f.write(f'{xmin} {zmin} {xmax} {zmax}\n')
        f.write(f'{dx} {dz}\n')
        f.write(f'{nx} {nz}\n')
        f.write(f'{min(vp)} {max(vp)} 0.0 0.0 {min(rho)} {max(rho)}\n')
        for k in range(nz*nx):
            f.write(f"{xcoords[k]} {zcoords[k]} {vp[k]} 0.0 {rho[k]}\n")