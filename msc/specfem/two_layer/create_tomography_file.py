import os
import numpy as np
from msc.specfem.multilayer.create_tomography_file import plot_field
from msc.specfem.multilayer.create_tomography_noisy import get_noise_snr

def create_tomo_1Dfile(mesh_size, mesh_res, two_layers_dict, iface_pos=0.5, noise_tar=None, gaussian_dim=None, dest_dir='./DATA', plot=None, save_xyz=True):
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
    
    # 2D models
    interface = zmax - iface_pos*(zmax - zmin)
    maskz = zi <= interface
    vp_2d = two_layers_dict[1][1]*np.ones((nz, nx))
    rho_2d = two_layers_dict[1][0]*np.ones((nz, nx))
    vp_2d[maskz,:] = two_layers_dict[2][1]
    rho_2d[maskz,:] = two_layers_dict[2][0]
    
    # Noise
    if noise_tar is not None:
        noise = get_noise_snr(vp_2d, noise_tar)
    else:
        noise = np.zeros_like(vp_2d)
    vp_2d += noise
    
    if (gaussian_dim is not None) and (noise_tar is not None):
        from scipy.ndimage import gaussian_filter
        if gaussian_dim == 'hori':
            # Horizontal filtering = constant z
            sigma = 3
            for i in range(nz):
                vp_2d[i,:] = gaussian_filter(vp_2d[i,:], sigma=sigma)
        elif gaussian_dim == 'both':
            # 2D filtering in both directions
            sigma = 7.5
            vp_2d = gaussian_filter(vp_2d, sigma=sigma)
    
    xcoords = []
    zcoords = []
    rho = []
    vp = []
    for i in range(nz):
        for j in range(nx):
            zcoords.append(zi[i])
            xcoords.append(xi[j])
            rho.append(rho_2d[i,j])
            vp.append(vp_2d[i,j])
            
    if plot is not None:
        assert plot in ['vp', 'rho'], "Field not recognised!"
        if plot == 'vp':
            plot_field(xcoords, zcoords, vp, plot)
        elif plot == 'rho':
            plot_field(xcoords, zcoords, rho, plot)
    
    # Create tomography file: profile.xyz
    if save_xyz:
        xyz_fname = 'profile.xyz'
        with open(os.path.join(dest_dir, xyz_fname), 'w') as f:
            f.write(f'{xmin} {zmin} {xmax} {zmax}\n')
            f.write(f'{dx} {dz}\n')
            f.write(f'{nx} {nz}\n')
            f.write(f'{min(vp)} {max(vp)} 0.0 0.0 {min(rho)} {max(rho)}\n')
            for k in range(nz*nx):
                f.write(f"{xcoords[k]} {zcoords[k]} {vp[k]} 0.0 {rho[k]}\n")
            
    return vp_2d, rho_2d