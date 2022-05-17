import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter

from msc.specfem.multilayer.create_tomography_file import read_material_file
from msc.specfem.utils.scale_signal import scale_sig

###
# FOR FULL DOCUMENTATION ABOUT THE REST OF VARIABLES, TAKE A LOOK AT 'create_tomography_file.py'
###

def get_noise_snr(signal, snr_tar_db):
    """
    White Gaussian noise from SNR target level (in dB)
    """
    if len(signal.shape) > 1:
        sig_watts = signal[:, 0]**2
    else:
        sig_watts = signal**2
    sig_avg_watts = np.mean(sig_watts, keepdims=True)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    
    noise_avg_db = sig_avg_db - snr_tar_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    
    # Generate white Gaussian noise
    mean_noise = 0
    std_noise = np.sqrt(noise_avg_watts)
    noise = np.random.normal(mean_noise, std_noise, size=signal.shape)
    
    return noise


def create_2D_noisy_tomo(mesh_res: tuple, xmin_max: tuple, uneven_dict: dict, noise_tar: float, gaussian_filter_dim: str, invertz=False, path2mesh='./MESH', dest_dir='./DATA', save_xyz=True, sig_scale=None):
    """ 
    Writes down the .xyz noisy tomo file. It first creates a clean 2D velocity model, onto which
    white noise is added. Then a gaussian filter is applied, whose dimensionality can be specified. 
    
    New args:
        noise_tar           (float): level of noise (SNR target level) in dB. Sensible value 35-40. 
                                     If noise_tar = None, then no noise is added. 
        gaussian_filter_dim (str)  : specifies the dimension along which apply the gaussian filter. It 
                                     can be 'hori' (for horizontal), 'vert' (for vertical) and 'both' 
                                     (for both directions). 
    """
    if gaussian_filter_dim != None and gaussian_filter_dim not in ['hori', 'vert', 'both']:
        sys.exit("Gaussian filter dimension is neither 'hori' nor 'verti' nor 'both'!")
    
    zbot = -sum(uneven_dict.values())
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
    L_mult = uneven_dict['L_mult']
    N_mult = n_layers - 2
    l_size = L_mult/N_mult
    
    # Intefraces from bottom to top
    interfaces = [zbot]
    for dom_id in list(d2v.keys())[::-1]:
        size_ = l_size
        if dom_id in uneven_dict:
            size_ = uneven_dict[dom_id]
        interfaces += [interfaces[-1] + size_]
    interfaces[-2] = -uneven_dict[1]
    interfaces[-1] = ztop
        
    vp_2d = np.zeros((nz, nx))
    vs_2d = np.zeros((nz, nx))
    rho_2d = np.zeros((nz, nx))
    for i, (inf_lim, sup_lim) in enumerate(zip(interfaces[:-1], interfaces[1:])):
        idx = (n_layers - 1) - i
        mask = (inf_lim <= z) & (z <= sup_lim)
        vp_2d[mask, :] = vp[idx]
        rho_2d[mask, :] = rho[idx]
    
    if sig_scale is not None:
        vp_2d = scale_sig(vp_2d, sig_scale)
        # rho_2d = scale_sig(rho_2d, sig_scale)
    
    if noise_tar is not None:
        noise = get_noise_snr(vp_2d, noise_tar)
    else: 
        # No noise
        noise = np.zeros_like(vp_2d)
    vp_2d += noise
    
    if gaussian_filter_dim is not None:
        if gaussian_filter_dim == 'hori':
            # Horizontal filtering = constant z
            sigma = 7.5
            for i in range(nz):
                vp_2d[i,:] = gaussian_filter(vp_2d[i,:], sigma=sigma)
        elif gaussian_filter_dim == 'vert':
            # Vertical filtering = constant x
            sigma = 3
            for j in range(nx):
                vp_2d[:,j] = gaussian_filter(vp_2d[:,j], sigma=sigma)
        elif gaussian_filter_dim == 'both':
            # 2D filtering in both directions
            sigma = 5
            vp_2d = gaussian_filter(vp_2d, sigma=sigma)
    
    # Collect data in the correct format
    xcoords = []
    zcoords = []
    collect_fields = {'vp': [], 'vs': [], 'rho': []}
    for i, zval in enumerate(z):
        for j, xval in enumerate(x):
            xcoords.append(xval)
            zcoords.append(zval)
            collect_fields['vp'].append(vp_2d[i,j])
            collect_fields['rho'].append(rho_2d[i,j])
            collect_fields['vs'].append(vs_2d[i,j])
    
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
    
    return vp_2d, rho_2d  # OJO! Check if I've switched off the noise for the density
    
    
