import numpy as np 
import warnings
from scipy.ndimage import gaussian_filter
from msc.specfem.multilayer.create_tomography_file import read_material_file
from msc.specfem.multilayer.create_tomography_noisy import get_noise_snr


def model_depth_domain(uneven_dict, nz=10000, path2mesh='./MESH', noise_level=None, sigma=20):
    """ 
    Gets the models (velocities and density) in the depth domain for the case of the
    multilayer. This is necessary for the convolutional model.
    """
    d2v, rho, vp, vs = read_material_file(path2mesh)
    rho, vp, vs = rho.values, vp.values, vs.values
    n_layers = len(vp)
    
    ztop, zbot = 0.0, -sum(uneven_dict.values())
    z = np.linspace(ztop, zbot, nz)
    
    # Multilayer
    N_mult = n_layers - 2  # substract top and bottom layers
    L_mult = uneven_dict['L_mult']
    layer_size = L_mult/N_mult 
    
    interfaces = [0.0]
    for dom_id in d2v:
        size_ = layer_size
        if dom_id in uneven_dict:
            size_ = uneven_dict[dom_id]
        interfaces += [interfaces[-1] - size_]
    interfaces[-1] = zbot
    
    vp_z = np.zeros(nz)
    rho_z = np.zeros(nz)
    vs_z = np.zeros(nz)
    for i, (sup_lim, inf_lim) in enumerate(zip(interfaces[:-1], interfaces[1:])):
        mask = (inf_lim <= z) & (z <= sup_lim)
        vp_z[mask] = vp[i]
        rho_z[mask] = rho[i]
        vs_z[mask] = vs[i]
    if not all(np.isin(vp_z, vp)):
        warnings.warn('Resolution is not enough!')
        
    if noise_level is not None:
        noise = get_noise_snr(vp_z, noise_level)
        vp_z = gaussian_filter(vp_z + noise, sigma=sigma)
        rho_z = gaussian_filter(rho_z + noise, sigma=sigma)
    
    return z, vp_z, rho_z, vs_z
    

def time_depth_domain(nz, dz, vp):
    twt_z = np.zeros(nz)
    twt_z[0] = 2.0*dz/vp[0]
    for i in range(1, nz):
        twt_z[i] = twt_z[i-1] + 2.0*dz/vp[i]
    
    return twt_z