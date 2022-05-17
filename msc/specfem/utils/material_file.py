import os
import numpy as np
import pandas as pd

def get_materials(path2refl, rho_ini, vp_ini, verbose=True):
    """
    Creates the material file from from the reflectivity profile. It assumes a linear 
    relationship between velocity and density: velocity = alpha * density
    
    Args:
        path2refl (str)  : path to the reflectivity profile
        rho_ini   (float): density of the first layer 
        vp_ini    (float): velocity of the first layer  
    
    Returns: 
        File is saved under the name 'material_file.txt', and returns the number of layers.
    """
    vs, Q_mu, Q_kappa = 0, 9999, 9999
    mat_id = 1
    alpha = 1.8
    if verbose:
        print('MATERIAL LAYER 1:')
        print(f"{'  Rho, Vp, Vs': <15} = {rho_ini}, {vp_ini}, {vs}")
        print(f"{'  Q_mu, Q_kappa': <15} = {Q_mu}, {Q_kappa}")
        print(f"{'  Material ID': <15} = {mat_id}")
        print(f"{'  Cte vp-rho': <15} = {alpha}")
    
    refls = np.loadtxt(path2refl)
    if not os.path.exists('MESH/'):
        os.system('mkdir -p MESH/')
    print('\nWritting down material_file.txt!\n')
    with open('MESH/material_file.txt', 'w') as f:
        domain_id = 1
        f.write(f"{domain_id} {mat_id} {rho_ini:.2f} {vp_ini:.2f} {vs} {0} {0} {Q_kappa} {Q_mu} 0 0 0 0 0 0\n")
        for R in refls:
            rho_new = np.sqrt(-(R + 1)/(R - 1) * vp_ini * rho_ini / alpha)
            vp_new  = alpha * rho_new
            domain_id += 1
            f.write(f"{domain_id} {mat_id} {rho_new:.2f} {vp_new:.2f} {vs} {0} {0} {Q_kappa} {Q_mu} 0 0 0 0 0 0\n")
            
            rho_ini, vp_ini = rho_new, vp_new
    
    return len(refls) + 1


def read_material_file(path2mesh):
    """
    Reads the material_file.txt contained in the MESH folder. 
    
    Args:
        path2mesh (str): path to the MESH folder. 
    
    Return: 
        Mapping between the domain id and the velocity-density values in that layer/domain, 
        together with the vp, vs and rho arrays.
    """
    fname = 'material_file.txt'
    if os.path.exists(os.path.join(path2mesh, fname)):
        mat_df = pd.read_csv(os.path.join(path2mesh, fname), header=None, sep=' ', index_col=0)
        # NB: domain_id is used as index in the DataFrame
    else:
        print("No file matching 'material_file.txt'!")
    
    # Mapping between df.columns and properties (vp, vs, rho)
    prop2col = {'rho': 2, 'vp': 3, 'vs': 4}
    
    # Mapping between domain_id and velocity and density models
    dom2vels = {}
    for dom_id in mat_df.index:
        df_ = mat_df.loc[dom_id][prop2col.values()]  # pd.Series
        df_.index = prop2col.keys()                  # Change indexes from {2, 3, 4} to {rho, vp, vs}
        dom2vels[dom_id] = df_
    
    return dom2vels, mat_df[prop2col['rho']], mat_df[prop2col['vp']], mat_df[prop2col['vs']]