import os
import pandas as pd

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