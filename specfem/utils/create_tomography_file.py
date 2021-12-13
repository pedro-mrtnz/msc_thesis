"""
EXAMPLE OF EXTERNAL VELOCITY MODEL
Build !D velocity model and write it in ASCII tomo file that can be read by SPECFEM2D. It is 1D in the sense that the 
velocity depends only on the depth (vertical axis).

We use the TOMOGRAPHY_FILE in order to create an arbitrary velocity model. Within the Par_file, 
it is enabled by using these values when the velocity model is defined: 

    2 -1 9999 9999 9999 9999 9999 9999 9999 0 0 0 0 0 0

    # external tomography file
    TOMOGRAPHY_FILE                 = ./profile.xyz

The program understands that the velocity model number 2 has to be read in the profile.xyz as a regular grid. It will then deal 
with the interpolation. 

The file profile.xyz has to be written under the following format:

    ! The xyz file TOMOGRAPHY_FILE that describe the tomography should be located in the TOMOGRAPHY_PATH
    ! directory, set in the Par_file. The format of the file, as read from define_external_model_from_xyz_file.f90 looks like :
    !
    ! ORIGIN_X ORIGIN_Z END_X END_Z
    ! SPACING_X SPACING_Z
    ! NX NZ
    ! VP_MIN VP_MAX VS_MIN VS_MAX RHO_MIN RHO_MAX
    ! x(1) z(1) vp vs rho
    ! x(2) z(1) vp vs rho
    ! ...
    ! x(NX) z(1) vp vs rho
    ! x(1) z(2) vp vs rho
    ! x(2) z(2) vp vs rho
    ! ...
    ! x(NX) z(2) vp vs rho
    ! x(1) z(3) vp vs rho
    ! ...
    ! ...
    ! x(NX) z(NZ) vp vs rho
    !
    ! Where:
    ! _x and z must be increasing
    ! _ORIGIN_X, END_X are, respectively, the coordinates of the initial and final tomographic
    !  grid points along the x direction (in meters)
    ! _ORIGIN_Z, END_Z are, respectively, the coordinates of the initial and final tomographic
    !  grid points along the z direction (in meters)
    ! _SPACING_X, SPACING_Z are the spacing between the tomographic grid points along the x
    !  and z directions, respectively (in meters)
    ! _NX, NZ are the number of grid points along the spatial directions x and z,
    !  respectively; NX is given by [(END_X - ORIGIN_X)/SPACING_X]+1; NZ is the same as NX, but
    !  for z direction.
    ! _VP_MIN, VP_MAX, VS_MIN, VS_MAX, RHO_MIN, RHO_MAX are the minimum and maximum values of
    !  the wave speed vp and vs (in m.s-1) and of the density rho (in kg.m-3); these values
    !  could be the actual limits of the tomographic parameters in the grid or the minimum
    !  and maximum values to which we force the cut of velocity and density in the model.
    ! _After these first four lines, in the file file_name the tomographic grid points are
    !  listed with the corresponding values of vp, vs and rho, scanning the grid along the x
    !  coordinate (from ORIGIN_X to END_X with step of SPACING_X) for each given z (from ORIGIN_Z
    !  to END_Z, with step of SPACING_Z).
"""
import os
import glob
import numpy as np
import pandas as pd
import meshio

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
        

def sign(x):
    """ Gets sign of x """
    return 1.0 if x >= 0.0 else -1.0

def create_tomo_1Dhomo_file(path2mesh='./MESH', dest_dir='./DATA', mesh_size=None, lc=10.0, mesh_res=None):
    """ Writes down the .xyz file which wraps up the 1D velocity and density model. It is 1D in the sense
    that depends only on the z direction.

    Args:
        path2mesh   (str): path to MESH folder. Defaults to './MESH'.
        dest_dir    (str): destination folder where to save the .xyz file. Defaults to './DATA'.
        mesh_size  (list): size of the mesh in the format [(xmin, xmax), (ztop, zbot)], if None the mesh is
                            loaded using meshio.read(*.msh) and points are fetched. Defaults to None.
        lc        (float): gmsh discretization parameter. Defaults to 10.0.
        mesh_res  (tuple): mesh resolution in the format (nx, nz), if None calculated from mesh_size 
                            and lc. Defaults to None.
    """
    if mesh_size is None:
        mesh = meshio.read(glob.glob(os.path.join(path2mesh, '*.msh'))[0])
        mesh_points = mesh.points
        xmesh = mesh_points[:,0]
        zmesh = mesh_points[:,1] 

        xmin, xmax = np.ceil(min(xmesh)), np.floor(max(xmesh))
        zmin, zmax = np.ceil(min(zmesh)), np.floor(max(zmesh))
    else:
        xmin, xmax = min(mesh_size[0]), max(mesh_size[0])
        zmin, zmax = min(mesh_size[1]), max(mesh_size[1])
    
    if mesh_res is None:
        nx = int(np.abs(xmax - xmin)/lc)
        nz = int(np.abs(zmax - zmin)/lc)
    else:
        nx, nz = mesh_res
        
    d2v, rho, vp, vs = read_material_file(path2mesh)
    N = len(d2v)                                    # Number of evenly spaced domain_ids
    dom_size = (zmax - zmin)/N                      # Size of each domain
    dom_intervals = np.cumsum([0] + [-dom_size]*N)  # Domain intervals
    # Thinkme: think case in which zmax != 0
    
    # Interpolating axis
    dx = (xmax - xmin)/(nx - 1)
    dz = (zmax - zmin)/(nz - 1)
    xi = np.linspace(xmin, xmax, nx)
    zi = np.linspace(zmin, zmax, nz)
    dom_in_zi = np.zeros_like(zi).astype('int32') 
    for dom_id, (sup_lim, inf_lim) in enumerate(zip(dom_intervals[:-1], dom_intervals[1:])):
        mask = (zi <= sup_lim) & (zi >= inf_lim)
        dom_in_zi[mask] = dom_id + 1
        
    # Create the velocity file
    xyz_fname = 'profile.xyz'
    with open(os.path.join(dest_dir, xyz_fname), 'w') as f:
        print(f'Name of the file: {f.name}')
        f.write(f'{xmin} {zmin} {xmax} {zmax}\n')
        f.write(f'{dx} {dz}\n')
        f.write(f'{nx} {nz}\n')
        f.write(f'{min(vp)} {max(vp)} {min(vs)} {max(vs)} {min(rho)} {max(rho)}\n')
        for i, z_val in enumerate(zi):
            for x_val in xi:
                dom_id = dom_in_zi[i]
                df_ = d2v[dom_id]  # pd.Series
                f.write(f"{x_val} {z_val} {df_['vp']} {df_['vs']} {df_['rho']}\n")
        
    
    