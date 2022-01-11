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
import argparse
import numpy as np
import pandas as pd
import meshio
import matplotlib.pyplot as plt

import msc.specfem.plotting.nice_plot as nplt
from msc.specfem.plotting.plot_models import grid


class ConfigPlots:
    aspect        = 'auto'
    cmap          = 'nice'
    vmin_max      = (None, None)
    figsize       = (10, 5)
    res           = (100, 200)
    sel_field     = 'vp'          # Select field to plot
    plot          = True          # Plot the resulting field
    interp_method = 'linear'


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

   
def plot_field(x, z, field, field_name, res=ConfigPlots.res, interp_method=ConfigPlots.interp_method):
    """
    Plot field (vp, vs or rho) stored tomography file. 
    """
    xmin, xmax = min(x), max(x)
    zmin, zmax = min(z), max(z)
    resX, resZ = res
    
    X, Z, F = grid(x, z, field, resX=resX, resY=resZ, method=interp_method)
    
    field2cbar_label = {
        'vp' : 'Vp [m/s]',
        'vs' : 'Vs [m/s]',
        'rho': r'Density [kg/m$^3$]'
    }
    if field_name not in field2cbar_label.keys():
        print('Field not recognised!')
        cbar_label = ''
    else:
        cbar_label = field2cbar_label[field_name]
    
    nplt.plotting_image(F,
                        extent = [xmin, xmax, zmin, zmax],
                        aspect = ConfigPlots.aspect,
                        cmap = ConfigPlots.cmap,
                        vmin_max = ConfigPlots.vmin_max,
                        figsize = ConfigPlots.figsize,
                        cbar_label = cbar_label)
    plt.xlabel(r'$x$ [m]')
    plt.ylabel(r'$z$ [m]')
    plt.show()


def create_tomo_1Deven_file(path2mesh='./MESH', dest_dir='./DATA', mesh_size=None, lc=10.0, mesh_res=None, plot=False):
    """ Writes down the .xyz file which wraps up the 1D velocity and density model of the MULTILAYER. It is 1D in the sense
    that depends only on the z direction. This model has EVENLY SPACED LAYERS. 
    
    NB: a more general model where you can specify which layers are UNEVEN is implemented further on.

    Args:
        path2mesh (str)  : path to MESH folder. Defaults to './MESH'.
        dest_dir  (str)  : destination folder where to save the .xyz file. Defaults to './DATA'.
        mesh_size (list) : size of the mesh in the format [(xmin, xmax), (ztop, zbot)], if None the mesh is
                           loaded using meshio.read(*.msh) and points are fetched. Defaults to None.
        lc        (float): gmsh discretization parameter. Defaults to 10.0.
        mesh_res  (tuple): mesh resolution in the format (nx, nz), if None calculated from mesh_size 
                           and lc. Defaults to None.
        plot      (bool) : if True the field is plotted. Defaults to True.
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
    # Fixme: think case in which zmax != 0
    
    # Interpolating axis
    dx = (xmax - xmin)/(nx - 1)
    dz = (zmax - zmin)/(nz - 1)
    xi = np.linspace(xmin, xmax, nx)
    zi = np.linspace(zmin, zmax, nz)
    dom_in_zi = np.zeros_like(zi).astype('int32') 
    for dom_id, (sup_lim, inf_lim) in enumerate(zip(dom_intervals[:-1], dom_intervals[1:])):
        mask = (zi <= sup_lim) & (zi >= inf_lim)
        dom_in_zi[mask] = dom_id + 1
    
    xcoords = []
    zcoords = []
    collect_fields = {'vp': [], 'vs': [], 'rho': []}    
    for i, zval in enumerate(zi):
        for xval in xi:
            dom_id = dom_in_zi[i]
            df_ = d2v[dom_id]  # pd.Series
            xcoords.append(xval)
            zcoords.append(zval)
            collect_fields['vp'].append(df_['vp'])
            collect_fields['vs'].append(df_['vs'])
            collect_fields['rho'].append(df_['rho'])
    
    assert len(xcoords) == len(zcoords), 'Mismatch in sizes!'        
    if plot:
        # One or more fields can be plotted
        field_name = ConfigPlots.sel_field
        field_names = field_name.split(',')
        if len(field_names) == 1:
            field = collect_fields[field_name]
            plot_field(xcoords, zcoords, field, field_name)
        else:
            for field_name in field_names:
                field = collect_fields[field_name.strip()]
                plot_field(xcoords, zcoords, field, field_name)
                      
    # Create the velocity file
    xyz_fname = 'profile.xyz'
    with open(os.path.join(dest_dir, xyz_fname), 'w') as f:
        print(f'Name of the file: {f.name}')
        f.write(f'{xmin} {zmin} {xmax} {zmax}\n')
        f.write(f'{dx} {dz}\n')
        f.write(f'{nx} {nz}\n')
        f.write(f'{min(vp)} {max(vp)} {min(vs)} {max(vs)} {min(rho)} {max(rho)}\n')
        for j in range(len(xcoords)):
            f.write(f"{xcoords[j]} {zcoords[j]} {collect_fields['vp'][j]} {collect_fields['vs'][j]} {collect_fields['rho'][j]}\n")


def create_tomo_1Dfile(path2mesh='./MESH', dest_dir='./DATA', mesh_size=None, lc=10.0, uneven=None, mesh_res=None, plot=False):
    """
    Generalizes the evenly spaced multilayer. It allows for UNEVEN layers that we have to specify. E.g. if we
    want the sandwich model, we would need: uneven = {1: 2000, L_mult: 500, 85: 2000}. This means that domain ids 1
    and 85 are 2000m big each. The rest of the multilayer is readjusted.
    
    New args:
        uneven (dict) : dictionary with domain ids as keys and the sizes of those domains as values. If None
                        then we retrieve the evenly spaced multilayer. 
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
        nx = np.ceil(np.abs(xmax - xmin)/lc).astype(int)
        nz = np.ceil(np.abs(zmax - zmin)/lc).astype(int)
    else:
        nx, nz = mesh_res
        # Fixme: think about the case in which the vertical resolution is not even
    
    d2v, rho, vp, vs = read_material_file(path2mesh)
    
    # Interpolating axis
    dx = (xmax - xmin)/(nx - 1)
    dz = (zmax - zmin)/(nz - 1)
    xi = np.linspace(xmin, xmax, nx)
    zi = np.linspace(zmin, zmax, nz)
    # Fixme: think about improving the mesh resolution around L_mult, really necessary?
    
    if uneven is not None:
        L_mult = uneven['L_mult']
        N = len(d2v) - (len(uneven) - 1)
    else:
        # We get multilayer model
        L_mult = zmax - zmin
        uneven = {}

    dom_size = L_mult/N
    dom_intervals = [0.0]
    for dom_id in d2v.keys():
        size_ = dom_size
        if dom_id in uneven:
            size_ = uneven[dom_id]
        dom_intervals += [dom_intervals[-1] - size_]
    
    dom_in_zi = np.zeros_like(zi).astype('int32')
    for i, (sup_lim, inf_lim) in enumerate(zip(dom_intervals[:-1], dom_intervals[1:])):
        mask = (zi <= sup_lim) & (zi > inf_lim)
        dom_in_zi[mask] = i + 1
    print(f'\nDOMAINS IN Z-COORDS FROM TOMO FILE:\n {dom_in_zi}')
    
    xcoords = []
    zcoords = []
    collect_fields = {'vp': [], 'vs': [], 'rho': []}    
    for i, zval in enumerate(zi):
        for xval in xi:
            dom_id = dom_in_zi[i]
            df_ = d2v[dom_id]  # pd.Series
            xcoords.append(xval)
            zcoords.append(zval)
            collect_fields['vp'].append(df_['vp'])
            collect_fields['vs'].append(df_['vs'])
            collect_fields['rho'].append(df_['rho'])
    
    assert len(xcoords) == len(zcoords), 'Mismatch in sizes!'    
    
    if plot:
        # One or more fields can be plotted
        field_name = ConfigPlots.sel_field
        field_names = field_name.split(',')
        if len(field_names) == 1:
            field = collect_fields[field_name]
            plot_field(xcoords, zcoords, field, field_name)
        else:
            for field_name in field_names:
                field = collect_fields[field_name.strip()]
                plot_field(xcoords, zcoords, field, field_name)
    
    # Create the velocity file
    xyz_fname = 'profile.xyz'
    with open(os.path.join(dest_dir, xyz_fname), 'w') as f:
        print(f'Name of the file: {f.name}')
        f.write(f'{xmin} {zmin} {xmax} {zmax}\n')
        f.write(f'{dx} {dz}\n')
        f.write(f'{nx} {nz}\n')
        f.write(f'{min(vp)} {max(vp)} {min(vs)} {max(vs)} {min(rho)} {max(rho)}\n')
        for j in range(len(xcoords)):
            f.write(f"{xcoords[j]} {zcoords[j]} {collect_fields['vp'][j]} {collect_fields['vs'][j]} {collect_fields['rho'][j]}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates tomography file and plots it (if desired)')
    parser.add_argument('xmin', type=float)
    parser.add_argument('xmax', type=float)
    parser.add_argument('ztop', type=float)
    parser.add_argument('zbot', type=float)
    parser.add_argument('lc', type=float)
    parser.add_argument('-L_mult', type=float, default=None,
                        help = 'Size of the multilayer zone.')
    parser.add_argument('-p2m', '--path2mesh', type=str,
                        default='./MESH')
    parser.add_argument('-dest', '--dest_dir', type=str,
                        default='./DATA')
    parser.add_argument('-res', '--gridplot_resolution', type=tuple)
    parser.add_argument('-f', '--sel_field', type=str,
                        help = "Field(s) to plot: vp, vs or rho. Pass 'vp,rho' to plot both.")
    parser.add_argument('-show', '--show_plot', action='store_false')
    parser.add_argument('-uev', '--uneven', type=bool, action='store_false')
    
    args = parser.parse_args()
    
    # NB: 'gridplot_resolution' refers to the resolution of the plot, whereas
    #     'mesh_res' is the resolution of the mesh, which is calculated based on 'lc'.
    if args.gridplot_resolution:
        ConfigPlots.res = args.gridplot_resolution
        
    if args.sel_field:
        ConfigPlots.sel_field = args.sel_field
    
    if args.uneven:
        # We read from a .txt file
        uneven_df   = pd.read_csv('./uneven_dict.txt', header=None, sep=' ', names=['dom_id', 'size'])    
        uneven_dict = {dom_id: size for dom_id, size in zip(uneven_df['dom_id'], uneven_df['size'])}
    else:
        uneven_dict = None
    
    create_tomo_1Dfile(args.path2mesh, 
                       args.dest_dir, 
                       uneven = uneven_dict,
                       mesh_size = [(args.xmin, args.xmax), (args.zbot, args.ztop)], 
                       lc = args.lc,
                       L_mult = args.L_mult,
                       mesh_res = None, 
                       plot = args.show_plot)
    