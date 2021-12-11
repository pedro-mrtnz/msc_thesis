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
import argparse
import glob
import numpy as np
from scipy.io import FortranFile

def parse_args():
    """
    Processes the input arguments
    """
    parser = argparse.ArgumentParser(description='Creates profile.xyz from 1D velocity file')
    parser.add_argument('-res', '--GRID_RESOLUTION', type=tuple, default=(100, 200), help='Grid resolution')
    parser.add_argument('-path', '--path2file', default='./DATA/', help='Path to .bin files')
    
    return parser.parse_args()


def join_proc(path, proc_name):
    """Joins data from different processors.

    Args:
        path      (str): path to folder
        proc_name (str): processor names
        
    Return: data joined
    """
    data_names = glob.glob(path + proc_name)
    
    data_dict = {}
    size = []
    for data_n in data_names:
        name_f = data_n.split('/')[-1].split('_')[0]
        if 'bin' in proc_name:
            f = FortranFile(data_n, 'r')
            data_dict[name_f] = f.read_reals(dtype='float32')
        elif 'dat' in proc_name:
            data_dict[name_f] = pd.read_csv(data_n, sep='\s+').values
    
        print(f'Joining: {name_f}')
        size += [data_dict[name_f].shape[0]]
    
    if 'bin' in proc_name:
        data_joined = np.zeros(sum(size))
    elif 'dat' in proc_name:
        data_joined = np.zeros((sum(size), 5))
        
    size = [0] + size
    size = np.cumsum(size)
    i = 0
    for d_name in data_dict:
        if 'bin' in proc_name:
            data_joined[size[i]:size[i+1]] = data_dict[d_name]
        elif 'dat' in proc_name:
            data_joined[size[i]:size[i+1], :] = data_dict[d_name]
        i += 1
    
    return data_joined
        
    
def create_tomo_from_1Dfile(path, res):
    # Fixme: so far only .bin extension is implemented
    fnames = [
        'proc0000*_x.bin', 
        'proc0000*_z.bin', 
        'proc0000*_vp.bin',
        'proc0000*_vs.bin',
        'proc0000*_rho.bin'
    ]
    data_names = [glob.glob(path + fname)[0] for fname in fnames]
    collect_data = {}
    for data_n in data_names:
        key = data_n.split('_')[1].split('.')[0]
        f = FortranFile(data_n, 'r')
        collect_data[key] = f.read_reals(dtype='float32')
    
    x, z, vp, vs, rho = collect_data.values()
    del collect_data
    
    nx, nz = res
    minX, maxX = min(x), max(x)
    minZ, maxZ = min(z), max(z)
    xi   = np.linspace(minX, maxX, nx)
    zi   = np.linspace(minZ, maxZ, nz)
    vpi  = np.interp(np.arange(nz), np.arange(len(vp)), vp)
    vsi  = np.interp(np.arange(nz), np.arange(len(vs)), vs)
    rhoi = np.interp(np.arange(nz), np.arange(len(rho)), rho)
    
    dx = np.abs(maxX - minX)/(nx - 1)
    dz = np.abs(maxZ - minZ)/(nz - 1)
    
    with open('DATA/profile.xyz', 'w') as f:
        print(f'Name of the file: {f.name}')
        f.write(f'{minX} {minZ} {maxX} {maxZ}\n')
        f.write(f'{dx} {dz}\n')
        f.write(f'{nx} {nz}\n')
        f.write(f'{min(vp)} {max(vp)} {min(vs)} {max(vs)} {min(rho)} {max(rho)}\n')
        for i, z_val in enumerate(zi):
            for x_val in xi:
                f.write(f'{x_val} {z_val} {vpi[i]} {vsi[i]} {rhoi[i]}\n')
    
if __name__ == "__main__":
    """
    Creates the tomo file containing the velocity and density models. 
    """
    args = parse_args()
    create_tomo_from_1Dfile(args.path2file, args.GRID_RESOLUTION)