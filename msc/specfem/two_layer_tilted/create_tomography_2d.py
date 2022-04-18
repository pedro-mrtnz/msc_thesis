import os 
import numpy as np
from msc.specfem.multilayer_tilted.create_tomography_2d import tilted_boundary 

def create_tomo_tilted_2d(xmin_max, ztop_bot, mesh_res, vel_dict, angle_deg, path2mesh='./MESH', dest_dir='./DATA', invertz=False, save_xyz=True):
    ztop, zbot = ztop_bot
    xmin, xmax = xmin_max
    
    nx, nz = mesh_res
    dx = (xmax - xmin)/(nx - 1)
    dz = (ztop - zbot)/(nz - 1)
    xi = np.linspace(xmin, xmax, nx)
    zi = np.linspace(zbot, ztop, nz) if not(invertz) else np.linspace(ztop, zbot, nz)
    
    # 2D model
    interface = ztop - 0.5*(ztop - zbot)
    tilted_interface = tilted_boundary(xi, angle_deg, interface, ztop_bot)
    
    xcoords = []
    zcoords = []
    rho = []
    vp, vp_tomo = [], np.zeros((nz, nx))
    for i in range(nz):
        for j in range(nx):
            zcoords.append(zi[i])
            xcoords.append(xi[j])
            if zi[i] <= tilted_interface[j]:
                rho.append(vel_dict[2][0])  # layer 2 = bottom layer
                vp.append(vel_dict[2][1])
                vp_tomo[i,j] = vel_dict[2][1]
            else:
                rho.append(vel_dict[1][0])  # layer 1 = top layer
                vp.append(vel_dict[1][1])
                vp_tomo[i,j] = vel_dict[1][1]
    
    if save_xyz:
        xyz_fname = 'profile.xyz'
        with open(os.path.join(dest_dir, xyz_fname), 'w') as f:
            print(f'Name of the file: {f.name}')
            f.write(f'{xmin} {zbot} {xmax} {ztop}\n')
            f.write(f'{dx} {dz}\n')
            f.write(f'{nx} {nz}\n')
            f.write(f'{min(vp)} {max(vp)} 0.0 0.0 {min(rho)} {max(rho)}\n')
            for k in range(len(xcoords)):
                f.write(f"{xcoords[k]} {zcoords[k]} {vp[k]} 0.0 {rho[k]}\n")
    
    return vp_tomo
    
