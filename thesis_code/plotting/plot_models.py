import argparse
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import nice_plot as nplt
from scipy.io import FortranFile

def parse_args():
    """
    This function run argparse to process the arguments given 
    by the user along with sfsection. Define default behaviour if they
    are not given and help message when sfsection -h is run
    """
    parser = argparse.ArgumentParser(description='Plots seismic record section')
    parser.add_argument('path2file', 
                        help = 'Path(s) to files, paths are separated by a ,')
    parser.add_argument('-e', '--extension', defailt='bin',
                        help = 'Plot .dat or .bin files (i.e. -e bin or -e dat)')
    # parser.add_argument('-f', '--format', default='SU',
    #                     help = 'Data format')
    
    # Optional formatting arguments
    parser.add_argument('-cm', '--cmap', default='viridis',
                        help = 'Colormap scheme')
    parser.add_argument('-vp1', '--vpmin', type=float, default=None,
                        help = 'vmin for vp color scale')
    parser.add_argument('-vp2', '--vpmax', type=float, default=None,
                        help = 'vmax for vp color scale')
    parser.add_argument('-vs1', '--vsmin', default='',
                        help = 'vmin for vs color scale')
    parser.add_argument('-vs2', '--vsmax', default='',
                        help = 'vmax for vs color scale')
    parser.add_argument('-rho1', '--rhomin', default='',
                        help = 'vmin for rho color scale')
    parser.add_argument('-rho2', '--rhomax', default='',
                        help = 'vmax for rho color scale')
    parser.add_argument('-s','--save', default='',
                        help = 'Save figure, full path to figure (e.g. -s path/models_init.pdf)')
    parser.add_argument('-xz','--path2xz',default='',
                        help = 'Path to proc*_x.bin and proc*_z.bin if not present')
    parser.add_argument('-t', '--title', type=str, default='',
                        help = 'Figure titles')
    parser.add_argument('-res', '--grid_resolution', type=str, default='200,160',
                        help = 'Grid resolution (-res resX,resY)')
    parser.add_argument('-wp', '--which_prop', type=str, default='',
                        help = 'Which property [vp,vs,rho]')
    parser.add_argument('-show','--show_plot',
                        help = 'Show a plot', action='store_true')
    parser.add_argument('-shot_pos', '--shot_pos_overlay', default='',
                        help = 'Path to file with source positions')
    parser.add_argument('-rec_pos', '--receiver_pos_overlay', default='',
                        help = 'Path to STATIONS file')
    parser.add_argument('-interp', '--grid_interpolation_method', default='linear',
                        help = 'Method for interpolating onto regular grid')
    parser.add_argument('-legend_pos', '--legend_position', default='best',
                        help = 'Legend position')
    # parser.add_argument('-xint', '--x_interval', type=float, default='1.0',
    #                     help = 'Offset axis tick spacing [km]')
    #
    # parser.add_argument('-yint', '--y_interval', type=float, default='1.0',
    #                     help = 'Time axis tick spacing [s]')
    
    return parser.parse_args()


def grid(x, y, z, resX=200, resY=300, method='linear'):
    """
    Converts 3 column data to matplotlib grid 
    """
    from scipy.interpolate import griddata
    
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)[::-1]
    Z = griddata((x, y), z, (xi[None,:], yi[:,None]), method=method)
    
    X, Y = np.meshgrid(xi, yi)
    
    return X, Y, Z

def join_proc(path, proc_name):
    """
    Join the data from different processors (proc*.bin or proc*.dat)
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
            data_joined[size[i]:size[i]+1] = data_dict[d_name]
        elif 'dat' in proc_name:
            data_joined[size[i]:size[i]+1, :] = data_dict[d_name]
        i += 1
    
    return data_joined

if __name__ == "__main__":
    """
    Specialised script used to visualize seismic section
    """
    args = parse_args()
    
    extension = args.extension
    
    vmin_max  = {}
    if args.vpmin:
        vmin_max['vp'] = (args.vpmin, args.vpmax)
    else:
        vmin_max['vp'] = (None, None)
    if args.vsmin:
        vmin_max['vs'] = (float(args.vsmin), float(args.vsmax))
    else:
        vmin_max['vs'] = (None, None)
    if args.rhomin:
        vmin_max['rho'] = (float(args.rhomin), float(args.rhomax))
    else:
        vmin_max['rho'] = (None, None)
        
    source_overlay   = args.shot_pos_overlay
    receiver_overlay = args.receiver_pos_overlay
    if source_overlay:
        source_pos = np.loadtxt(source_overlay)
    if receiver_overlay:
        receiver_pos = pd.read_csv(receiver_overlay, set='\s+', header=None).values
    
    resolution = args.grid_resolution
    resX, resY = resolution.split(',')
    resX, resY = int(resX), int(resY)
    
    interp_method = args.grid_interpolation_method
    
    fnames = {
        'x'  : '/proc0000*_x.bin',
        'z'  : '/proc0000*_z.bin',
        'vp' : '/proc0000*_vp.bin',
        'vs' : '/proc0000*_vs.bin',
        'rho': '/proc0000*_rho.bin'
    }
    
    paths = args.path2file
    paths = paths.split(',')
    
    path2xz = args.path2xz
    
    savefigure = args.save
    savefigures = savefigure.split(',')
    if not savefigures:
        savefigures = [False] * len(paths)
    
    cbar_labels = {
        'vp' : 'Vp [m/s]',
        'vs' : 'Vs [m/s]',
        'rho': 'Density [kg/m^3]'
    }
    
    data = {}
    n = 0
    for path in paths:
        # 'bin' extension
        if extension == 'bin':
            # Join processors data
            for fname in fnames:
                if fname in ['x', 'z'] and path2xz:
                    data[fname] = join_proc(path2xz, fnames[fname])
                else:
                    data[fname] = join_proc(path, fnames[fname])
            
            X = data['x']
            Z = data['z']
            
            max_X = np.max(X)
            min_X = np.min(X)
            max_Z = np.max(Z) 
            min_Z = np.min(Z)
            
            i = 1
            if not args.which_prop:
                plt.figure(figsize=(10, 10))
                for prop in ['vp', 'vs', 'rho']:
                    x, z, prop_array = grid(X, Z, data[prop], resX=resX, resY=resY, method=interp_method)
                    vm = vmin_max[prop]
                    
                    plt.subplot(3, 1, i)
                    plt.imshow(prop_array,
                               extent = [min_X, max_X, min_Z, max_Z], 
                               aspect = 'auto',
                               cmap = args.cmap,
                               vmin = vm[0], vmax = vm[1])
                    plt.colorbar()
                    plt.title('/'.join(path.split('/')[-2:]) + f'{prop}.{extension}')
                    
                    i += 1
            else:
                prop = args.which_prop
                x, z, prop_array = grid(X, Z, data[prop], resX=resX, resY=resY, method=interp_method)
                vm = vmin_max[prop]
                
                nplt.plotting_image(prop_array, 
                                    extent = [min_X, max_X, min_Z, max_Z],
                                    aspect = 'auto',
                                    cmap = args.cmap,
                                    vmin = vm[0], vmax = vm[1],
                                    figsize = (10, 5),
                                    cbar_label = cbar_labels[prop])
                print(f'Data: min, max = {np.min(data[prop])}, {np.max(data[prop])}')
                if args.title:
                    plt.title(args.title)
                else:
                    plt.title('/'.join(path.split('/')[-2:]) + f'{prop}.{extension}') 
                plt.xlabel('X [m]')
                plt.ylabel('Z [m]')
        
        # 'dat' extension 
        if 'dat' in extension:
            data = join_proc(path, f'/proc000*_rho_vp_vs.{extension}')
            X = data[:,0]
            Z = data[:,1]
            data_dict = {
                'rho': data[:,2], 
                'vp': data[:,3], 
                'vs': data[:,4]
            }
            
            max_X = np.max(X)
            min_X = np.min(X)
            max_Z = np.max(Z) 
            min_Z = np.min(Z)
            
            i = 1
            if not args.which_prop:
                plt.figure(figsize=(10, 10))
                for prop in data_dict:
                    x, z, prop_array = grid(X, Z, data_dict[prop], resX=resX, resY=resY, method=interp_method)
                    vm = vmin_max[prop]
                    
                    plt.subplot(3, 1, i)
                    plt.imshow(prop_array,
                               extent = [min_X, max_X, min_Z, max_Z], 
                               aspect = 'auto',
                               cmap = args.cmap,
                               vmin = vm[0], vmax = vm[1])
                    plt.colorbar()
                    plt.title('/'.join(path.split('/')[-2:]) + f'{prop}.{extension}')
                    
                    i += 1
            else:
                prop = args.which_prop
                x, z, prop_array = grid(X, Z, data_dict[prop], resX=resX, resY=resY, method=interp_method)
                vm = vmin_max[prop]
                
                nplt.plotting_image(prop_array, 
                                    extent = [min_X, max_X, min_Z, max_Z],
                                    aspect = 'auto',
                                    cmap = args.cmap,
                                    vmin = vm[0], vmax = vm[1],
                                    figsize = (10, 5),
                                    cbar_label = cbar_labels[prop])
                print(f'Data: min, max = {np.min(data_dict[prop])}, {np.max(data_dict[prop])}')
                if args.title:
                    plt.title(args.title)
                else:
                    plt.title('/'.join(path.split('/')[-2:]) + f'{prop}.{extension}') 
                plt.xlabel('X [m]')
                plt.ylabel('Z [m]')
        
        # 'xyz' extension
        if 'xyz' in extension:
            data = pd.read_csv(path, sep='\s+', skiprows=4).values
            X = data[:,0]
            Z = data[:,1]
            data_dict = {
                'rho': data[:,2], 
                'vp': data[:,3], 
                'vs': data[:,4]
            }
            
            max_X = np.max(X)
            min_X = np.min(X)
            max_Z = np.max(Z) 
            min_Z = np.min(Z)
            
            i = 1
            if not args.which_prop:
                plt.figure(figsize=(10, 10))
                for prop in data_dict:
                    x, z, prop_array = grid(X, Z, data_dict[prop], resX=resX, resY=resY, method=interp_method)
                    vm = vmin_max[prop]
                    
                    plt.subplot(3, 1, i)
                    plt.imshow(prop_array,
                               extent = [min_X, max_X, min_Z, max_Z], 
                               aspect = 'auto',
                               cmap = args.cmap,
                               vmin = vm[0], vmax = vm[1])
                    plt.colorbar()
                    plt.title('/'.join(path.split('/')[-2:]) + f'{prop}.{extension}')
                    
                    i += 1
            else:
                prop = args.which_prop
                x, z, prop_array = grid(X, Z, data_dict[prop], resX=resX, resY=resY, method=interp_method)
                vm = vmin_max[prop]
                
                nplt.plotting_image(prop_array, 
                                    extent = [min_X, max_X, min_Z, max_Z],
                                    aspect = 'auto',
                                    cmap = args.cmap,
                                    vmin = vm[0], vmax = vm[1],
                                    figsize = (10, 5),
                                    cbar_label = cbar_labels[prop])
                print(f'Data: min, max = {np.min(data_dict[prop])}, {np.max(data_dict[prop])}')
                if args.title:
                    plt.title(args.title)
                else:
                    plt.title('/'.join(path.split('/')[-2:]) + f'{prop}.{extension}') 
                plt.xlabel('X [m]')
                plt.ylabel('Z [m]')
        
        if receiver_overlay:
            x, z = receiver_pos[:, 2], receiver_pos[:, 3]
            plt.plot(x, z, '*', label='Receivers')
            
        if source_overlay:
            i = 0
            if source_pos.ndim == 1:
                x, z = source_pos
                plt.plot(x, z, '*', label=f'Shot position {i}', markersize=10)
            else:
                for pos in source_pos:
                    x, z = pos
                    plt.plot(x, z, '*', label=f'Shot position {i}', markersize=10)
                    i += 1
        if receiver_overlay or source_overlay:
            plt.legend(loc = args.legend_position)
        
        if savefigures[n]:
            print(f'Saving {path} to {savefigures[n]}')
            plt.savefig(savefigures[n])
        
        n += 1
        
    if args.show_plot:
        plt.show()
            