import sys
import os
import math

def create_stations_file(model_size, pos_ini, nsts_x, nsts_z, dest_dir='./DATA'):
    """Creates the STATIONS (receivers) file called from the Par_file.

    Args:
        model_size (tuple): size of the model ([xmin, xmax], [zmin, zmax]).
        pos_ini    (tuple): starting position (x0, z0).
        nsts_x     (int)  : number of receivers along x.
        nsts_z     (int)  : number of receivers along z.
        dest_dir   (str)  : path where to save the STATIONS fie.
    """
    if not os.path.isdir(dest_dir):
        sys.exit("Destination path doesn't exist!")
    
    fname = 'STATIONS'
    try:
        f = open(os.path.join(dest_dir, fname), 'w')
    except:
        sys.tracebacklimit = 0
        raise Exception(f"File won't open: {os.path.join(dest_dir, fname)}")

    x0, z0 = pos_ini
    
    xlims, zlims = model_size
    dim_x = max(xlims) - min(xlims)
    dim_z = max(zlims) - min(zlims)
    
    dx = dim_x/(nsts_x - 1) if nsts_x > 1 else 0.0
    dz = dim_z/(nsts_z - 1) if nsts_z > 1 else 0.0
    
    # String formatting
    idx_len = int(math.ceil(math.log10(nsts_x * nsts_z)))
    
    # Create nsts_x * nsts_z receivers
    irec = 0
    for iz in range(nsts_z):
        for ix in range(nsts_x):
            irec += 1
            x = x0 + ix * dx                    # Station position
            z = z0 + iz * dz
            
            st_name = f"S{irec:000{idx_len}d}"  # Station name
            network = "AA"                      # Network station
            
            f.write(f"{st_name}\t{network}\t{x:.2f}\t{z:.2f}\t{0.0}\t{0.0}\n")
    
    f.close()
                        
    