import sys
import os
import math

def create_stations_file(rec_lims, pos_ini, nsts_x, nsts_z, dest_dir='./DATA'):
    """Creates the STATIONS (receivers) file called from the Par_file.

    Args:
        rec_lims (tuple): range of surface occupied by receivers (xrange, zrange).
        pos_ini  (tuple): starting position (x0, z0).
        nsts_x   (int)  : number of receivers along x.
        nsts_z   (int)  : number of receivers along z.
        dest_dir (str)  : path where to save the STATIONS fie.
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
    xrange, zrange = rec_lims
    dx = xrange/(nsts_x - 1) if nsts_x > 1 else 0.0
    dz = zrange/(nsts_z - 1) if nsts_z > 1 else 0.0
    
    # String formatting
    idx_len = int(math.ceil(math.log10(nsts_x * nsts_z)))
    
    # Create nsts_x * nsts_z receivers
    irec = 0
    for iz in range(nsts_z):
        for ix in range(nsts_x):
            irec += 1
            x = x0 + ix * dx                  # Station position
            z = z0 + iz * dz
            
            st_name = f"S{irec:0{idx_len}d}"  # Station name
            network = "AA"                    # Network station
            
            f.write(
                f"{st_name:<10s} {network:<10s} {f'{x:.2f}':<15s} {f'{z:.2f}':<15s} {str(0.0):<15s} {0.0}\n")
    
    f.close()
                        
    