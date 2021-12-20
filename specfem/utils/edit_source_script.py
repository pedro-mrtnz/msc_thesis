import os

def set_dom_freq(f0, path2source='./DATA'):
    """Edits the dominant frequency f0 contained in the SOURCE file to one of our desire.

    Args:
        f0          (float): new dominant frequency.
        path2source (str)  : path to the SOURCE file we want to edit. Defaults to './DATA'.
        
    Returns: re-writes the existing SOURCE script with the new f0.
    """
    fname = os.path.join(path2source, 'SOURCE')
    with open(fname, 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if l[:2] == 'f0':
                line_num = i
        line_txt = f'f0                              = {f0}           # dominant source frequency (Hz) if not Dirac or Heaviside\n'
        lines[line_num] = line_txt
    
    with open(fname, 'w') as f:
        f.writelines(lines)
        
def set_source_coords(new_coords, path2source='./DATA'):
    """Edits the coordinates of the source in the SOURCE script.

    Args:
        new_coords  (tuple): (x, z) with the coordinates
        path2source (str)  : path to the SOURCE file we want to edit. Defaults to './DATA'.
        
    Returns: re-writes the source coordinates.
    """
    xs, zs = new_coords
    xs_txt = f'xs                              = {xs}          # source location x in meters'
    zs_txt = f'zs                              = {zs}          # source location z in meters (zs is ignored if source_surf is set to true, it is replaced with the topography height)'
    
    fname = os.path.join(path2source, 'SOURCE')
    with open(fname, 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if l[:2] == 'xs':
                lines[i] = xs_txt
            if l[:2] == 'zs':
                lines[i] = zs_txt
    
    with open(fname, 'w') as f:
        f.writelines(lines)