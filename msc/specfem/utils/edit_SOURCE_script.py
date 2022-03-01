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
    xs_txt = f'xs                              = {xs}          # source location x in meters\n'
    zs_txt = f'zs                              = {zs}          # source location z in meters (zs is ignored if source_surf is set to true, it is replaced with the topography height)\n'
    
    fname = os.path.join(path2source, 'SOURCE')
    with open(fname, 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if l.startswith('xs'):  # l[:2] == 'xs':
                lines[i] = xs_txt
            if l.startswith('zs'):  # l[:2] == 'zs':
                lines[i] = zs_txt
    
    with open(fname, 'w') as f:
        f.writelines(lines)
        
        
def set_own_time_source(path2source='./DATA', time_source_fname='time_source_own.txt'):
    """ Sets the time source to be of our own. """
    funtion_type_line = "time_function_type              = 8\n"
    funtion_fname_line = f"name_of_source_file             = {time_source_fname}       # Only for option 8 : file containing the source wavelet  YYYYYYYY\n"
    
    fname = os.path.join(path2source, 'SOURCE')
    with open(fname, 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if l.startswith('time_function_type'):
                lines[i] = funtion_type_line
            if l.startswith('name_of_source_file'):
                lines[i] = funtion_fname_line 
    
    with open(fname, 'w') as f:
        f.writelines(lines)


def undo_own_time_source(path2source='./DATA'):
    funtion_type_line = "time_function_type              = 1\n"
    funtion_fname_line = "name_of_source_file             = YYYYYYYY       # Only for option 8 : file containing the source wavelet  YYYYYYYY\n"
    
    fname = os.path.join(path2source, 'SOURCE')
    with open(fname, 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if l.startswith('time'):
                lines[i] = funtion_type_line
            if l.startswith('name'):
                lines[i] = funtion_fname_line 
    
    with open(fname, 'w') as f:
        f.writelines(lines)