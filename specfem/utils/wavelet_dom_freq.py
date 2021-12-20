"""
SCRIPT THAT SETS THE DOMINANT FREQUENCY OF THE WAVELET IN THE 'SOURCE' SCRIPT.
"""
import os

def set_dom_freq(f0, path2source='./DATA'):
    fname = os.path.join(path2source, 'SOURCE')
    with open(fname, 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            line_num = i
        line_txt = f'f0                              = {f0}           # dominant source frequency (Hz) if not Dirac or Heaviside\n'
        lines[line_num] = line_txt
    
    with open(fname, 'w') as f:
        f.writelines()