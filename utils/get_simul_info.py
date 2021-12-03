import os
import argparse
import obspy
import numpy as np

def get_simul_info(path2file: str):
    """
    Funciton that gets information about the simulation contained in the SU files.
    
    Args:
        path2file (str): path to the SU file
    
    Return:
        Information like: number of traces, number of samples, dt, elapsed times
    """
    seismic_traces = obspy.read(path2file)
    n_traces  = len(seismic_traces)
    n_samples = seismic_traces[0].stats.npts
    dt        = seismic_traces[0].stats.delta
    
    return seismic_traces, n_traces, n_samples, dt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Information from SU files.')
    parser.add_argument('path2file', type=str, 
                        default='./OUTPUT_FILES/Uz_file_single_d.su',
                        nargs='?', const=1)
    
    args = parser.parse_args()
    
    _, n_traces, n_samples, dt = get_simul_info(args.path2file)
    
    print(' ')
    print(f'Number of traces: {n_traces}')
    print(f'Number of samples: {n_samples}')
    print(f'Time step dt: {dt}')
    print(f'Elapsed time: {n_samples*dt} seconds')
    print(' ')
    
    
    

    