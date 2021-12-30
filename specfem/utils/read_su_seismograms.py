#!/usr/bin/env python

import argparse
from obspy import read
import matplotlib.pyplot as plt
import numpy as np
import specfem.plotting.nice_plot as nplt

def parse_args():
    """ This function run argparse (see
    https://docs.python.org/2/howto/argparse.html) to process the arguments
    given by the user along with sfsection. Define default behaviour if they
    are not given and help message when sfsection -h is run
    """
    parser = argparse.ArgumentParser(
        description='save seismic unix files as npz files')

    # Input file arguments
    parser.add_argument('path2su',
                        help='path to .su file')
    # parser.add_argument('path2npz',
    #                     help='path to the saved npz file')

    parser.add_argument('-show','--show_plot',
                        help='show a plot of seismic', action = 'store_true')
    parser.add_argument('-only_show','--only_show_plot',
                        help='only show a plot of seismic', action = 'store_true')
    parser.add_argument('-s', '--save', default='',
                         help='save plot (-s plots/seismic.png)')
    parser.add_argument('-t', '--title', default='',
                         help='plot title (-t clean model seismic)')
    parser.add_argument('-interp', '--interpolation', default='nearest',
                         help='interpolation method (-interp linear)')
    parser.add_argument('-vmm','--vminmax',default = '',
                        help='vmin,vmax for imshow')
    parser.add_argument('-filter','--filtering',default = '',
                        help='lowpass filtering frequency')

    return parser.parse_args()


def read_su_seismogram(seismic_traces):
    nr_traces  = len(seismic_traces)
    nr_samples = seismic_traces[0].stats.npts
    dt         = seismic_traces[0].stats.delta

    data_array = np.zeros((nr_samples, nr_traces))
    time_array = seismic_traces[0].times()

    for i in range(nr_traces):
        data_array[:,i] = seismic_traces[i].data

    return time_array, data_array

def plot_seis(time, data, vmin_max=False, title='', interpolation=''):
    if not vmin_max:
        vmin_max = np.percentile(data, 99.)
        
    extent = [1, data.shape[1], time[-1], time[0]]

    nplt.plotting_image(data,
                        aspect = 'auto',
                        vmin_max = (-vmin_max, vmin_max),
                        extent = extent,
                        title = title,
                        interpolation = interpolation)
    plt.ylabel('Receiver number')
    plt.xlabel('Time [s]')


if __name__ == "__main__":
    args = parse_args()  # Parse the arguments given along command sfsection

    seismic_traces = read(args.path2su)
    if args.filtering:
        seismic_traces.filter('lowpass', freq=float(args.filtering), zerophase=True)
    time_array, data_array = read_su_seismogram(seismic_traces)

    # if not args.only_show_plot:
    #     print('saving '+path2su+' to '+path2npz+'.npz')
    #     np.savez(path2npz,time_array=time_array,data_array=data_array)

    if args.vminmax:
        vminmax = float(args.vminmax)
    else:
        vminmax = False

    if not args.title:
        title = args.path2su.split('/')[-1]
    else:
        title = args.title
    plot_seis(time_array, data_array, title=title, interpolation=args.interpolation, vmin_max=vminmax)
    
    if args.save:
        plt.savefig(args.save)
        
    if args.show_plot:
        plt.show()
