"""
WARPPERS FOR :mod:'matplotlib' TO VISUALIZE SEISMOGRAMS

- Guidance: https://github.com/fatiando/fatiando/blob/master/fatiando/vis/mpl.py
"""
import numpy as np
import matplotlib.pyplot as plt

import msc.plotting.nice_plot as nplt

def seismic_wiggle(section, dt, scale=1.0, color='k', ranges=None):
    """Plot seismic section as wiggles.

    Args:
        section (np.ndarray): 1D or 2D array with the trace(s)
        dt      (float)     : time sample rate in seconds.
        scale   (float)     : scale fator. Defaults to 1.0.
        ranges  (tuple)     : (x1, x2). Defaults to None.
    """
    npts = section.shape[0]
    t = np.linspace(0, dt*npts, npts)
    if section.ndim > 1:
        n_traces = section.shape[1]
        if ranges is None:
            ranges = (0, n_traces)
        x0, x1 = ranges
        dx = (x1 - x0)/n_traces
        for i in range(n_traces):
            trace = section[:,i]*scale*dx
            x = x0 + i*dx
            plt.plot(x+trace, t, color)
            plt.fill_betweenx(t, x+trace, x, trace>0, color=color, alpha=0.75)
    else:
        trace = section*scale
        plt.plot(trace, t, color)
        plt.fill_betweenx(t, trace, 0, trace>0, color=color, alpha=0.75)
    plt.ylim(max(t), 0)


def seismic_image(section, dt, fig=None, ranges=None, cmap=None, aspect=None, vmin_max=None, cbar_pad=0.0, cbar_ori='horizontal', cbar_power=False):
    """Plot seismic section (2D array) as an image.

    Args:
        section  (2D array): matrix of traces. 
        dt       (float)   : time sample rate in seconds. 
        ranges   (tuple)   : min and max horizontal coordinate values. Defaults to None.
        cmap     (str)     : colormap to be used. Defaults to None.
        aspect   (float)   : aspect ratio between axes. Defaults to None.
        vmin_max (tuple)   : min and max values for imshow. Defaults to None.
    """
    npts, n_traces = section.shape
    t = np.linspace(0, dt*npts, npts)
    if ranges is None:
        ranges = (0, n_traces)
    x0, x1 = ranges
    extent = [x0, x1, t[-1], t[0]]
    if aspect is None:
        # Guarantee a rectangular picture
        aspect = np.round((x1 - x0)/np.max(t))
        aspect -= aspect*0.2
    if vmin_max is None:
        vmin_max = np.percentile(section, 99.)
        vmin, vmax = -vmin_max, vmin_max
        # scale = np.abs([section.max(), section.min()]).max()
        # vmin = -scale
        # vmax = scale
    else:
        vmin, vmax = vmin_max
    if cmap is None:
        cmap = 'nice'
        
    nplt.plotting_image(section,
                        aspect = aspect,
                        vmin_max = (vmin, vmax),
                        extent = extent,
                        cmap = cmap,
                        fig = fig,
                        cbar_pad = cbar_pad,
                        cbar_ori = cbar_ori,
                        cbar_power = cbar_power)
    plt.xlabel('Receiver number')
    plt.ylabel('Time (s)')
