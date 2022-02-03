"""
WARPPERS FOR :mod:'matplotlib' TO VISUALIZE SEISMOGRAMS

- Guidance: https://github.com/fatiando/fatiando/blob/master/fatiando/vis/mpl.py
"""
import numpy as np
import matplotlib.pyplot as plt

def seismic_wiggle(section, dt, ranges=None, scale=1.):
    npts = section.shape[0]
    t = np.linspace(0, dt*npts, npts)
    if section.ndim > 1:
        n_traces = section.shape[1]
        if ranges is None:
            ranges = (0, n_traces)
        x0, x1 = ranges
        dx = (x1 - x0)/n_traces
        for i in range(n_traces):
            trace = section[:,i]
            x = x0 + i*dx
            plt.plot(x+trace, t, 'k')
            plt.fill_betweenx(t, x+trace, x, trace>0, color='k', alpha=0.75)
    else:
        plt.plot(section, t, 'k')
        plt.fill_betweenx(t, section, 0, section>0, color='k', alpha=0.75)
    plt.ylim(max(t), 0)