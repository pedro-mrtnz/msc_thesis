import numpy as np
import matplotlib.pyplot as plt
from pandas.core.base import DataError
import tqdm as tqdm
from scipy import interpolate

class ConfigPlots:
    aspect        = 'auto'
    cmap          = 'Greys'
    interpolation = 'none'
    fig_size      = (8,8)


def reflection_time(t0, x, vnmo):
    """
    Calculates the travel-time of the reflected wave. Does not consider reflactions or
    changes in velocity.
    
    Args:
        t0   (float): the 0-offset (normal-incident) travel time.
        x    (float): the offset of the receiver.
        vnmo (float): the NMO velocity. 
    
    Returns: 
        The reflection travel-time.
    """
    return np.sqrt(t0**2 + x**2/vnmo**2)


def sample_trace(trace, t, dt):
    """
    Sample an amplitude at a given time using interpolation. 
    
    Args:
        trace (array): array containing the amplitudes of a single trace.
        t     (float): time at which I want to sample the amplitude. 
        dt    (float): sampling interval.
    
    Returns:
        amp   (float): the interpolated amplitude. Will be None if y is
                       beyond the end of the trace or if there are less 
                       than 2 points between t and the end. 
    """
    N = trace.size
    # Floor function will give the sample number right before the desired time
    before = int(np.floor(t/dt))
    # Use 4 samples around the time to interpolate
    samples = np.arange(before-1, before+3)
    if any(samples < 0) or any(samples >= N):
        amp = None
    else:
        times = samples*dt
        amps = trace[samples]
        interpolator = interpolate.CubicSpline(times, amps)
        amp = interpolator(t)

    return amp


def nmo_correction(cmp, dt, offsets, velocities):
    """
    Performs the NMO correction on the given CMP. 
    
    Args: 
        cmp        (2D array): the CMP gather (data) that we want to correct. 
        dt         (float)   : the sampling interval.
        offsets    (1D array): offsets of the traces.
        velocities (1D array): NMO velocities for each time. Should have the 
                               same number of elements as the CMP has samples. 
    Returns:
        nmo        (2D array): the NMO corrected gather.
    """
    nmo = np.zeros_like(cmp)
    n_samples = cmp.shape[0]
    times = np.arange(0, n_samples*dt, dt)
    for i, t0 in tqdm(enumerate(times)):
        for j, x in enumerate(offsets):
            t = reflection_time(t0, x, velocities[i])
            A = sample_trace(cmp[:, j], t, dt)          # Amplitude
            # If t is outside of the CMP time regime, A will be None
            if A is not None:
                nmo[i, j] = A

    return nmo

def plot_cmp(cmp_data, time, t0, x_offsets, vnmo):
    """
    Plot CMP data (not-corrected) with instance of reflected times for a given t0.

    Args:
        cmp_data  (2D array): CMP data 
        time      (1D array): Two Way Time 
        t0        (float)   : time instance upon which we compute the reflected times
        x_offsets (1D array): offsets positions
        vnmo      (float)   : normal velocity
    """
    refl_times = reflection_time(t0=t0, x=x_offsets, vnmo=vnmo)
    
    fig = plt.figure(figsize=ConfigPlots.fig_size)
    ax = fig.add_subplot(111)
    vmin_max = np.percentile(cmp_data, 99.)
    extent = [1, cmp_data.shape[1], time[-1], time[0]]
    ax.imshow(
        cmp_data,
        aspect = ConfigPlots.aspect,
        cmap = ConfigPlots.cmap,
        vmin = -vmin_max,
        vmax = vmin_max,
        interpolation = ConfigPlots.interpolation,
        exten = extent
    )
    ax.plot(np.arange(1, len(x_offsets)+1), refl_times, '-r', lw=3)
    ax.set(xlabel='Receiver number', ylabel='TWT [s]')
    
def plot_nmo(nmo_data, time):
    """ 
    Plot the NMO corrected data.
    """
    fig = plt.figure(figsize=ConfigPlots.fig_size)
    ax = fig.add_subplot(111)
    vmin_max = np.percentile(nmo_data, 99.)
    extent = [1, nmo_data.shape[1], time[-1], time[0]]
    ax.imshow(
        nmo_data,
        aspect = ConfigPlots.aspect,
        cmap = ConfigPlots.cmap,
        vmin = -vmin_max,
        vmax = vmin_max,
        interpolation = ConfigPlots.interpolation,
        exten = extent
    )
    ax.set(xlabel='Receiver number', ylabel='TWT [s]')
