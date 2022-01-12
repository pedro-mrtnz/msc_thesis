import numpy as np
from scipy import interpolate

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
    for i, t0 in enumerate(times):
        for j, x in enumerate(offsets):
            t = reflection_time(t0, x, velocities[i])
            A = sample_trace(cmp[:, j], t, dt)          # Amplitude
            # If t is outside of the CMP time regime, A will be None
            if A is not None:
                nmo[i, j] = A

    return nmo
