"""
ZERO-OFFSET CONVOLUTIONAL SEISMIC MODELING.

Given a depth model we obtain a seismic zero-offset convolutional gather.
"""
import numpy as np
from scipy import interpolate


def resampling(model, tmax, twt_z, twt_t, dt, dt_dwn):
    """Resamples input data to adjust it after time conversion
    with chosen time sample rate, dt.

    Args:
        model   (1D array): model field in in depth domain.
        tmax    (float)   : largest time data point from depth domain. 
        twt_z   (1D array): Two-Way-Time in depth domain. 
        twt_t   (1D array): Two-Way-Time in depth domain (with dt_dwn sample rate).
        dt      (float)   : model time sample rate.
        dt_dwn  (float)   : dowsized time sample rate.
        
    Returns:
        model_t (1D array): model field in time domain. 
    """
    model_tdwn = np.ones_like(twt_t)
    t0_idx = int(np.ceil(twt_z[0]/dt_dwn))
    tf_idx = int(np.ceil(twt_z[-1]/dt_dwn)) - 1

    tck = interpolate.interp1d(twt_z, model)
    model_tdwn[t0_idx:tf_idx] = tck(twt_t[t0_idx:tf_idx])
    model_tdwn[tf_idx:] = model_tdwn[tf_idx-1]
    model_tdwn[:t0_idx] = model[0]

    # Resampling from dt_dwn to dt
    resampl = int(dt/dt_dwn)
    nt = int(np.ceil(tmax/dt))
    model_t = np.zeros(nt)
    model_t[0] = model_tdwn[0]
    for i in range(1, nt):
        model_t[i] = model_tdwn[resampl*i]

    return model_t


def depth2time(vp, model, dt, dz):
    """Converts depth property model to time model.

    Args:
        vp      (1D array): velocity model.
        model   (1D array): model field we want in time domain.
        dt      (float)   : time sample rate.
        dz      (float)   : depth sample rate.

    Returns:
        model_t (1D array): model field in time domain. 
    """
    nz = vp.shape[0]
    dt_dwn = dt/10.
    if dt_dwn > dz/np.max(vp):
        dt_dwn = (dz/np.max(vp))/10.

    # TWT in depth domain
    twt_z = np.zeros(nz)
    twt_z[0] = 2.0*dz/vp[0]
    for i in range(1, nz):
        twt_z[i] = twt_z[i-1] + 2.0*dz/vp[i]

    # TWT in time domain
    tmax = twt_z[-1]
    nt_dwn = int(np.ceil(tmax/dt_dwn))
    twt_t = np.zeros(nt_dwn)
    for i in range(1, nt_dwn):
        twt_t[i] = twt_t[i-1] + dt_dwn

    # Resample model property according to time (e.g. velocity)
    model_t = resampling(model, tmax, twt_z, twt_t, dt, dt_dwn)

    return model_t


def list2array(obj):
    if isinstance(obj, list):
        return np.array(obj)
    else:
        return obj


def get_reflectivity(vel, rho):
    """Computes the reflectivity response by means of the impedance.

    Args:
        vel (1D array): velocity model.
        rho (1D array): density model.

    Returns:
        R   (1D array): reflectivity profile.
    """
    Z = list2array(vel) * list2array(rho)
    R = (Z[1:] - Z[:-1])/(Z[1:] + Z[:-1])
    return R


def ricker_wavelet(f0, dt):
    assert f0 < 0.2 * 1/(2*dt), "Frequency too high for the dt chosen."
    nw = 2.2/f0/dt
    nw = 2*int(np.floor(nw/2)) + 1
    nc = int(np.floor(nw/2))
    k = np.arange(1, nw+1)
    a = (nc - k+1)*f0*dt*np.pi
    b = a**2

    ricker = (1 - b**2)*np.exp(-b)
    return ricker


def convolutional_model(rc, f0, dt):
    """Computes de convolutional seismogram by convoluting the
    reflectivity with the Ricker wavelet.

    Args:
        rc      (1D array): reflectivity profile. 
        f0      (float)   : dominant frequency of the Ricker wavelet.
        dt      (float)   : time sample rate.

    Returns:
        synth_t (1D array): convolved seismogram. 
    """
    w = ricker_wavelet(f0, dt)
    synth_t = np.zeros_like(rc)
    if np.shape(rc)[0] >= len(w):
        synth_t = np.convolve(rc, w, mode='same')
    else:
        aux = int(np.floor(len(w)/2.))
        synth_t = np.convolve(rc, w, mode='full')[aux:-aux]

    return synth_t
