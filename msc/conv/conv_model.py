"""
ZERO-OFFSET CONVOLUTIONAL SEISMIC MODELING.

Given a depth model we obtain a seismic zero-offset convolutional gather.

- Check: https://github.com/fatiando/fatiando/blob/master/fatiando/seismic/conv.py
"""
import numpy as np
from scipy import interpolate
from scipy import linalg as la
from scipy.ndimage import gaussian_filter
from msc.specfem.multilayer.create_tomography_noisy import get_noise_snr


def resampling(model, tmax, twt_z, twt_tdwn, dt, dt_dwn):
    """Resamples input data to adjust it after time conversion
    with chosen time sample rate, dt.

    Args:
        model    (1D array): model field in in depth domain.
        tmax     (float)   : largest time data point from depth domain. 
        twt_z    (1D array): Two-Way-Time in depth domain. 
        twt_tdwn (1D array): Two-Way-Time in depth domain (with dt_dwn sample rate).
        dt       (float)   : model time sample rate.
        dt_dwn   (float)   : dowsized time sample rate.
        
    Returns:
        model_t (1D array): model field in time domain. 
    """
    model_tdwn = np.ones_like(twt_tdwn)
    t0_idx = int(np.ceil(twt_z[0]/dt_dwn))
    tf_idx = int(np.ceil(twt_z[-1]/dt_dwn)) - 1

    tck = interpolate.interp1d(twt_z, model)
    model_tdwn[t0_idx:tf_idx] = tck(twt_tdwn[t0_idx:tf_idx])
    model_tdwn[tf_idx:] = model_tdwn[tf_idx-1]
    model_tdwn[:t0_idx] = model[0]

    # Resampling from dt_dwn to dt
    resampl = int(dt/dt_dwn)
    nt = int(np.ceil(tmax/dt))  # int(len(twt_t)/resampl)
    model_t, twt_t = np.zeros(nt), np.zeros(nt)
    model_t[0] = model_tdwn[0]
    twt_t[0] = twt_tdwn[0]
    for i in range(1, nt):
        model_t[i] = model_tdwn[resampl*i]
        twt_t[i] = twt_tdwn[resampl*i]

    return model_t, twt_t


def interpolate2npts(npts, time, model, kind):
    interpolator = interpolate.interp1d(time, model, kind=kind)
    t_npts = np.linspace(time[0], time[-1], npts)
    model_npts = interpolator(t_npts)
    
    return model_npts, t_npts


def depth2time(vp, model, dt, dz, npts=None, kind='linear', return_t=False):
    """Converts depth property model to time model.

    Args:
        vp      (1D array): velocity model.
        model   (1D array): model field we want in time domain.
        dt      (float)   : time sample rate.
        dz      (float)   : depth sample rate.
        npts    (int)     : number of samples from simulation.

    Returns:
        model_t (1D array): model field in time domain. 
    """
    nz = vp.shape[0]
    t_fact = 10.0
    dt_dwn = dt/t_fact
    if dt_dwn > dz/np.max(vp):
        dt_dwn = (dz/np.max(vp))/t_fact
    
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
    model_t, twt_t = resampling(model, tmax, twt_z, twt_t, dt, dt_dwn)
    if npts is not None:
        model_t, twt_t = interpolate2npts(npts, twt_t, model_t, kind)

    if return_t:
        return model_t, twt_t
    else:
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


def get_spectrum(signal, dt, npts=None):
    from scipy.fft import rfft, rfftfreq
    
    ftrans = rfft(signal, npts)
    ampli = np.abs(ftrans)
    phase = np.angle(ftrans)
    power = 20 * np.log10(ampli)
    if npts is None:
        f = rfftfreq(len(signal), d=dt)
    else:
        f = rfftfreq(npts, d=dt)
    
    return ampli, power, phase, f


# def ricker_wavelet(f0, dt):
#     assert f0 < 0.2 * 1/(2*dt), "Frequency too high for the dt chosen."
#     nw = 2.2/f0/dt
#     nw = 2*int(np.floor(nw/2)) + 1
#     nc = int(np.floor(nw/2))
#     k = np.arange(1, nw+1)
#     a = (nc - k+1)*f0*dt*np.pi
#     b = a**2

#     ricker = (1 - b**2)*np.exp(-b)
#     return ricker


def _get_time(duration, dt, shift):
    duration += shift
    n = int(duration/dt)
    odd = n % 2
    k = int(10**-np.floor(np.log10(dt)))
    dti = int(k * dt)
    
    if odd:
        t = np.arange(n)
    else:
        t = np.arange(n + 1)
    t -= t[-1] // 2
    
    return dti * t / k - shift/2


def ricker_wavelet(f0, dt, duration=None, return_t=True, shift=0.):
    if duration is None:
        duration = 2.2/f0
    if shift == -1:
        shift = duration
    t = _get_time(duration, dt, shift)
    a = (np.pi * f0)**2
    w = (1 - 2*a*t**2)*np.exp(-a*t**2)
    
    if return_t:
        return w, t
    else:
        return w


def ricker_own(time, f0):
    """ Variable time already takes into account dt """
    t_w = 2.5/f0
    t = time - t_w/2
    a = (np.pi * f0)**2
    w = (1 - 2*a*t**2)*np.exp(-a*t**2)
    return w, t


def convolutional_model(rc, f0, dt, own_w=None, noise_level=None, shift=0., return_t=False):
    """Computes de convolutional seismogram by convoluting the
    reflectivity with the Ricker wavelet.

    Args:
        rc      (1D array): reflectivity profile. 
        f0      (float)   : dominant frequency of the Ricker wavelet.
        dt      (float)   : time sample rate.
        shift   (float)   : shift if needed to match simulation. 

    Returns:
        synth_t (1D array): convolved seismogram. 
    """
    if own_w is None:
        w, twav = ricker_wavelet(f0, dt, return_t=True, shift=shift)
    else:
        w = own_w
        
    synth_t = np.zeros_like(rc)
    if np.shape(rc)[0] >= len(w):
        synth_t = np.convolve(rc, w, mode='same')
        # R = la.convolution_matrix(rc, len(w), mode='same')
        # synth_t = np.dot(R, w)
    else:
        aux = int(np.floor(len(w)/2.))
        synth_t = np.convolve(rc, w, mode='full')[aux:-aux]
        # R = la.convolution_matrix(rc, len(w), mode='full')
        # synth_t = np.dot(R, w)[aux:-aux]
        
    if noise_level is not None:
        noise = get_noise_snr(synth_t, noise_level)
        synth_t = gaussian_filter(synth_t + noise, sigma=1.)

    if return_t is False:
        return synth_t
    else:
        return synth_t, twav
