import numpy as np

def scale_sig(sig, scale):
    """ 
    Scales signal conserving its mean. 
    """
    avg = np.mean(sig, axis=0)
    sig_0 = sig - avg
    sig_0 *= scale
    return sig_0 + avg