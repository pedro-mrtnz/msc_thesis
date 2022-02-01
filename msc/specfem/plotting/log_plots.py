""" 
SCRIPT TO PLOT SEISMIC LOGS
"""
import numpy as np
import matplotlib.pyplot as plt


# def plot_logs(refl=None, wavelet=None, s_traces=None, rho=None, vp=None, td_rel=None):
#     """
#     Plots all the features that pertain to the model logs. It will plot only those 
#     which are not None. 
    
#     Args:
#         refl     (array)        : reflectivity profile. 
#         wavelet  (array)        : wavelet.
#         s_traces (array or list): seismic traces to plot. It can be a list of traces.
#         rho      (array)        : density log. 
#         vp       (array)        : velocity log. 
#         td_rel   (array)        : time-depth relationship. 
#     """
#     collect_features = {}
#     if refl is not None:
#         collect_features['Reflectivity'] = refl
#     if wavelet is not None:
#         if isinstance(wavelet, dict):
#             f0 = wavelet['f0']
#             wav = wavelet['wavelet']
#             collect_features[f'Wavelet (F = {f0}Hz)'] = wav
#         else:
#             collect_features['Wavelet'] = wavelet
