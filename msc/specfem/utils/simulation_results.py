import os
import numpy as np
import pandas as pd
import obspy
from tqdm import tqdm

from msc.specfem.utils.dtw_matching import dtw_path
from msc.specfem.utils.nmo_correction import nmo_correction
from msc.conv.conv_model import depth2time

from scipy import interpolate

class Simulation():
    def __init__(self, xmin_max, sim_file, parent_dir):
        self.path2mesh = os.path.join(parent_dir, 'MESH')
        if len(sim_file) > 0:
            self.path2output_files = os.path.join(parent_dir, 'OUTPUT_FILES_' + sim_file)
        else:
            self.path2output_files = os.path.join(parent_dir, 'OUTPUT_FILES')
        self.parent_dir = parent_dir
        self.uneven_dict = self.get_uneven_dict()
        
        self.xmin, self.xmax = xmin_max
        self.ztop, self.zbot = self.get_ztop_bot()
        
        self.mid = None
        self.offsets = None
        self.x_offsets = None
        self.get_middle_trace()
        
        self.vp_tomo = None
        self.rho_tomo = None
        self.load_tomo_models()
        
        self.time = None 
        self.dt = None
        self.data = None
        self.load_data()
        
        self.t0 = 0.5
        self.data_m = None
        self.mute_data()
        
        self.middle_trace = self.data_m[:,self.mid]
        
        self.vp_z = self.vp_tomo[:, self.mid]
        self.rho_z = self.rho_tomo[:, self.mid]
        self.vp_t = None
        self.rho_t = None
        self.depth2time()
        
        self.vp_rms = self.get_rms_vel()
        
    def get_ztop_bot(self):
        with open(os.path.join(self.path2output_files, 'simul_info.txt'), 'r') as f:
            lines = f.readlines()
            for i, l in enumerate(lines):
                if l.startswith('ztop'):
                    ztop, zbot = l.split('=')[-1].split(',')
                    ztop = float(ztop.strip())
                    zbot = float(zbot.strip())
                    
        return ztop, zbot
        
    def get_uneven_dict(self):
        uneven_df = pd.read_csv(os.path.join(self.parent_dir, 'uneven_dict.txt'), header=None, sep=' ', names=['dom_id', 'size'])
        uneven_dict = {}
        for dom_id, size in zip(uneven_df['dom_id'], uneven_df['size']):
            try:
                uneven_dict[int(dom_id)] = size
            except:
                uneven_dict[dom_id] = size
        
        return uneven_dict
    
    def get_middle_trace(self):
        stations = pd.read_csv(os.path.join(self.path2output_files, 'STATIONS'), header=None, delim_whitespace=True)
        self.offsets = stations[2].values

        # Source position
        source_fname = os.path.join(self.path2output_files, 'SOURCE')
        with open(source_fname, 'r') as f:
            lines = f.readlines()
            for l in lines:
                if l[:2] == 'xs':
                    xs = float(l.split('=')[1].split('#')[0].strip())
                if l[:2] == 'zs':
                    zs = float(l.split('=')[1].split('#')[0].strip())

        self.x_offsets = self.offsets - xs
        self.mid = np.argmin(np.abs(self.x_offsets))
    
    def load_tomo_models(self):
        tomo_models = np.load(os.path.join(self.path2output_files, 'tomo_models.npz'))
        self.vp_tomo = np.flip(tomo_models['vp_tomo'])
        self.rho_tomo = np.flip(tomo_models['rho_tomo'])
    
    def load_data(self):
        fname = 'Uz_file_single_d.su'
        traces = obspy.read(os.path.join(self.path2output_files, fname))
        self.time = traces[0].times()
        self.dt = traces[0].stats.delta
        self.data = np.array([trace.data for trace in traces]).T
        
    def mute_data(self):
        v_ini = self.vp_tomo[0, self.mid]
        t_dir = lambda x: self.t0 + x/v_ini

        self.data_m = self.data.copy()
        for j, x_ in enumerate(self.x_offsets):
            t_ = t_dir(abs(x_))
            self.data_m[self.time < t_, j] = 0.0
            
    def depth2time(self):
        nz = self.vp_tomo.shape[0]
        dz = (self.ztop - self.zbot)/nz
        vp_t, twt = depth2time(self.vp_z, self.vp_z, self.dt, dz, return_t=True)
        rho_t = depth2time(self.vp_z, self.rho_z, self.dt, dz, return_t=False)
        
        print(twt[-1], self.time[-1])
        
        # get times within the simulation time 
        tmax = self.data.shape[0] * self.dt
        mask = twt < tmax
        twt  = twt[mask]
        vp_t = vp_t[mask]
        rho_t = rho_t[mask]
        
        # interpolate so everything is aligned
        interpolator = interpolate.interp1d(twt, vp_t)
        self.vp_t = interpolator(self.time)
        interpolator = interpolate.interp1d(twt, rho_t)
        self.rho_t = interpolator(self.time)
        
    def get_rms_vel(self):
        tdiff = np.diff(self.time)
        vp2_t = []
        for i in range(1, self.data.shape[0]):
            dt_i = self.time[i] - self.time[i-1]
            vp2_t.append(self.vp_t[i]**2 * dt_i)
        vp_rms = np.sqrt(np.cumsum(vp2_t)/np.cumsum(tdiff))
        vp_rms = np.concatenate(([self.vp_t[0]], vp_rms))
        return vp_rms
    
    def interpolate2npts(self, data, npts_dwn):
        t_dwn = np.linspace(self.time[0], self.time[-1], npts_dwn)
        data_dwn = np.zeros((npts_dwn, data.shape[1]))
        for j in range(data.shape[1]):
            interpolator_ = interpolate.interp1d(self.time, data[:,j])
            data_dwn[:,j] = interpolator_(t_dwn)
            
        return data_dwn, t_dwn
    
    def run_conventional_nmo(self, npts_dwn=None):
        if npts_dwn is None:
            return nmo_correction(self.data_m, self.dt, self.x_offsets, self.vp_rms)
        else:
            data_dwn, t_dwn = self.interpolate2npts(self.data_m, npts_dwn)
            interpolator_rms = interpolate.interp1d(self.time, self.vp_rms)
            vp_rms_dwn = interpolator_rms(t_dwn)
            
            nmo = nmo_correction(data_dwn, np.mean(np.diff(t_dwn)), self.x_offsets, vp_rms_dwn)
            
            return nmo, t_dwn
        
    def dtw(self, data):
        data_dtw = data.copy()
        for k in tqdm(range(self.mid+1, data.shape[1])):
            path_R, _ = dtw_path(data_dtw[:,k-1], data_dtw[:,k])
            path_L, _ = dtw_path(data_dtw[:,2*self.mid-(k-1)], data_dtw[:,2*self.mid-k])
            for i, j in path_R:
                data_dtw[i, k] = data[j, k]
            for i, j in path_L:
                data_dtw[i, 2*self.mid-k] = data[j, 2*self.mid-k]
        
        return data_dtw
    
    def run_dtw_nmo(self, npts_dwn=None):
        if npts_dwn is None:
            return self.dtw(self.data_m)
        else:
            data_, t_dwn = self.interpolate2npts(self.data_m, npts_dwn)
            return self.dtw(data_), t_dwn