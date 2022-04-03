import numpy as np
from scipy import linalg as la

class WSolver():
    def __init__(self, sdata, refl, noise=None):
        self.sdata = sdata
        self.refl = refl
        self.noise = noise
        
        self.R = la.toeplitz(refl)
        self.size = len(refl)
    
    @staticmethod
    def norm(arr):
        """ Normalize array """
        return arr/la.norm(arr)
    
    @staticmethod
    def _get_exp_diag(a: float, r: float, k: float, x: np.ndarray):
        """ Exponential-like diagonal: a·r^(k·x) """
        # NB: not normalised
        return a * r ** (k * x)
    
    def get_diag(self, a, r, k, **kwargs):
        """ Get the exponential decay to be used for regularization. """
        x = np.arange(self.size)
        diag = self._get_exp_diag(1.0, r, k, x)
        diag = a * la.norm(np.ones(self.size)) * self.norm(diag)
        if kwargs:
            # Include initial decay to the L-matrix
            a2 = kwargs['a2']
            r2 = kwargs['r2']
            k2 = kwargs['k2']
            assert k2 < 0., "Not a decay!"
            
            diag2 = self._get_exp_diag(a2, r2, k2, x)
            # diag2 = la.norm(np.ones(self.size)) * self.norm(diag2)
            diag += diag2 
            
        return diag
    
    def find_tikho_sol(self, eps, a, r, k, **kwargs):
        """ Higher Order Tikhonov Solution with exponential(s) as diagonal """
        diag = self.get_diag(a, r, k, **kwargs)
        L = np.diag(diag)
        
        RR_eps = np.dot(self.R.T, self.R) + eps**2 * L
        Rs = np.dot(self.R.T, self.sdata)
        w_est = np.linalg.solve(RR_eps, Rs)
        # w_est = la.inv(self.R.T @ self.R + eps**2 * L) @ self.R.T @ self.sdata
        
        return w_est
    
    def run_tikho_with_external_diag(self, eps, diag):
        L = np.diag(diag)
        RR_eps = np.dot(self.R.T, self.R) + eps**2 * L
        Rs = np.dot(self.R.T, self.sdata)
        w_est = np.linalg.solve(RR_eps, Rs)
        
        return w_est
    