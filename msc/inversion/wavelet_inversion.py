import numpy as np
from scipy import linalg as la

class HigherOrderTikhonov():
    def __init__(self, sdata, refl, noise=None):
        self.sdata = sdata
        self.refl = refl
        self.noise = noise
        
        self.r_toeplz = la.toeplitz(refl)
    
    @staticmethod
    def get_toeplitz(a):
        return la.toeplitz(a)
    
    def find_tikho_sol(self, eps, ratio, cte):
        size = self.r_toeplz.shape[0]
        ncte = np.sqrt(size)
        
        diag = np.exp(ratio*np.arange(size)) - cte 
        L = np.diag(diag)
        w_est = la.inv(self.r_toeplz.T @ self.r_toeplz + eps**2 * L) @ self.r_toeplz.T @ self.sdata
        
        return w_est, L