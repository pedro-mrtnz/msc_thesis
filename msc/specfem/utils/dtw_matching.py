import warnings
import numpy as np
from numba import njit, prange

GLOBAL_CONSTRAINT_CODE = {None: 0, "": 0, "itakura": 1, "sakoe_chiba": 2}

# Fixme: implement path limited warpings or window

"""
Check:
- Implementation: https://github.com/tslearn-team/tslearn/blob/main/tslearn/metrics/dtw_variants.py
- Tutorial      : https://rtavenar.github.io/blog/dtw.html#dynamic-time-warping
"""

def ts_size(ts):
    """
    Returns actual time series size.

    Examples:
        >>> ts_size([[1, 2],
        ...          [2, 3],
        ...          [3, 4],
        ...          [np.nan, 2],
        ...          [np.nan, np.nan]])
        4
        >>> ts_size([np.nan, 3, np.inf, np.nan])
        3
    """
    ts_ = to_time_series(ts)
    sz = ts_.shape[0]
    while sz > 0 and np.all(np.isnan(ts_[sz - 1])):
        sz -= 1
    return sz
        

def to_time_series(ts, remove_nans=False):
    """
    Transforms a time-dependent signal so that we deal with the same format all the time.

    Args:
        ts          (array): time series to be transformed.
        remove_nans (bool) : whether trailing NaNs at the end of the time series should be
                             removed or not. Defaults to False.
    Returns:
        ts_out      (array): transformed time series. 
    """
    ts_out = np.array(ts, copy=True)
    if ts_out.ndim <= 1:
        ts_out = ts_out.reshape((-1,1))
    if ts_out.dtype != np.float:
        ts_out = ts_out.astype(np.float)
    if remove_nans:
        ts_out = ts_out[:ts_size(ts_out)]
        
    return ts_out


@njit()
def _local_squared_dist(x, y):
    dist = 0.0
    for di in range(x.shape[0]):
        diff = (x[di] - y[di])
        dist += diff * diff
    return dist


@njit()
def njit_accumulated_matrix(s1, s2, mask):
    """
    Compute the accumulated cost matrix score between two time series.

    Args:
        s1   (1D array): first time series. Shape (sz1,)
        s2   (1D array): second time series. Shape (sz2,)
        mask (2D array): mask with shape (sz1, sz2). Unconsidered cells
                         must have infinite values. 
    Return:
        mat  (2d array): accumulated cost matrix. Shape (sz1, sz2).
    """
    n, m = s1.shape[0], s2.shape[0]
    cum_sum = np.inf * np.ones((n+1, m+1))
    cum_sum[0,0] = 0.0
    for i in range(n):
        for j in range(m):
            if np.isfinite(mask[i,j]):
                cum_sum[i+1, j+1] = _local_squared_dist(s1[i], s2[j])
                cum_sum[i+1, j+1] += min(cum_sum[i,j],
                                         cum_sum[i+1,j],
                                         cum_sum[i,j+1])
    
    return cum_sum[1:, 1:]
    

@njit(nogil=True)
def njit_dtw(s1, s2, mask):
    """
    Compute the accumulated cost score score between two time series.

    Args:
        s1         (1D array): first time series. Shape (sz1,)
        s2         (1D array): second time series. Shape (sz2,)
        mask       (2D array): mask with shape (sz1, sz2). Unconsidered cells
                               must have infinite values. 
    Return:
        dtw_score  (float)   : DTW score between both time series.
    """
    cum_sum = njit_accumulated_matrix(s1, s2, mask)
    return np.sqrt(cum_sum[-1,-1])


@njit()
def _return_path(acc_cost_mat):
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1-1, sz2-1)]
    while path[-1] != (0,0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j-1))
        elif j == 0:
            path.append((i-1, 0))
        else:
            arr = np.array([acc_cost_mat[i-1][j-1],
                            acc_cost_mat[i-1][j],
                            acc_cost_mat[i][j-1]])
            argmin = np.argmin(arr)
            if argmin == 0:
                path.append((i-1, j-1))
            elif argmin == 1:
                path.append((i-1, j))
            else:
                path.append((i, j-1))
    
    return path[::-1]


def dtw_path(s1, s2, global_constraint=None, sakoe_chiba_radius=None, itakura_max_slope=None):
    """
    Computes DTW (Dynamical Time Warping) similarity measure between time series
    and returns both the path and the similarity. 

    Args:
        s1                (1D array)      : time series. Shape (sz1,).
        s2                (1D array)      : time series. Shape (sz2,).
        global_constraint (str or None)   : global constraint to restrict admissible
                                            paths for DTW. Strings can be "itakura" or
                                            "sakoe_chiba". Defaults to None.
        sakoe_chiba_radius (int or None)  : radius to be used for Sakoe-Chiba band global
                                            constraint. If None and 'global_constraint' is 
                                            set to "sakoe_chiba", a radius of 1 is used. 
                                            Defaults to None.
        itakura_max_slope  (float or None): maximum slope for the Itakura parallelogram 
                                            constraint. If None and 'global_constraint' is 
                                            set to "itakura", a maximum slope of 2 is used. 
                                            Defaults to None.
    Returns:
        List of paths and the similarity score. 
        
    NB: If both 'sakoe_chiba_radius' and 'itakura_max_slope' are set, 'global_constraint' is
    used to infer which constraint to use among the two. In this case, if 'global_constraint' 
    corresponds to no global constraint, a 'RuntimeWarning' is raised and no global constraint 
    is used.
    """
    s1 = to_time_series(s1, remove_nans=True)
    s2 = to_time_series(s2, remove_nans=True)
    if len(s1) == 0 or len(s2) == 0:
        raise ValueError("One of the input time series contains only NaNs or has zero length.")
    if s1.shape[0] != s2.shape[1]:
        raise ValueError("All input time series must have the same feature size.")

    mask = compute_mask(
        s1, 
        s2, 
        GLOBAL_CONSTRAINT_CODE[global_constraint], 
        sakoe_chiba_radius, 
        itakura_max_slope
    )
    acc_cost_mat = njit_accumulated_matrix(s1, s2, mask=mask)
    path = _return_path(acc_cost_mat)
    
    return path, np.sqrt(acc_cost_mat[-1,-1])


def compute_mask(s1, s2, global_constraint=0, sakoe_chiba_radius=None, itakura_max_slope=None):
    """
    Computes the mask (region constraint).

    Args:
        s1                (1D array)      : time series. Shape (sz1,).
        s2                (1D array)      : time series. Shape (sz2,).
        global_constraint (int or None)   : global constraint to restrict admissible
                                            paths for DTW. Integers can be
                                            - 1 for "itakura"
                                            - 2 for "sakoe_chiba"
                                            - 0 for no constraint
                                            Defaults to 0.
        sakoe_chiba_radius (int or None)  : radius to be used for Sakoe-Chiba band global
                                            constraint. If None and 'global_constraint' is 
                                            set to "sakoe_chiba", a radius of 1 is used. 
                                            Defaults to None.
        itakura_max_slope  (float or None): maximum slope for the Itakura parallelogram 
                                            constraint. If None and 'global_constraint' is 
                                            set to "itakura", a maximum slope of 2 is used. 
                                            Defaults to None.
    Returns:
        Mask (array). 
        
    NB: If both 'sakoe_chiba_radius' and 'itakura_max_slope' are set, 'global_constraint' is
    used to infer which constraint to use among the two. In this case, if 'global_constraint' 
    corresponds to no global constraint, a 'RuntimeWarning' is raised and no global constraint 
    is used.
    """
    if isinstance(s1, int) and isinstance(s2, int):
        sz1, sz2 = s1, s2
    else:
        sz1 = s1.shape[0]
        sz2 = s2.shape[0]
        
    if (global_constraint == 0) and (sakoe_chiba_radius is not None) and (itakura_max_slope is not None):
        raise RuntimeWarning("'global_constraint' is not set for DTW, but both "
                             "'sakoe_chiba_radius' and 'itakura_max_slope' are "
                             "set, hence 'global_constraint' cannot be inferred "
                             "and no global constraint will be used.")
    
    if global_constraint == 2 or (global_constraint == 0 and sakoe_chiba_radius is not None):
        if sakoe_chiba_radius is None:
            sakoe_chiba_radius = 1
        mask = sakoe_chiba_mask(sz1, sz2, radius=sakoe_chiba_radius)
    elif global_constraint == 1 or (global_constraint == 0 and itakura_max_slope is not None):
        if itakura_max_slope is None:
            itakura_max_slope = 2.0
        mask = itakura_mask(sz1, sz2, max_slope=itakura_max_slope)
    else:
        mask = np.zeros((sz1, sz2))
    
    return mask


@njit()
def _njit_itakura_mask(sz1, sz2, max_slope=2.0):
    """
    Computes the itakura mask without checking that the constraints are feasible. In most
    cases, you should use itakura_mask instead. 
    """
    min_slope = 1 / float(max_slope)
    max_slope *= (float(sz1) / float(sz2))
    min_slope *= (float(sz1) / float(sz2))

    lower_bound = np.empty((2, sz2))
    lower_bound[0] = min_slope * np.arange(sz2)
    lower_bound[1] = ((sz1 - 1) - max_slope * (sz2 - 1)
                      + max_slope * np.arange(sz2))
    lower_bound_ = np.empty(sz2)
    for i in prange(sz2):
        lower_bound_[i] = max(round(lower_bound[0, i], 2),
                              round(lower_bound[1, i], 2))
    lower_bound_ = np.ceil(lower_bound_)

    upper_bound = np.empty((2, sz2))
    upper_bound[0] = max_slope * np.arange(sz2)
    upper_bound[1] = ((sz1 - 1) - min_slope * (sz2 - 1)
                      + min_slope * np.arange(sz2))
    upper_bound_ = np.empty(sz2)
    for i in prange(sz2):
        upper_bound_[i] = min(round(upper_bound[0, i], 2),
                              round(upper_bound[1, i], 2))
    upper_bound_ = np.floor(upper_bound_ + 1)

    mask = np.full((sz1, sz2), np.inf)
    for i in prange(sz2):
        mask[int(lower_bound_[i]):int(upper_bound_[i]), i] = 0.
        
    return mask


def itakura_mask(sz1, sz2, max_slope=2.):
    """ Computes the Itakura mask. """
    mask = _njit_itakura_mask(sz1, sz2, max_slope=max_slope)

    # Post-check
    raise_warning = False
    for i in prange(sz1):
        if not np.any(np.isfinite(mask[i])):
            raise_warning = True
            break
    if not raise_warning:
        for j in prange(sz2):
            if not np.any(np.isfinite(mask[:, j])):
                raise_warning = True
                break
    if raise_warning:
        warnings.warn("'itakura_max_slope' constraint is unfeasible "
                      "(ie. leads to no admissible path) for the "
                      "provided time series sizes",
                      RuntimeWarning)

    return mask


@njit()
def sakoe_chiba_mask(sz1, sz2, radius=1):
    """Compute the Sakoe-Chiba mask. """
    mask = np.full((sz1, sz2), np.inf)
    if sz1 > sz2:
        width = sz1 - sz2 + radius
        for i in prange(sz2):
            lower = max(0, i - radius)
            upper = min(sz1, i + width) + 1
            mask[lower:upper, i] = 0.
    else:
        width = sz2 - sz1 + radius
        for i in prange(sz1):
            lower = max(0, i - radius)
            upper = min(sz2, i + width) + 1
            mask[i, lower:upper] = 0.
            
    return mask
        

# My own ones...
def get_dtw_matrix(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.inf * np.zeros((n+1,m+1))
    dtw_matrix[0,0] = 0.0
    for i in range(n):
        for j in range(m):
            best_prev_cost = min(
                dtw_matrix[i,j],
                dtw_matrix[i+1,j],
                dtw_matrix[i,j+1]  
            )
            cost = abs(s[i] - t[j])
            dtw_matrix[i+1, j+1] = cost + best_prev_cost
    
    return dtw_matrix

def get_dtw_matrix2(s, t, w):
    n, m = len(s), len(t)
    w = np.max([w, abs(n-m)])
    dtw_matrix = np.inf * np.ones((n+1, m+1))
    dtw_matrix[0,0] = 0.0

    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            dtw_matrix[i, j] = 0

    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            cost = abs(s[i-1] - t[j-1])
            last_min = np.min([
                dtw_matrix[i-1, j], 
                dtw_matrix[i, j-1], 
                dtw_matrix[i-1, j-1]
            ])
            dtw_matrix[i, j] = cost + last_min
            
    return dtw_matrix

