from datetime import datetime, timedelta
import numpy as np

import scipy as sp
import scipy.signal

def caliop_tai_to_datetime(tai):
    """
    Converts CALIOP profile time in IAT format to a datetime object.

    Arguments:

        tai(float): The CALIOP profile time.

    Returns:

        A datetime object representing the profile time.
    """
    t0 = datetime(1993, 1, 1)
    dt = timedelta(seconds = tai)

    return t0 + dt

def block_average(data, subsampling_factor):
    """
    Subsample :code:`data` with a given :code:`subsampling_factor` by computing
    strided block averages.
    """
    sf = subsampling_factor
    k = np.ones((sf, sf)) / sf ** 2
    data = sp.signal.convolve2d(data, k, mode = "valid")[::sf, ::sf]
    return data
