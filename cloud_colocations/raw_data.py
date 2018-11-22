"""
This module contains an interface to the raw colocation data.
It mainly provides convenience functions that simplify accessing
the extracted colocation data.
"""

import numpy as np
import os

from netCDF4 import Dataset

import cloud_colocations.products as products
from cloud_colocations.products import FileCache
from datetime import datetime


default_path = "/home/simonpf/Dendrite/UserAreas/Simon/cloud_colocations"

def profile_time_to_datetime(time):
    """
    Convert CALIOP profile time to datetime.

    Arguments:

        time(:code:`numpy.float`): Profile UTC time as contained in the
        CALIOP 01kmclay product.

    Returns:

        :code:`datetime.datetime` object representing the given time.
    """
    time_i = int(np.floor(time))

    day   = time_i % 10
    month = (time_i // 100) % 10
    year  = 2000 + time_i // 10000

    print(year, month, day)

    fod    = time - np.floor(time) 
    hour   = int(np.floor(fod * 24.0))
    minute = int(np.floor((fod * 24.0 - hour) * 60.0))
    second = int(np.floor(((fod * 24.0 - hour) * 60.0 - minute) * 60.0))

    print(hour, minute, second)
    t = datetime(year = year, month = month, day = day,
                 hour = hour, minute = minute, second = second)
    print(t.strftime("%j"))
    return t

################################################################################
# RawData Class
################################################################################

class RawData:

    def __init__(self, year, day, dn,
                 basepath = default_path,
                 cache = "cache"):

        day_str = str(day)
        day_str = "0" * (3 - len(day_str)) + day_str
        self.path = os.path.join(basepath, str(year), day_str)

        print("Loading files from ", self.path)

        filename = "meta_" + str(dn) + ".nc"
        self.meta_data = Dataset(os.path.join(self.path, filename), mode = "r")

        # Load the MODIS data.
        filename = "modis_" + str(dn) + ".nc"
        self.modis_data = Dataset(os.path.join(self.path, filename), mode = "r")
        filename = "modis_ss_5_" + str(dn) + ".nc"
        self.modis_ss_5_data = Dataset(os.path.join(self.path, filename), mode = "r")
        filename = "modis_ss_11_" + str(dn) + ".nc"
        self.modis_ss_11_data = Dataset(os.path.join(self.path, filename), mode = "r")

        # Load the CALIOP data.
        filename = "caliop_" + str(dn) + ".nc"
        self.caliop_data = Dataset(os.path.join(self.path, filename), mode = "r")
        filename = "caliop_ss_5_" + str(dn) + ".nc"
        self.caliop_ss_5_data = Dataset(os.path.join(self.path, filename), mode = "r")
        filename = "caliop_ss_11_" + str(dn) + ".nc"
        self.caliop_ss_11_data = Dataset(os.path.join(self.path, filename), mode = "r")

        self.n_colocations = self.modis_data.variables["band_1"].shape[0]

    def _get_modis_granule(self, index):

        t = self.meta_data.variables["time"][index]
        t = profile_time_to_datetime(t)
        modis_granule     = products.modis.get_file_by_date(t)
        modis_geo_granule = products.modis_geo.get_file_by_date(t)

        modis_granule     = products.modis.download_file(modis_granule)
        modis_geo_granule = products.modis.download_file(modis_geo_granule)

        self.modis_granule = modis_granule
        self.modis_geo_granule = modis_geo_granule

        return t

