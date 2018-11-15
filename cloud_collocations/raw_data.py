"""
This module contains an interface to the raw collocation data.
It mainly provides convenience functions that simplify accessing
the extracted collocation data.
"""

import numpy as np
import os

from netCDF4 import Dataset

default_path = "/home/simonpf/Dendrite/UserAreas/Simon/cloud_collocations"

class RawData:

    def __init__(self, year, day, dn, basepath = default_path):
        day_str = str(day)
        day_str = "0" * (3 - len(day_str)) + day_str
        self.path = os.path.join(default_path, str(year), day_str)

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

