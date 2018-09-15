from datetime import datetime, timedelta
import numpy as np
from cloud_collocations.utils import caliop_tai_to_datetime
from cloud_collocations import products

from cloud_collocations.formats import Caliop01kmclay, ModisMyd021km, ModisMyd03

products.file_cache = products.FileCache("/home/simonpf/cache")

class Collocation:

    def __init__(self, caliop_file):
        self.caliop_file = caliop_file

        times   = self.caliop_file.get_profile_times()

        self.t0 = caliop_tai_to_datetime(times.ravel()[0]) - timedelta(minutes = 5)
        self.t1 = caliop_tai_to_datetime(times.ravel()[-1])

        self.lats = caliop_file.get_latitudes()[:, 0]
        self.lons = caliop_file.get_longitudes()[:, 0]

        self.modis_files     = ModisMyd021km.get_files_in_range(self.t0, self.t1)
        self.modis_geo_files = ModisMyd03.get_files_in_range(self.t0, self.t1)
        self.file_cache = None

    def get_collocation(self, profile_index, d_max = 1.0, use_cache = True):

        lat = self.lats[profile_index]
        lon = self.lons[profile_index]

        i, j, k = 0, 0, 0
        d = np.finfo("d").max

        if use_cache and self.file_cache:
            modis_geo_file = self.modis_geo_files[self.file_cache]
            j_t, k_t, d_t = modis_geo_file.get_collocation(lat, lon, d_max, use_cache)
            if d_t < d_max:
                return self.file_cache, j_t, k_t, d_t

        for file_index, modis_geo_file in enumerate(self.modis_geo_files):
            j_t, k_t, d_t = modis_geo_file.get_collocation(lat, lon, d_max, use_cache)
            if d_t < d:
                i, j, k, d = file_index, j_t, k_t, d_t
                if d < d_max:
                    self.file_cache = file_index
                    break

        return i, j, k, d
