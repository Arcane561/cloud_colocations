import numpy as np
import os

from datetime import datetime, timedelta

from netCDF4 import Dataset

from cloud_collocations.utils import caliop_tai_to_datetime
from cloud_collocations import products
from cloud_collocations.formats import Caliop01kmclay, ModisMyd021km, ModisMyd03

products.file_cache = products.FileCache("/home/simonpf/cache")

class ModisOutputFile:

    def __init__(self, filename, n, overwrite = False):
        self.n = n
        path = os.path.dirname(filename)
        if not os.path.exists(path):
            os.makedirs(path)

        # Create new file.
        if not os.path.isfile(filename) or overwrite:
            root = Dataset(filename, "w")
            self.root = root
            self.dims = (root.createDimension("collocation", None),
                         root.createDimension("along_track", 2 * self.n + 1),
                         root.createDimension("across_track", 2 * self.n + 1))

            self.bands = []
            for i in range(38):
                print(i)
                dims = ("collocation", "along_track", "across_track")
                self.bands += [root.createVariable("band_{0}".format(i + 1),
                                                   "f4",
                                                   dims)]
            self.collocation_index = 0

        # Read existing file.
        else:
            root = Dataset(filename, mode = "a")
            self.root = root
            self.dims = root.dimensions
            self.bands = []
            for i in range(38):
                self.bands += [root.variables["band_{0}".format(i + 1)]]
            self.collocation_index = self.bands[0].shape[0]

    def __del__(self):
        self.root.close()

    def add_collocation(self, i, j, data):
        m, n, _ = data.shape

        for i in range(38):
            i_start = i - self.n
            i_end   = i + self.n + 1
            j_start = j + self.n
            j_end   = j + self.n + 1
            if i_start >= 0 and i_end < m \
               and j_start >= 0 and j_end < n:
                self.bands[i][self.collocation_index, :, :] = \
                    data[i_start, i_end, j_start, j_end]
            else:
                self.bands[i][self.collocation_index, :, :] = np.nan
        self.collocation_index += 1


class Collocation:

    def __init__(self, n, caliop_file, result_path):

        self.n           = n
        self.caliop_file = caliop_file
        self.result_path = result_path

        times   = self.caliop_file.get_profile_times()

        self.t0 = caliop_tai_to_datetime(times.ravel()[0]) - timedelta(minutes = 5)
        self.t1 = caliop_tai_to_datetime(times.ravel()[-1])

        self.lats = caliop_file.get_latitudes()[:, 0]
        self.lons = caliop_file.get_longitudes()[:, 0]

        self.modis_files     = ModisMyd021km.get_files_in_range(self.t0, self.t1)
        self.modis_geo_files = ModisMyd03.get_files_in_range(self.t0, self.t1)
        self.file_cache = None

    def create_output_files(self, overwrite = False):

        date = products.caliop.name_to_date(self.caliop_file.filename)
        path = os.path.join(self.result_path,
                            str(date.year),
                            date.strftime("%j"))
        filename = os.path.join(path, "modis_{0}.nc".format(self.n))
        self.modis_output = ModisOutputFile(filename, self.n, overwrite)



    def get_collocation(self, profile_index, d_max = 1.0, use_cache = True):

        lat = self.lats[profile_index]
        lon = self.lons[profile_index]

        i, j, k = 0, 0, 0
        d = np.finfo("d").max

        if use_cache and self.file_cache:
            modis_geo_file = self.modis_geo_files[self.file_cache]
            j_t, k_t, d_t = modis_geo_file.get_collocation(lat, lon, d_max,
                                                           use_cache)
            if d_t < d_max:
                return self.file_cache, j_t, k_t, d_t

        for file_index, modis_geo_file in enumerate(self.modis_geo_files):
            j_t, k_t, d_t = modis_geo_file.get_collocation(lat, lon, d_max,
                                                           use_cache)
            if d_t < d:
                i, j, k, d = file_index, j_t, k_t, d_t
                if d < d_max:
                      self.file_cache = file_index
                      break

        return i, j, k, d

    def add_collocation(self, profile_index, modis_file_index, modis_i, modis_j):

        # MODIS output.
        mf = self.modis_files[modis_file_index]
        self.modis_output.add_collocation(modis_i, modis_j, mf.data)



