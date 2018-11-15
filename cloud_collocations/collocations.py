"""
The :code:`collocations` module contains high-level abstractions for the
extraction of A-train collocations.
"""
import numpy as np
import scipy as sp
import scipy.signal
import os
import shutil
import tempfile

from datetime import datetime, timedelta
from netCDF4  import Dataset

from cloud_collocations.utils   import caliop_tai_to_datetime
from cloud_collocations         import products
from cloud_collocations.formats import Caliop01kmclay, ModisMyd021km, ModisMyd03


################################################################################
# ModisOutputFile
################################################################################

class ModisOutputFile:
    """
    The Modis input data is stored in NetCDF format. The file contains
    one variable for each of the 38 channels. Each variable has three
    dimensions: The first dimensions contains the different collocations.
    Dimensions two and three contain the along-track and across track
    pixels, respectively.
    """
    def __init__(self, filename, dn, overwrite = False):
        """
        Open or create a new Modis output file.

        Arguments:

            filename(str): Name of the file

            n(int): Extent of the input region.

            overwrite(bool): Whether or not to overwrite any
                existing files.
        """
        self.dn = dn
        path = os.path.dirname(filename)
        if not os.path.exists(path):
            os.makedirs(path)

        # Create new file.
        if not os.path.isfile(filename) or overwrite:
            root = Dataset(filename, "w")
            self.root = root
            self.dims = (root.createDimension("collocation", None),
                         root.createDimension("along_track", 2 * self.dn + 1),
                         root.createDimension("across_track", 2 * self.dn + 1))

            self.bands = []
            for i in range(38):
                dims = ("collocation", "along_track", "across_track")
                self.bands += [root.createVariable("band_{0}".format(i + 1),
                                                   "f4",
                                                   dims)]
            self.lats = root.createVariable("lats", "f4", dims)
            self.lons = root.createVariable("lons", "f4", dims)
            self.collocation_index = 0

        # Read existing file.
        else:
            root = Dataset(filename, mode = "a")
            self.root = root
            self.dims = root.dimensions
            self.bands = []
            for i in range(38):
                self.bands += [root.variables["band_{0}".format(i + 1)]]
            self.lats = root.variables["lats"]
            self.lons = root.variables["lons"]
            self.collocation_index = self.bands[0].shape[0]

    def __del__(self):
        if hasattr(self, "root"):
            self.root.close()

    def add_collocation(self, i, j, modis_file, modis_geo_file):
        data = modis_file.data
        _, m, n = data.shape

        i_start = i - self.dn
        i_end   = i + self.dn + 1
        j_start = j - self.dn
        j_end   = j + self.dn + 1

        if i_start < 0 or i_end >= m or j_start < 0 or j_end >= n:
            print("Collocation out of bounds: [{0} : {1}, {2}, {3}]"
                  .format(i_start, i_end, j_start, j_end))
            raise Exception("Collocation out of bounds of MODIS file.")

        for i in range(38):
            if i_start >= 0 and i_end < m \
               and j_start >= 0 and j_end < n:
                self.bands[i][self.collocation_index, :, :] = \
                    data[i, i_start : i_end, j_start : j_end]
            else:
                self.bands[i][self.collocation_index, :, :] = np.nan

        self.lats[self.collocation_index, :, :] = \
                        modis_geo_file.lats[i_start : i_end, j_start : j_end]
        self.lons[self.collocation_index, :, :] = \
                        modis_geo_file.lons[i_start : i_end, j_start : j_end]

        self.collocation_index += 1

class ModisSubsampledOutputFile(ModisOutputFile):
    """
    Outputfile that stores subsampled modis data.
    """
    def __init__(self, filename, dn, sampling_factor, overwrite = False):
        super().__init__(filename, dn, overwrite = overwrite)
        self.sf = sampling_factor

    def block_average(self, data):
        k = np.ones((self.sf, self.sf)) / self.sf ** 2
        data = sp.signal.convolve2d(data, k, mode = "valid")[::self.sf, ::self.sf]
        return data

    def add_collocation(self, i, j, modis_file, modis_geo_file):
        data = modis_file.data
        _, m, n = data.shape


        i_start = i - (self.dn + 0) * self.sf
        i_end   = i + (self.dn + 1) * self.sf
        j_start = j - (self.dn + 0) * self.sf
        j_end   = j + (self.dn + 1) * self.sf

        if i_start < 0 or i_end >= m or j_start < 0 or j_end >= n:
            print("Subsampled collocation out of bounds: [{0} : {1}, {2}, {3}]"
                  .format(i_start, i_end, j_start, j_end))
            raise Exception("Collocation out of bounds of MODIS file.")

        for i in range(38):
            if i_start >= 0 and i_end < m \
               and j_start >= 0 and j_end < n:
                data_ss = self.block_average(data[i, i_start : i_end, j_start : j_end])
                self.bands[i][self.collocation_index, :, :] = data_ss
            else:
                self.bands[i][self.collocation_index, :, :] = np.nan

        self.lats[self.collocation_index, :, :] = \
                        modis_geo_file.lats[i_start : i_end : self.sf,
                                            j_start : j_end : self.sf]
        self.lons[self.collocation_index, :, :] = \
                        modis_geo_file.lons[i_start : i_end : self.sf,
                                            j_start : j_end : self.sf]

        self.collocation_index += 1

class CaliopOutputFile:
    """
    Class that handles the Caliop output data.
    """

    def __init__(self, filename, dn, overwrite = False):
        """
        Create caliop output file at given location for given input width.

        Parameters:

            filename(str): Path to the file to create or to append to.

            n(int): Half-width of the input field.

            overwrite(bool): Whether or not existings files should be overwritten
                or appended to.
        """
        self.dn = dn
        path = os.path.dirname(filename)
        if not os.path.exists(path):
            os.makedirs(path)

        # Create new file.
        if not os.path.isfile(filename) or overwrite:
            root = Dataset(filename, "w")
            self.root = root
            self.dims = (root.createDimension("collocation", None),
                         root.createDimension("along_track", 2 * self.dn + 1))

            dims = ("collocation", "along_track")

            self.cloud_top_height   = root.createVariable("cth",
                                                        "f4", dims)
            self.cloud_top_pressure = root.createVariable("ctp",
                                                        "f4", dims)
            self.feature_class      = root.createVariable("feature_class",
                                                        "i4", dims)
            self.feature_class_quality = root.createVariable("feature_class_quality",
                                                        "i4", dims)
            self.cloud_class        = root.createVariable("cloud_class",
                                                        "i4", dims)
            self.lats               = root.createVariable("lats",
                                                        "f4", dims)
            self.lons               = root.createVariable("lons",
                                                        "f4", dims)


            self.collocation_index = 0

        # Read existing file.
        else:
            root = Dataset(filename, mode = "a")
            self.root = root
            self.dims = root.dimensions
            self.cloud_top_height   = root.variables["cloud_top_height"]
            self.cloud_top_pressure = root.variables["cloud_top_pressure"]
            self.feature_class  = root.variables["feature_class"]
            self.cloud_class    = root.variables["cloud_class"]
            self.lats = root.variables["lats"]
            self.lons = root.variables["lons"]
            self.collocation_index = self.lons.shape[0]

    def __del__(self):
        if hasattr(self, "root"):
            self.root.close()

    def add_collocation(self, i, caliop_file):
        """
        Add a collocation to the output file.

        """
        self.cloud_top_height[self.collocation_index, :] = \
                    caliop_file.get_cloud_top_height(i, self.dn)
        self.cloud_top_pressure[self.collocation_index, :] = \
                    caliop_file.get_cloud_top_pressure(i, self.dn)
        self.feature_class[self.collocation_index, :] = \
                    caliop_file.get_feature_class(i, self.dn)
        self.feature_class_quality[self.collocation_index, :] = \
                    caliop_file.get_feature_class_quality(i, self.dn)
        self.cloud_class[self.collocation_index, :] = \
                    caliop_file.get_cloud_class(i, self.dn)
        self.lats[self.collocation_index, :] = caliop_file.get_latitudes(i, self.dn).ravel()
        self.lons[self.collocation_index, :] = caliop_file.get_longitudes(i, self.dn).ravel()

        self.collocation_index += 1

################################################################################
# MetaOutputFile
################################################################################

class MetaOutputFile:
    """
    Class that handles meta data files.
    """

    def __init__(self, filename, dn, overwrite = False):
        """
        Create meta data file at given location for given input width.

        Parameters:

            filename(str): Path to the file to create or to append to.

            overwrite(bool): Whether or not existings files should be overwritten
                or appended to.
        """
        self.dn = dn
        path = os.path.dirname(filename)
        if not os.path.exists(path):
            os.makedirs(path)

        # Create new file.
        if not os.path.isfile(filename) or overwrite:
            root = Dataset(filename, "w")
            self.root = root
            self.dims = (root.createDimension("collocation", None),)

            dims = ("collocation",)

            self.time           = root.createVariable("time", "f4", dims)
            self.lat            = root.createVariable("lat", "f4", dims)
            self.lon            = root.createVariable("lon", "f4", dims)
            self.d              = root.createVariable("d", "f4", dims)

            self.collocation_index = 0

        # Read existing file.
        else:
            root = Dataset(filename, mode = "a")
            self.time           = root.variables["time"]
            self.lat            = root.variables["lat"]
            self.lon            = root.variables["lon"]
            self.d              = root.variables["d"]

    def __del__(self):
        if hasattr(self, "root"):
            self.root.close()

    def add_collocation(self, i, caliop_file, d):
        """
        Add a collocation to the output file.



        """
        self.time[self.collocation_index] = caliop_file.get_utc_time(i)
        self.lat[self.collocation_index]  = caliop_file.get_latitudes(i)
        self.lon[self.collocation_index]  = caliop_file.get_longitudes(i)
        self.d[self.collocation_index]    = d

        self.collocation_index += 1

################################################################################
# Collocation
################################################################################

class Collocation:
    """
    The collocation class matches a given Caliop file with Modis observation
    files and provides functions to extract the collocation data from the
    files.

    General workflow to extract collocations: ::

        colls = Collocation(dn, cf, result_path)
        colls.create_output_files(True)

        # Get collocation around the ith caliop profile:
        j, k, l, d = colls.get_collocation(i)

        # If distance between found collocation and caliop
        # profile is sufficiently small, add collocations
        # to output file.

        if d < 1.0:
            colls.add_collocation(i, j, k, l, d)
    """

    def __init__(self, dn, caliop_file, result_path):
        """
        Create collocation from a given caliop file.

        Arguments:

            dn(int): Half-width of the input field.

            caliop_file(Caliop01kmclay): Caliop file for which to extract
                collocations.

            results_path(str): Path to where to store extracted collocations.
        """
        self.dn          = dn
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
        """
        Create output files to store extracted collocation results.

        Arguments:

            overwrite(bool): Whether or to overwrite (True) or append to
                (False) existing files.

        """
        path = self.result_path
        filename = os.path.join(path, "modis_{0}.nc".format(self.dn))
        self.modis_output = ModisOutputFile(filename, self.dn, overwrite)

        filename = os.path.join(path, "modis_ss_5_{0}.nc".format(self.dn))
        self.modis_ss_5_output = ModisSubsampledOutputFile(filename, self.dn, 5, overwrite)

        filename = os.path.join(path, "modis_ss_11_{0}.nc".format(self.dn))
        self.modis_ss_11_output = ModisSubsampledOutputFile(filename, self.dn, 11, overwrite)

        filename = os.path.join(path, "caliop_{0}.nc".format(self.dn))
        self.caliop_output = CaliopOutputFile(filename, self.dn, overwrite)

        filename = os.path.join(path, "caliop_ss_5_{0}.nc".format(self.dn))
        self.caliop_ss_5_output = CaliopOutputFile(filename, self.dn * 5, overwrite)

        filename = os.path.join(path, "caliop_ss_11_{0}.nc".format(self.dn))
        self.caliop_ss_11_output = CaliopOutputFile(filename, self.dn * 11, overwrite)

        filename = os.path.join(path, "meta_{0}.nc".format(self.dn))
        self.meta_output = MetaOutputFile(filename, self.dn, overwrite)

    def get_collocation(self, profile_index, d_max = 1.0, use_cache = True):
        """
        Get collocation centered around a given Caliop profile.

        Arguments:

            profile_index(int): Index of the center profile in the Caliop1kmClay
                file.

            d_max(float): Maximum distance for a found collocation to be valid.

            use_cache(bool): Whether or not to use the cache for faster
            collocation lookup.

        Returns:

            Tuple :code:`(i, j, k, d)` containing:
                - the index :code:`i` of the Myd021km file which
                  contains the collocations.
                - the along-track index :code:`j` of the center pixel of the
                  collocation.
                - the across-track index :code:`k` of the center pixel of the
                  collocation.
                - the distance :code:`d` of the Caliop center profile and
                  the Modis center pixel in kilometer.

        """
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

    def add_collocation(self, profile_index, modis_file_index, modis_i, modis_j, d):
        """
        Add collocation to output files.

        Arguments:

            profile_index(int): The Caliop profile index of the collocation
                center.

            modis_file_index(int): The index of the Modis Myd021km file that
                contains the collocation.

            modis_i(int): Modis along-track index of the collocation center.

            modis_j(int): Modis across-track index of the collocation center.

            d(float): Distance of the Caliop center profile and the Modis
                center pixel in kilometer.

        """

        # MODIS output.
        modis_file = self.modis_files[modis_file_index]
        modis_geo_file = self.modis_geo_files[modis_file_index]

        #try:
        #    self.modis_output.add_collocation(modis_i, modis_j, modis_file, modis_geo_file)
        #    self.modis_ss_5_output.add_collocation(modis_i, modis_j, modis_file, modis_geo_file)
        #    self.modis_ss_11_output.add_collocation(modis_i, modis_j, modis_file, modis_geo_file)
        #except:
        #    return None
        try:
            self.modis_ss_11_output.add_collocation(modis_i, modis_j, modis_file, modis_geo_file)
            self.modis_ss_5_output.add_collocation(modis_i, modis_j, modis_file, modis_geo_file)
            self.modis_output.add_collocation(modis_i, modis_j, modis_file, modis_geo_file)
        except:
            return None

        # Caliop output.
        self.caliop_output.add_collocation(profile_index, self.caliop_file)
        self.caliop_ss_5_output.add_collocation(profile_index, self.caliop_file)
        self.caliop_ss_11_output.add_collocation(profile_index, self.caliop_file)

        # Meta data.
        self.meta_output.add_collocation(profile_index, self.caliop_file, d)


################################################################################
# ProcessDay
################################################################################

class ProcessDay:
    """
    Extract collocation for a given day.

    This class encapsules the tasks necessary to process a day of collocations.
    Files are saved in a directory tree with its base at the given :code:`path`.
    Two intermediate folder levels are inserted for :code:`year` and
    :code:`day`:. ::

        +-- path
        |   +-- year
        |   |    +-- day
        |   |    |   +-- modis_<dn>.nc
        |   |    |   +-- caliop_<dn>.nc
        |   |    |   +-- meta_<dn>.nc
        |   |    |   +-- cache
        |   |    |   |    +-- Modis + Caliop files

    """
    def __init__(self,
                 year,
                 day,
                 path,
                 dn = 20,
                 cache = None):

        self.day  = day
        self.year = year
        self.dn   = dn

        # Create output tree.
        if cache is None:
            self.cache = tempfile.mkdtemp()
        else:
            self.cache = cache

        day_str = str(self.day)
        day_str = "0" * (3 - len(day_str)) + day_str
        self.result_path = os.path.join(path, str(year), day_str)
        #self.cache = os.path.join(self.result_path, "cache")

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        if not os.path.exists(self.cache):
            os.makedirs(self.cache)

        # Set file cache.
        products.file_cache = products.FileCache(self.cache)
        if cache is None:
            products.file_cache.temp = True


    def run(self):
        """
        Start processing the day.

        This will start downloading the files and extract the collocations
        for the given day.
        """
        day_str = str(self.day)
        day_str = "0" * (3 - len(day_str)) + day_str

        t0 = datetime.strptime(str(self.year) + "_" + day_str + "_000000",
                               "%Y_%j_%H%M%S")

        day_str = str(self.day + 1)
        day_str = "0" * (3 - len(day_str)) + day_str
        t1 = datetime.strptime(str(self.year) + "_" + day_str + "_000000",
                               "%Y_%j_%H%M%S")
        caliop_files = Caliop01kmclay.get_files_in_range(t0, t1)

        for cf in caliop_files:
            colls = Collocation(self.dn, cf, self.result_path)
            colls.create_output_files(True)
            lats_caliop = cf.get_latitudes()

            i = self.dn * 11
            while i < lats_caliop.size - self.dn * 11:
                j, k, l, d = colls.get_collocation(i)

                if d < 1.0:
                    colls.add_collocation(i, j, k, l, d)
                i += 2 * self.dn

            print("Finished processing " + cf.filename)
