"""
The :code:`colocations` module contains high-level abstractions for the
extraction of A-train colocations.
"""
import numpy as np
import scipy as sp
import scipy.signal
import os
import shutil
import tempfile

from datetime import datetime, timedelta
from netCDF4  import Dataset

from cloud_colocations.colocations         import products, utils
from cloud_colocations.colocations.formats import Caliop01kmclay, ModisMyd021km, ModisMyd03

################################################################################
# OutputFile
################################################################################

class OutputFile:
    """
    Generic output file the stores extracted collocations for given source and
    matched products.

    The colocation output file contains three groups:
       - 1 for the data from the source product
       - 1 for the data from the matched product
       - 1 with meta data on the colocations.
    """
    def __init__(self,
                 name,
                 path,
                 source_product,
                 matched_product, mode = "w"):

        self.source_product  = source_product
        self.matched_product = matched_product
        self.fh = Dataset(os.path.join(path, name + ".nc"), mode = mode)

        self.fh.createDimension("colocation_index", size = None)

        #
        # Source product
        #

        self.sp_group = self.fh.createGroup(source_product.name)
        self.sp_dims  = []
        for (n, s) in self.source_product.dimensions:
            self.sp_group.createDimension(n, size = s)
            self.sp_dims += [n]

        self.sp_vars = {}
        for (n, t, ds) in self.source_product.variables:
            dims = ("colocation_index",) + ds
            self.sp_vars[n] = self.sp_group.createVariable(n, t, dimensions = dims)
        #
        # Matched product
        #

        if not self.matched_product is None:
            self.mp_group = self.fh.createGroup(matched_product.name)
            self.mp_dims = []
            for (n, s) in self.matched_product.dimensions:
                self.mp_group.createDimension(n, size = s)
                self.mp_dims += [n]

            self.mp_vars = {}
            for (n, t, ds) in self.matched_product.variables:
                dims = ("colocation_index",) + ds
                self.mp_vars[n] = self.mp_group.createVariable(n, t, dimensions = dims)

        self.ci = 0

    def close(self):
        self.fh.close()

    def _get_variable(self, product, variable_name, indices):
        name = "get_{0}".format(variable_name)
        try:
            f = getattr(product, name)

            if hasattr(product, "get_kwargs"):
                kwargs = product.get_kwargs
            else:
                kwargs = {}

            data = f(*indices, **kwargs)

        except Exception as e:
            raise Exception("The following error occurred when trying to"
                            "retrieve variable {0} from product {1}:"
                            " {2}".format(variable_name, product.name, e))
        return data

    def add_colocation(self, source_file, source_inds, matched_file, matched_inds):

        for (n, _, _) in self.source_product.variables:
            data = self._get_variable(source_file, n, source_inds)
            self.sp_vars[n][self.ci, :] = data

        if not self.matched_product is None:
            for (n, _, _) in self.matched_product.variables:
                data = self._get_variable(matched_file, n, matched_inds)
                self.mp_vars[n][self.ci, :] = data

        self.ci += 1

################################################################################
# ProcessDay
################################################################################

class Colocations:
    """
    The :class:`Colocations` class processes colocations for a julian day of
    a given year.
    """
    def __init__(self,
                 year,
                 day,
                 path,
                 cache = None):
        """
        Arguments:

            year(int): The year from which to process colocations.

            day(int): Julian day of :code:`year` from which to process
                colocations.

            path(str): Path in which to store extracted colocations.

            cache(str): Folder to use for caching. If None a temporary
                folder will be created.
        """
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


class ProcessDay:
    """
    Extract colocation for a given day.

    This class encapsules the tasks necessary to process a day of colocations.
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
                 name,
                 year,
                 day,
                 path,
                 source_product,
                 matched_product,
                 cache = None):

        self.day  = day
        self.year = year

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

        self.source_product  = source_product
        self.matched_product = matched_product

        self.output_file = OutputFile(name, self.result_path,
                                      source_product, matched_product)
        self.file_cache = None

    def _get_source_files(self):
        day_str = str(self.day)
        day_str = "0" * (3 - len(day_str)) + day_str

        t0 = datetime.strptime(str(self.year) + "_" + day_str + "_000000",
                               "%Y_%j_%H%M%S")

        day_str = str(self.day + 1)
        day_str = "0" * (3 - len(day_str)) + day_str
        t1 = datetime.strptime(str(self.year) + "_" + day_str + "_000000",
                               "%Y_%j_%H%M%S")

        try:
            fs = self.source_product.get_files_in_range(t0, t1)
        except Exception as e:
            raise Exception("The following error was encountered when "
                            "processing collocations for day {0} from "
                            "year {1}: {2}".format(self.day,
                                                   self.year,
                                                   e))
        return fs

    def _get_matched_files(self, source_file):

        if self.matched_product is None:
            return [source_file]

        t0 = source_file.get_start_time()
        t1 = source_file.get_end_time()

        fs = self.matched_product.get_files_in_range(t0, t1)
        print("matched files: ", fs)
        return fs

    def get_colocation(self, lat, lon, matched_files, d_max = 1.0,
                       use_cache = True):
        """
        Get colocation centered around a given Caliop profile.

        Arguments:

            profile_index(int): Index of the center profile in the Caliop1kmClay
                file.

            d_max(float): Maximum distance for a found colocation to be valid.

            use_cache(bool): Whether or not to use the cache for faster
            colocation lookup.

        Returns:

            Tuple :code:`(i, j, k, d)` containing:
                - the index :code:`i` of the Myd021km file which
                  contains the colocations.
                - the along-track index :code:`j` of the center pixel of the
                  colocation.
                - the across-track index :code:`k` of the center pixel of the
                  colocation.
                - the distance :code:`d` of the Caliop center profile and
                  the Modis center pixel in kilometer.

        """
        d = 1e100

        if use_cache and self.file_cache:
            f = self.file_cache
            inds, d_t = f.get_colocation(lat, lon, d_max, use_cache)
            if d_t < d_max:
                return f, inds, d_t

        inds_min = None
        f_min = None

        for f in matched_files:
            inds, d_t = f.get_colocation(lat, lon, d_max, use_cache)
            if d_t < d:
                inds_min = inds
                f_min = f
                d = d_t

        if d < d_max:
            self.file_cache = f_min

        return f_min, inds_min, d

    def run(self):
        """
        Start processing the day.

        This will start downloading the files and extract the colocations
        for the given day.
        """
        source_files = self._get_source_files()
        for source_file in source_files:

            matched_files = self._get_matched_files(source_file)

            if hasattr(self.source_product, "get_kwargs"):
                kwargs = self.source_product.get_kwargs
            else:
                kwargs = {}

            for source_inds in source_file.get_colocation_centers(**kwargs):
                lat = source_file.get_latitude(*source_inds)
                lon = source_file.get_longitude(*source_inds)
                matched_file, matched_inds, d = self.get_colocation(lat, lon, matched_files)
                if d < 1.0:
                    print("Found colocation: ", d)
                    self.output_file.add_colocation(source_file, source_inds,
                                                    matched_file, matched_inds)
        self.output_file.close()
