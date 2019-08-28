"""
The :code:`formats` module provides classes to simplify the handling
of the file formats of the different data products.
"""

import os
import numpy as np
import scipy as sp

from h5py import File

from cloud_colocations.colocations import utils
from cloud_colocations.colocations.products import file_cache, caliop, modis,\
    modis_geo, cloudsat, gpm_2b_cmb, gpm_2a_gprofgmi, get_cache_path, gpm_1c_r

from datetime import datetime

################################################################################
# ProductFile
################################################################################

class ProductFile:
    """
    Base class for files from the Icare data center. Provides abstract class
    methods to obtain files for a given point in time or time range.
    """
    @classmethod
    def get_by_date(cls, t):
        """
        Return file with the latest start point earlier than the given
        datetime :code:`t`.

        Parameters:

            t(datetime.datetime): :code:`datetime` object representing the point
            in time for which to find the icare file.

        Returns:

            The file object with the latest found start time berfore the given
            time.
        """
        filename = cls.product.get_file_by_date(t)
        path     = cls.product.download_file(filename)
        return cls(path)

    @classmethod
    def get_files_in_range(cls, t0, t1, t0_inclusive = False):
        """
        Get files within time range.

        Parameters:

            t0(datetime.datetime): :code:`date`

        """
        filenames = cls.product.get_files_in_range(t0, t1, t0_inclusive)
        objs = []
        for f in filenames:
            path = cls.product.download_file(f)
            objs += [cls(path)]
        return objs

################################################################################
# Combined
################################################################################

class Combined:
    """
    Base class for products that need to be combined for processing such as for
    example the MODIS files.
    """
    @classmethod
    def get_by_date(cls, t):
        paths = []
        for p in cls.products:
            filename = p.get_file_by_date(t)
            paths += [p.download_file(filename)]
        return cls(*paths)

    @classmethod
    def get_files_in_range(cls, t0, t1):
        """
        Get files within time range.

        Parameters:

            t0(datetime.datetime): :code:`date`

        """
        paths = []
        for p in cls.products:
            filenames = p.get_files_in_range(t0, t1)
            paths += [[]]
            for f in filenames:
                paths[-1] += [p.download_file(f)]
            print(paths)
        return [cls(*ps) for ps in zip(*paths)]

################################################################################
# Hdf4File
################################################################################

class Hdf4File:
    """
    Base class for file formats using HDF4File format. The :class:`Hdf4File`
    wraps around the pyhdf.SD class to implement RAII.
    """
    def __init__(self, filename):
        """
        Open an HDF4 file for reading.

        Arguments:

            filename(str): The path to the file to open.
        """
        from pyhdf.SD import SD, SDC
        self.filename = filename
        self.file_handle = SD(self.filename, SDC.READ)

    def __del__(self):
        self.file_handle.end()

################################################################################
# Caliop01kmclay
################################################################################
def caliop_utc_to_string(t):
    f = t - np.trunc(t)
    h = np.trunc(f * 24.0)
    m = np.trunc((f * 24.0 - h) * 60)
    s = np.trunc(((f * 24.0 - h) * 60.0 - m) * 60.0)
    return "{0:06.0f}{1:02.0f}{2:02.0f}{3:02.0f}".format(t, h, m, s)

class Caliop01kmclay(Hdf4File, ProductFile):

    product = caliop

    """
    The CALIOP 1 km cloud layer data format.

    This class provide a high-level interface that wraps around the HDF
    file and provides simplified access to the data that is extracted
    for the colocations.
    """
    def __init__(self, filename):
        """
        Create :code:`Caliop01kmclay` object from file.

        Arguments:

            filename(str): Path to the file to read.

        """
        super().__init__(filename)

        self.profile_times = self.file_handle.select('Profile_Time')[:].ravel()


    def get_latitude(self, c_i = -1, dn = 0):
        """
        Get latitudes of profile in file as :code:`numpy.ndarray`.
        """
        if c_i < 0:
            return self.file_handle.select('Latitude')[:, 0]
        elif dn == 0:
            return self.file_handle.select('Latitude')[c_i, 0]
        else:
            return self.file_handle.select('Latitude')[c_i - dn : c_i + dn + 1, 0]

    def get_longitude(self, c_i = -1, dn = 0):
        """
        Get longitudes of profile in file as :code:`numpy.ndarray`.
        """
        if c_i < 0:
            return self.file_handle.select('Longitude')[:, 0]
        elif dn == 0:
            return self.file_handle.select('Longitude')[c_i, 0]
        else:
            return self.file_handle.select('Longitude')[c_i - dn : c_i + dn + 1, 0]

    def get_cloud_top_height(self, c_i = -1, dn = 0):
        """
        Get altitude of topmost layer as :code:`numpy.ndarray`. By default
        this function will return the top altitudes for the first detected
        layer for all profiles. However, if c_i and dn are give, it will
        return only the profiles in the region containing the :code:`2 * dn + 1`
        profiles centered around :code:`c_i`.

        Arguments:

            c_i(int): Profile index of colocation center.

            dn(int): Half extent of the region to extract.

        """
        if c_i < 0:
            return self.file_handle.select('Layer_Top_Altitude')[:, 0]
        else:
            return self.file_handle.select('Layer_Top_Altitude')\
                [c_i - dn : c_i + dn + 1, 0]

    def get_feature_class(self, c_i = -1, dn = 0):
        """
        Return classification flag for uppermost detected layer.

        Arguments:

            c_i(int): Profile index of colocation center.

            dn(int): Half extent of the region to extract.
        """
        if c_i < 0:
            classes = self.file_handle.select('Feature_Classification_Flags')[:, 0] % 8
        else:
            classes = self.file_handle.select('Feature_Classification_Flags')\
                [c_i - dn : c_i + dn + 1, 0] % 8
        return classes

    def get_feature_class_curtain(self, c_i = -1, dn = 0):
        if c_i < 0:
            classes = self.file_handle.select('Feature_Classification_Flags')[:, :] % 8
        else:
            classes = self.file_handle.select('Feature_Classification_Flags')\
                [c_i - dn : c_i + dn + 1, :] % 8
        return classes

    def get_cloud_class(self, c_i = -1, dn = 0):
        """
        Return classification flag for uppermost detected layer.

        Arguments:

            c_i(int): Profile index of colocation center.

            dn(int): Half extent of the region to extract.
        """
        if c_i < 0:
            classes = self.file_handle.select('Feature_Classification_Flags')[:, 0]
        else:
            classes = self.file_handle.select('Feature_Classification_Flags') \
                [c_i - dn : c_i + dn + 1, 0]

        cloud_types = np.zeros(classes.shape)
        inds = (classes % 8) == 2
        cloud_types[inds] = (classes[inds] // 512) % 8
        return cloud_types

    def get_feature_class_quality(self, c_i = -1, dn = 0):
        """
        Return classification flag for uppermost detected layer.

        Arguments:

            c_i(int): Profile index of colocation center.

            dn(int): Half extent of the region to extract.
        """
        if c_i < 0:
            classes = self.file_handle.select('Feature_Classification_Flags')[:, 0]
        else:
            classes = self.file_handle.select('Feature_Classification_Flags') \
                [c_i - dn : c_i + dn + 1, 0]
        return (classes // 8) % 4

    def get_cloud_top_pressure(self, c_i = -1, dn = 0):
        """
        Get pressure of topmost layer as :code:`numpy.ndarray`. By default
        this function will return the top pressure for the first detected
        layer for all profiles. However, if c_i and dn are give, it will
        return only the profiles in the region containing the :code:`2 * dn + 1`
        profiles centered around :code:`c_i`.

        Arguments:

            c_i(int): Profile index of colocation center.

            dn(int): Half extent of the region to extract.

        """
        if c_i < 0:
            return self.file_handle.select('Layer_Top_Pressure')[:, 0]
        else:
            return self.file_handle.select('Layer_Top_Pressure')[c_i - dn : c_i + dn + 1, 0]


    def get_utc_time(self, c_i = -1, dn = 0):
        """
        Returns the UTC times for all profiles in the file as numpy array.
        """
        if c_i < 0:
            return self.file_handle.select('Profile_UTC_Time')[:]
        else:
            return self.file_handle.select('Profile_UTC_Time')[c_i - dn : c_i + dn + 1]

    def get_start_time(self):
        date_str = caliop_utc_to_string(self.get_utc_time()[0, 0])
        return datetime.strptime(date_str, "%y%m%d%H%M%S")

    def get_end_time(self):
        date_str = caliop_utc_to_string(self.get_utc_time()[-1, 0])
        return datetime.strptime(date_str, "%y%m%d%H%M%S")

    def get_profile_times(self, c_i = -1, dn = 0):
        """
        Returns the profile times for all profiles in the file as numpy array.
        """
        if c_i < 0 or dn < 0:
            return self.file_handle.select('Profile_Time')[:]
        else:
            return self.file_handle.select('Profile_Time')[c_i - dn : c_i + dn + 1]

    def get_profile_id(self, c_i, dn):
        return self.file_handle.select('Profile_ID')[c_i - dn : c_i + dn + 1]

    def get_colocation_centers(self, dn = 100):
        n  = self.file_handle.select("Latitude")[:].shape[0]
        for i in range(dn, n - dn - 1, dn):
            yield (i,)

################################################################################
# MODIS MYD03
################################################################################

class ModisMyd03(Hdf4File, ProductFile):
    """
    The MODIS Aqua geolocation file format containing geolocation data
    corresponding to the L1B radiances.
    """

    product = modis_geo

    def __init__(self, filename):
        super().__init__(filename)

        self.lats = self.file_handle.select('Latitude')[:, :]
        self.lons = self.file_handle.select('Longitude')[:, :]

        self.i_cached = None
        self.j_cached = None
        self.cache_range = 10

    def get_latitudes(self):
        return self.lats

    def get_longitudes(self):
        return self.lons

    def _get_colocation(self, lat, lon, lats, lons):
        from geopy.distance import distance

        m, n = lats.shape

        dlats = lats - lat
        dlons = lons - lon

        deglen = np.cos(np.pi * lat / 180.0)
        d = (dlats) ** 2 + (deglen * dlons) ** 2
        ind = np.argmin(d.ravel())

        i = ind // n
        j = ind % n

        return i, j, distance((lat, lon), (lats[i, j], lons[i, j])).km

    def get_colocation(self, lat, lon, d_max = 1.0, use_cache = True):

        if use_cache and self.i_cached and self.j_cached:
            i_start = max(self.i_cached - self.cache_range, 0)
            i_end   = self.i_cached + self.cache_range
            j_start = max(self.j_cached - self.cache_range, 0)
            j_end   = self.j_cached + self.cache_range
            i, j, d = self._get_colocation(lat, lon,
                                            self.lats[i_start : i_end, j_start : j_end],
                                            self.lons[i_start : i_end, j_start : j_end])
            if d < d_max:
                return i, j, d

        i, j, d = self._get_colocation(lat, lon, self.lats, self.lons)

        if d < d_max:
            self.i_cached = i
            self.j_cached = j
            print("found colocation", i, j, d)

        return (i, j), d


class ModisMyd021km(Hdf4File, ProductFile):
    """
    The MODIS Aqua Level1B calibrated radiances at 1 km resolution.
    """

    product = modis

    def __init__(self, filename):
        Hdf4File.__init__(self, filename)
        self._data = None

    def load_data(self):

        raw_data = self.file_handle.select("EV_250_Aggr1km_RefSB")
        shape = raw_data.info()[2]

        # Channels 1 - 2 
        self._data = np.zeros((38, shape[1], shape[2]))
        self.m = shape[1]
        self.n = shape[2]
        self._data[:2, :, :] = raw_data[:, :, :]
        data = self._data[:2, :, :]

        attributes = raw_data.attributes()
        valid_range = attributes["valid_range"]
        valid_min = valid_range[0]
        valid_max = valid_range[1]
        offsets     = np.asarray(attributes["reflectance_offsets"])
        scales      = np.asarray(attributes["reflectance_scales"])
        fill_value  = attributes["_FillValue"]

        invalid = np.logical_or(data > valid_max,
                                data < valid_min)
        invalid = np.logical_or(invalid, data == fill_value)
        data[invalid] = np.nan

        for i in range(2):
            data[i, :, :] = (data[i, :, :] - offsets[i]) * scales[i]

        # Channels 3 - 8 
        raw_data = self.file_handle.select("EV_500_Aggr1km_RefSB")
        self._data[2:7, :, :] = raw_data[:, :, :]
        data = self._data[2:7, :, :]

        attributes = raw_data.attributes()
        valid_range = attributes["valid_range"]
        valid_min = valid_range[0]
        valid_max = valid_range[1]
        offsets     = np.asarray(attributes["reflectance_offsets"])
        scales      = np.asarray(attributes["reflectance_scales"])
        fill_value  = attributes["_FillValue"]

        invalid = np.logical_or(data > valid_max,
                                data < valid_min)
        invalid = np.logical_or(invalid, data == fill_value)
        data[invalid] = np.nan

        for i in range(5):
            data[i, :, :] = (data[i, :, :] - offsets[i]) * scales[i]

        # Channels 8 - 22 
        raw_data = self.file_handle.select("EV_1KM_RefSB")
        self._data[7:21, :, :] = raw_data[:14, :, :]
        data = self._data[7:21, :, :]

        attributes = raw_data.attributes()
        valid_range = attributes["valid_range"]
        valid_min = valid_range[0]
        valid_max = valid_range[1]
        offsets     = np.asarray(attributes["reflectance_offsets"])
        scales      = np.asarray(attributes["reflectance_scales"])
        fill_value  = attributes["_FillValue"]

        invalid = np.logical_or(data > valid_max,
                                data < valid_min)
        invalid = np.logical_or(invalid, data == fill_value)
        data[invalid] = np.nan

        for i in range(14):
            data[i, :, :] = (data[i, :, :] - offsets[i]) * scales[i]

        # Channels 20 - 26 
        raw_data = self.file_handle.select("EV_1KM_Emissive")
        self._data[21:27, :, :] = raw_data[:6, :, :]
        data = self._data[21:27, :, :]

        attributes = raw_data.attributes()
        valid_range = attributes["valid_range"]
        valid_min = valid_range[0]
        valid_max = valid_range[1]
        offsets     = np.asarray(attributes["radiance_offsets"])
        scales      = np.asarray(attributes["radiance_scales"])
        fill_value  = attributes["_FillValue"]

        invalid = np.logical_or(data > valid_max,
                                data < valid_min)
        invalid = np.logical_or(invalid, data == fill_value)
        data[invalid] = np.nan

        for i in range(6):
            data[i, :, :] = (data[i, :, :] - offsets[i]) * scales[i]

        # Channel 26
        raw_data = self.file_handle.select("EV_Band26")
        self._data[27, :, :] = raw_data[:, :]
        data = self._data[27, :, :]

        attributes = raw_data.attributes()
        valid_range = attributes["valid_range"]
        valid_min = valid_range[0]
        valid_max = valid_range[1]
        offsets     = np.asarray(attributes["radiance_offsets"])
        scales      = np.asarray(attributes["radiance_scales"])
        fill_value  = attributes["_FillValue"]

        invalid = np.logical_or(data > valid_max,
                                data < valid_min)
        invalid = np.logical_or(invalid, data == fill_value)
        data[invalid] = np.nan

        data[:, :] = (data[:, :] - offsets) * scales

        # Channels 28 - 38 
        raw_data = self.file_handle.select("EV_1KM_Emissive")
        self._data[28:38, :, :] = raw_data[6:, :, :]
        data = self._data[28:38, :, :]

        attributes = raw_data.attributes()
        valid_range = attributes["valid_range"]
        valid_min = valid_range[0]
        valid_max = valid_range[1]
        offsets     = np.asarray(attributes["radiance_offsets"])
        scales      = np.asarray(attributes["radiance_scales"])
        fill_value  = attributes["_FillValue"]

        invalid = np.logical_or(data > valid_max,
                                data < valid_min)
        invalid = np.logical_or(invalid, data == fill_value)
        data[invalid] = np.nan

        for i in range(9):
            data[i, :, :] = (data[i, :, :] - offsets[6 + i]) * scales[6 + i]

    @property
    def data(self):
        if self._data is None:
            self.load_data()
        return self._data


    def subsample_data(self, band_index, subsampling_factor):
        return utils.block_average(self.data[band_index, :, :], subsampling_factor)

    def get_input_data(self, c_i, c_j, dn):
        bands = [20, 27, 28, 29, 31, 32, 33]
        band_offsets = [20, 21, 21, 21, 21, 21, 21]
        ds_name = "EV_1KM_Emissive"

        raw_data = self.file_handle.select(ds_name)
        data = raw_data

        res = np.zeros((len(bands), 2 * dn + 1, 2 * dn + 1))

        attributes = raw_data.attributes()
        valid_range = attributes["valid_range"]
        offsets = attributes["radiance_offsets"]
        scales  = attributes["radiance_scales"]
        fill_value = attributes["_FillValue"]

        for i, (b, o) in enumerate(zip(bands, band_offsets)):
            valid_min = valid_range[0]
            valid_max = valid_range[1]
            offset = offsets[b - o]
            scale_factor = scales[b - o]

            res[i, :, :] = data[int(b - o),
                                int(c_i - dn) : int(c_i + dn + 1),
                                int(c_j - dn) : int(c_j + dn + 1)]
            invalid = np.logical_or(res[i, :, :] > valid_max,
                                    res[i, :, :] < valid_min)
            invalid = np.logical_or(invalid, res[i, :, :] == fill_value)
            res[i, invalid] = np.nan

            res[i, :, :] = (res[i, :, :] - offset) * scale_factor
        return res


class ModisCombined(Combined, ModisMyd021km):
    """
    This class combined the MODIS Level1B radiances with the geolocation
    information.
    """

    products = [modis, modis_geo]

    def __init__(self, filename, geo_filename, dn = 100):
        ModisMyd021km.__init__(self, filename)
        self._data = None
        self.geo_file = ModisMyd03(geo_filename)
        self.dn = dn

    def get_colocation(self, lat, lon, d_max = 1.0, use_cache = True):
        return self.geo_file.get_colocation(lat, lon, d_max = d_max, use_cache = use_cache)

    def get_latitudes(self, i, j):
        i_start = i - self.dn
        i_end   = i + self.dn + 1
        j_start = j - self.dn
        j_end   = j + self.dn + 1
        return self.geo_file.lats[i_start : i_end, j_start : j_end]

    def get_longitudes(self, i, j):
        i_start = i - self.dn
        i_end   = i + self.dn + 1
        j_start = j - self.dn
        j_end   = j + self.dn + 1
        return self.geo_file.lons[i_start : i_end, j_start : j_end]

    def get_radiances(self, i, j):
        i_start = i - self.dn
        i_end   = i + self.dn + 1
        j_start = j - self.dn
        j_end   = j + self.dn + 1
        print(self.data.shape, i_start, i_end, j_start, j_end)
        return self.data[:, i_start : i_end, j_start : j_end]

################################################################################
# 2B-CLDCLASS-LIDAR    
################################################################################


class TWOBCLDCLASS(Hdf4File, ProductFile):

    product = cloudsat

    """
    The CloudSat cloud layer data format (10 vertical layers in each profile). Cloudlayers are detected based on CPR and CALIOP. 

    This class provide a high-level interface that wraps around the HDF
    file and provides simplified access to the data that is extracted
    for the colocations.
    """
    def __init__(self, filename):
        """
        Create :code:`2B-CLDCLASS-LIDAR` object from file.

        Arguments:

            filename(str): Path to the file to read.

        """
        super().__init__(filename)

        self.profile_times = self.f['Profile_time'].ravel()

    def get_latitudes(self, c_i = -1, dn = 0):
        """
        Get latitudes of profile in file as :code:`numpy.ndarray`.
        """
        if c_i < 0:
            return self.f['Latitude'][:]
        else:
            return self.f['Latitude'][c_i - dn : c_i + dn + 1]

    def get_longitudes(self, c_i = -1, dn = 0):
        """
        Get longitudes of profile in file as :code:`numpy.ndarray`.
        """
        if c_i < 0:
            return self.f['Longitude'][:]
        else:
            return self.f['Longitude'][c_i - dn : c_i + dn + 1]



#################################################### information for upper and lower cloud layer ##############################################


    def get_cldtype_low(self, c_i = -1, dn = 0):
        """
        Get number of all detected cloud layers as :code:`numpy.ndarray` (max= 10).

        Arguments:

            c_i(int): Profile index of colocation center.

            dn(int): Half extent of the region to extract.

        """
        if c_i < 0:
            return self.f['CloudLayerType'][:, 0]
        else:
            return self.f['CloudLayerType']\
                [c_i - dn : c_i + dn + 1, 0]


    def get_cldtype_high(self, c_i = -1, dn = 0):
        """
        Get number of all detected cloud layers as :code:`numpy.ndarray` (max= 10).

        Arguments:

            c_i(int): Profile index of colocation center.

            dn(int): Half extent of the region to extract.

        """
        if c_i < 0:
            return self.f['CloudLayerType'][:, 1]
        else:
            return self.f['CloudLayerType']\
                [c_i - dn : c_i + dn + 1, 0]



    def get_prec_low(self, c_i = -1, dn = 0):
        """
        Get number of all detected cloud layers as :code:`numpy.ndarray` (max= 10).

        Arguments:

            c_i(int): Profile index of colocation center.

            dn(int): Half extent of the region to extract.

        """
        if c_i < 0:
            return self.f['PrecipitationFlag'][:, 0]
        else:
            return self.f['PrecipitationFlag']\
                [c_i - dn : c_i + dn + 1, 0]


    def get_prec_high(self, c_i = -1, dn = 0):
        """
        Get number of all detected cloud layers as :code:`numpy.ndarray` (max= 10).

        Arguments:

            c_i(int): Profile index of colocation center.

            dn(int): Half extent of the region to extract.

        """
        if c_i < 0:
            return self.f['PrecipitationFlag'][:, 1]
        else:
            return self.f['PrecipitationFlag']\
                [c_i - dn : c_i + dn + 1, 0]

    def get_profile_times(self, c_i = -1, dn = 0):
        """
        Returns the profile times for all profiles in the file as numpy array.
        """
        if c_i < 0 or dn < 0:
            return self.f['Profile_time'][:]
        else:
            return self.f['Profile_time'][c_i - dn : c_i + dn + 1]



    def get_elevation(self, c_i = -1, dn = 0):
        """
        Returns the elevation (m ASL) for all profiles in the file as numpy array.
        """
        if c_i < 0 or dn < 0:
            return self.f['DEM_elevation'][:]
        else:
            return self.f['DEM_elevation'][c_i - dn : c_i + dn + 1]


################################################################################
# GPM files
################################################################################

class GPMGMI1C(ProductFile):
    """
    Class representing files of the remapped GPM GMI Level 1CR data product.
    """

    product = gpm_2a_gprofgmi
    dn = 64
    name = "gmi"
    dimensions = [("along_track", dn),
                  ("across_track", dn),
                  ("channels", 13)]
    variables = [("y", "f4", ("along_track", "across_track", "channels")),
                 ("latitudes", "f4", ("along_track", "across_track")),
                 ("longitudes", "f4", ("along_track", "across_track"))]

    def __init__(self, filename):
        self.filename = filename
        self.dn = GPMGMI1C.dn
        self.file_handle = File(filename)
        g = self.file_handle["S1"]
        self.lat_s1 = g['Latitude'][:]
        self.lon_s1 = g['Longitude'][:]
        self.y_s1 = g['Tc'][:]
        g = self.file_handle['S2']
        self.lat_s2 = g['Latitude'][:]
        self.lon_s2 = g['Longitude'][:]
        self.y_s2 = g['Tc'][:]

        self.c = self.lat_s2.shape[1] // 2 - 1

    def _get_indices(self, i, j, dn):
        if dn is None:
            dn = GPMGMI1C.dn
        i_start = i - dn // 2
        i_end   = i + dn // 2
        j_start = j - dn // 2
        j_end   = j + dn // 2
        return i_start, i_end, j_start, j_end, dn

    def get_start_time(self,):
        return GPMGMI1C.product.name_to_date(os.path.basename(self.filename))

    def get_end_time(self,):
        g = self.file_handle['S2']['ScanTime']
        Y = g['Year'][-1]
        M = g['Month'][-1]
        D = g['DayOfMonth'][-1]
        h = g['Hour'][-1]
        m = g['Minute'][-1]
        s = g['Second'][-1]
        return datetime(Y, M, D, h, m, s)

    def get_y(self, i, j, dn = None):
        i_start, i_end, j_start, j_end, dn = self._get_indices(i, j, dn)
        y = np.zeros((2 * dn + 1, 2 * dn + 1, 13))
        y[:, :, :9] = self.y_s1[i_start : i_end, j_start : j_end, :]
        y[:, :, 9:] = self.y_s2[i_start : i_end, j_start : j_end, :]
        return y

    def get_latitudes(self, i, j, dn = None):
        i_start, i_end, j_start, j_end, _ = self._get_indices(i, j, dn)
        return self.lat_s1[i_start : i_end, j_start : j_end]

    def get_longitudes(self, i, j, dn = None):
        i_start, i_end, j_start, j_end, _ = self._get_indices(i, j, dn)
        return self.lon_s1[i_start : i_end, j_start : j_end]

    def get_latitude(self, i = -1, j = -1):
        m, n = self.lat_s1.shape

        if i < 0:
            i = slice(0, m)

        if j < 0:
            j = slice(0, n)

        return self.lat_s1[i, j]

    def get_longitude(self, i = -1, j = -1):
        m, n = self.lon_s1.shape

        if i < 0:
            i = slice(0, m)

        if j < 0:
            j = slice(0, n)

        return self.lon_s1[i, j]

    def get_colocation_centers(self, dn = 50):
        n  = self.lat_s1.shape[0]
        c  = self.lat_s1.shape[1] // 2 - 1
        for i in range(dn // 2, n - dn // 2, dn):
            self.cache = [(i - dn // 2), (i + dn // 2)]
            yield (i, c)

class GPM(Combined):

    kernel = np.array(
        [[8.51724e-11, 3.30060e-08, 2.15373e-06,  2.51412e-05,  5.63712e-05,
          2.51412e-05, 2.15373e-06,  3.30060e-08,  8.51724e-11],
         [6.67905e-10, 2.58826e-07, 1.68892e-05, 0.000197153, 0.000442052,
          0.000197153, 1.68892e-05, 2.58826e-07, 6.67905e-10],
         [2.90088e-09, 1.12415e-06, 7.33538e-05, 0.000856283, 0.00191994,
          0.000856283, 7.33538e-05, 1.12415e-06,  2.90088e-09],
         [6.99457e-09, 2.71053e-06, 0.000176870, 0.00206466, 0.00462935,
          0.00206466, 0.000176870, 2.71053e-06,  6.99457e-09],
         [9.37759e-09, 3.63400e-06, 0.000237129, 0.00276809, 0.00620655,
          0.00276809, 0.000237129, 3.63400e-06,  9.37760e-09],
         [6.99457e-09, 2.71053e-06, 0.000176870, 0.00206466, 0.00462934,
          0.00206466, 0.000176870, 2.71053e-06,  6.99457e-09],
         [2.90088e-09, 1.12415e-06, 7.33538e-05, 0.000856283, 0.00191994,
          0.000856283, 7.33538e-05, 1.12415e-06,  2.90088e-09],
         [6.67905e-10, 2.58826e-07, 1.68892e-05, 0.000197153, 0.000442052,
          0.000197153, 1.68892e-05, 2.58826e-07,  6.67905e-10],
         [8.51723e-11, 3.30060e-08, 2.15373e-06, 2.51412e-05, 5.63712e-05,
          2.51412e-05, 2.15373e-06, 3.30060e-08,  8.51724e-11]]
    )

    products = [gpm_1c_r, gpm_2b_cmb]
    name = "combined"
    dimensions = [("along_track", 2 * GPMGMI1C.dn + 1),
                  ("across_track", 2 * GPMGMI1C.dn + 1)]
    variables = [("rr", "f4", ("along_track", "across_track"))]
    apply_smoothing = True

    def __init__(self, gpm_1c_r_file, gpm_cmb_file):
        from pyresample import kd_tree, geometry
        self.dn = GPMGMI1C.dn

        # GMI for remapping
        self.gmi_file = GPMGMI1C(gmi_file)

        # Combined data
        self.gpm_cmb_file = File(gpm_cmb_file)
        g = self.gpm_cmb_file['MS']
        self.lat_c = g['Latitude'][:]
        self.lon_c = g['Longitude'][:]

        self.sp = g['surfPrecipTotRate'][:]
        self.sp[self.sp == -9999.9] = 0.0
        self.lrf = g['surfLiqRateFrac'][:]
        self.iwc = g['cloudLiqWaterCont'][:]
        self.lwc = g['cloudLiqWaterCont'][:]
        self.pwc = g['precipTotWaterCont'][:]

        if GPM.apply_smoothing:
            self.lat_c = sp.signal.convolve(self.lat_c, GPM.kernel)
            self.lon_c = sp.signal.convolve(self.lon_c, GPM.kernel)
            self.sp = sp.signal.convolve(self.sp, GPM.kernel)
            self.lrf = sp.signal.convolve(self.lrf, GPM.kernel)

            for i in range(self.iwc.shape[-1]):
                self.iwc[:, :, i] = sp.signal.convolve(self.iwc[:, :, i], GPM.kernel)
                self.lwc[:, :, i] = sp.signal.convolve(self.lwc[:, :, i], GPM.kernel)
                self.pwc[:, :, i] = sp.signal.convolve(self.pwc[:, :, i], GPM.kernel)

        try:
            minimum = self.precip.min()
            self.precip[:, [0, -1]] = minimum
            self.precip[[0, -1], :] = minimum
        except:
            pass
        swath_dpr = geometry.SwathDefinition(lats = self.lat_dpr, lons = self.lon_dpr)
        swath_gmi = geometry.SwathDefinition(lats = self.gmi_file.lat_s1, lons = self.gmi_file.lon_s1)
        self.precip_r = kd_tree.resample_nearest(swath_dpr, self.precip, swath_gmi, radius_of_influence = 5e3)

    def get_latitudes(self, i = -1, j = -1):
        m, n = self.lat_dpr.shape

        if i < 0:
            i = slice(0, m)

        if j < 0:
            j = slice(0, n)

        return self.lat_dpr[i, j]

    def get_longitudes(self, i = -1, j = -1):
        m, n = self.lon_dpr.shape

        if i < 0:
            i = slice(0, m)

        if j < 0:
            j = slice(0, n)

        return self.lon_dpr[i, j]


    def get_colocation(self, lat, lon, d_max = 1.0, use_cache = True):
        i, j = np.where(np.logical_and(self.gmi_file.lat_s1 == lat,
                                       self.gmi_file.lon_s1 == lon))
        if len(i) == 0 or len(j) == 0:
            return (0, 0), 1e10
        return (i[0], j[0]), 0.0

    def get_rr(self, i, j, dn = 50):
        i_start = i - dn
        i_end   = i + dn + 1
        j_start = j - dn
        j_end   = j + dn + 1

        lats = self.lat_dpr[i_start : i_end, j_start : j_end]
        lons = self.lon_dpr[i_start : i_end, j_start : j_end]

        return self.precip_r[i_start : i_end, j_start : j_end]


class GPM_2B_CMB(ProductFile):
    """
    GPM Level 2B combined data.

    These files contain retrieved precipitation data from the combined
    retrieval that combined the DPR with GMI observations at the
    swath center.

    In addition to simplified data access, this class defines the variables
    that are extracted from found co-locations.

    Attributes:

        product: The corresponding product object (:code:`products.gpm_2b_cmb`)

        dimensions: Dimensions of the co-located data to extract.

        variables: Names and dimensions of the co-located variables variables
            to extract.
    """

    product = gpm_2b_cmb
    name = "gpm_2b_combined"
    dimensions = [("along_track", 49),
                  ("across_track", 49)]
    variables = [("rr", "f4", ("along_track", "across_track")),
                 ("latitudes", "f4", ("along_track", "across_track")),
                 ("longitudes", "f4", ("along_track", "across_track"))]

    def __init__(self, file):
        self.file_handle = File(file, "r")
        g = self.file_handle['NS']
        self.lat = g['Latitude'][:]
        self.lon = g['Longitude'][:]
        self.precip = g['surfPrecipTotRate']

    def _get_indices(self, i, j):
        dn = 24
        i_start = i - dn
        i_end   = i + dn + 1
        j_start = 0
        j_end   = 49
        return i_start, i_end, j_start, j_end

    def get_latitudes(self, i, j):
        """
        Returns the latitudes of the NS swath of the file.
        """
        i_start, i_end, j_start, j_end = self._get_indices(i, j)
        return self.lat[i_start : i_end, j_start : j_end]

    def get_longitudes(self, i, j):
        """
        Returns the longitudes of the NS swath of the file.
        """
        i_start, i_end, j_start, j_end = self._get_indices(i, j)
        return self.lon[i_start : i_end, j_start : j_end]

    def get_colocation(self, lat, lon, d_max = 1.0, use_cache = True):
        m, n = self.lat.shape
        d = (self.lat - lat) ** 2 + (self.lon - lon) ** 2
        ii = np.argmin(d.ravel())
        i = ii // n 
        j = ii % n

        return (i, j), d[i, j]

    def get_minimum_distance(self, lat, lon):
        """
        Get minimum distance of a given point from the center
        of the swath.

        Arguments:

            lat: The latitude of the point to which to compute the distance.

            lon: The longitude of the point to which to compute the distance
        """
        i_c = self.lat.shape[1] // 2
        lats = self.lat[:, i_c]
        lons = self.lon[:, i_c]
        d = np.sqrt((lats - lat) ** 2 + (lons - lon) ** 2).min()
        return d

    def get_rr(self, i, j):
        """
        Get rain rates around a given co-location center.

        Arguments:

            i: Along-swath index of the co-location index.

            j: Across-swath index of the co-location index.

        """
        dn = 24
        i_start = i - dn
        i_end   = i + dn + 1
        j_start = j - dn - 1
        j_end   = j + dn
        return self.precip[i_start : i_end, :]

    def get_colocation_centers(self):
        """
        Iterate over co-location centers in this file.
        """
        dn = 24
        n  = self.lat.shape[0]
        c  = self.lat.shape[1] // 2 - 1
        for i in range(dn, n - dn, dn):
            yield (i, c)
