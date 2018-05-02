from bs4 import BeautifulSoup
import json
from urllib.request import urlopen, urlretrieve
from cache import CachedFile
from datetime import datetime
import numpy as np
from pyhdf.SD import SD, SDC
import os

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

laads_base_url = "http://ladsweb.modaps.eosdis.nasa.gov/archive/allData/6/"

def __json_to_list__(url, t = int):
    try:
        response = urlopen(url)
    except:
        raise Exception("Can't open product URL " + url +
                        ". Are you sure this is a LAAD product?")

    data = json.loads(response.read())
    ls = []
    for e in data:
        try:
            ls += [t(e["name"])]
        except:
            pass

    return ls

def filename_to_datetime(filename):
    """
    Extract date from LAADS filename.

    Args
        filename(str): The LAADs filename as string.

    Returns: A datetime object representing the date of the first
             entry in the file.
    """
    parts = filename.split(".")
    date = parts[1][1:]
    time = parts[2]
    return datetime.strptime(date + time, "%Y%j%H%M")

def filename_to_product(filename):
    parts = filename.split(".")
    return parts[0]

def filename_to_geo_filename(filename, product):
    parts = filename.split(".")
    parts[0] = product
    return ".".join(parts)

def get_file_by_date(product, date):
    """
    Get LAADS product file that with the first entry that is closest
    in time but before the given date.

    Args
        product(str): Name of the product which to download, i.e. "MYD021KM"
                      for the MODIS on Aqua 1 km dataset.
        date(datetime or str): datetime object or string in format %Y%j%H%M
                               representing the date for which to look for
                               a file.

    Returns
        LAADS filename of the file of the given product that is closest
        to the given date.

    """
    if type(date) is str:
        try:
            date = datetime.strptime(date, "%Y%j%H%M")
        except:
            raise Exception(date + " is not a compatible with date fomat " +
                            "%Y%j%H%M.")
    elif type(date) == datetime:
        pass
    else:
        date = datetime(date)

    dates_available = get_files(product, date.year, date.timetuple().tm_yday)
    dt = [date - filename_to_datetime(d)  for d in dates_available]
    dt_days    = np.array([t.days for t in dt])
    dt_seconds = np.array([t.seconds for t in dt])

    inds = (dt_days == 0)
    if sum(inds) == 0:
        raise Exception("Couldn't find any file close to the given date.")

    inds_sorted = np.argsort(dt_seconds)
    inds = inds_sorted[inds]

    return dates_available[inds[0]]


def get_files(product, year, day):
    day_str = str(day)
    day_str = "0" * (3 - len(day_str)) + day_str

    url = laads_base_url + product + "/" + str(year) + "/" \
            + day_str + ".json"
    return __json_to_list__(url, str)

def get_file(filename, dest):
    parts = filename.split(".")
    product = parts[0]
    date    = datetime.strptime(parts[1][1:] + parts[2], "%Y%j%H%M")

    day_str = str(date.timetuple().tm_yday)
    day_str = "0" * (3 - len(day_str)) + day_str
    print(day_str)

    url = os.path.join(laads_base_url, product, str(date.year), day_str, filename)
    url_base, url_ext = os.path.splitext(url)
    if url_ext is None:
        url = os.path.join(url_base, ".hdf")

    dest_base, dest_ext = os.path.splitext(dest)
    if dest_ext is None:
        dest = dest_base + ".hdf"

    try:
        urlretrieve(url, dest)
    except:
        raise ValueError("File " + str(url) + " not found on LAADS server.")


class LAADSFile(CachedFile):

    def __init__(self, filename, subfolder = "."):
        self.name = filename
        self.date = filename_to_datetime(filename)
        super().__init__(subfolder, filename)

    def get_file(self, dest):
        print("Downloading file: " + self.name)
        get_file(self.name, dest)
        self.file = dest

class ModisFile:

    def from_date(self, product, geo_product, date):
        filename = get_file_by_date(product, date)
        return ModisFile(filename, geo_product)

    def __init__(self, filename, geo_product):
        self.product     = filename_to_product(filename)
        date = filename_to_datetime(filename)
        geo_filename = get_file_by_date(geo_product, date)
        self.geo_product = filename_to_product(geo_filename)

        self.file = LAADSFile(filename, subfolder = self.product)
        self.geo_file = LAADSFile(geo_filename, subfolder = self.product)

        self.file_handle = SD(self.file.file, SDC.READ)
        self.geo_file_handle = SD(self.geo_file.file, SDC.READ)

        self.raw_data = self.file_handle.select("EV_1KM_Emissive")
        self.attributes = self.raw_data.attributes(full = 1)

    def get_lats(self):
        return np.asarray(self.geo_file_handle.select('Latitude')[:, :])

    def get_lons(self):
        return np.asarray(self.geo_file_handle.select('Longitude')[:, :])

    def get_radiances(self, channel = 0):
        data = self.raw_data[channel, :, :].astype(np.double)

        valid_range = self.attributes["valid_range"]
        valid_min = valid_range[0][0]
        valid_max = valid_range[0][1]
        fill_value = self.attributes["_FillValue"]

        invalid = np.logical_or(data > valid_max,
                                data < valid_min)
        invalid = np.logical_or(invalid, data == fill_value)
        data[invalid] = np.nan

        offsets = self.attributes["radiance_offsets"]
        offset = offsets[0][0]

        scales = self.attributes["radiance_scales"]
        scale_factor = scales[0][0]

        data = (data - offset) * scale_factor

        return data

    def plot(self, channel = 0, ax = None):
        if ax is None:
            ax = plt.gca()

        lats = self.get_lats()
        lons = self.get_lons()

        lat_0 = np.mean(lats)
        lon_0 = np.mean(lons)
        # lon_0, lat_0 are the center point of the projection.
        # resolution = 'l' means use low resolution coastlines.
        m = Basemap(projection='ortho',lon_0 = lon_0, lat_0 = lat_0,
                    resolution='l', ax = ax)
        m.drawcoastlines(color = 'grey')
        # draw parallels and meridians.
        m.drawparallels(np.arange(-90.,120.,30.))
        m.drawmeridians(np.arange(0.,420.,60.))

        X, Y = m(lons, lats)
        ax.pcolormesh(X, Y, self.get_radiances(channel = channel))
        plt.title("Full Disk Orthographic Projection")
        plt.show()

class LAADSData:

    base_url = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/6/"

    def __json_to_list__(url, t = int):
        try:
            response = urlopen(url)
        except:
            raise Exception("Can't open product URL " + url +
                            ". Are you sure this is a LAAD product?")

        data = json.loads(response.read())
        ls = []
        for e in data:
            try:
                ls += [t(e["name"])]
            except:
                pass

        return ls

    def get_years(product):

        url = LAADSData.base_url + product + ".json"
        return LAADSData.__json_to_list__(url, int)

    def get_days(product, year):

        url = LAADSData.base_url + product + "/" + str(year) + ".json"
        return LAADSData.__json_to_list__(url, int)

    def __init__(self, product, year, day):
        day_str = str(day)
        day_str = "0" * (3 - len(day_str)) + day_str

        self.url = self.base_url + product + "/" + str(year) + "/" + day_str
        try:
            urlopen(url)
        except:
            raise Exception("Can't open product URL. Are you sure this is a LAAD product?")

        self.files = LAADSData.get_files(product, year, day)

    def get_file_by_index(self, index = 0):
        if index < len(self.files):
            file_url = self.url + "/" + self.files[0]
            urlretrieve(file_url, "cache/" + self.files[0])
        else:
            raise ValueError("File index " + str(index) + "is out of range.")

    def get_file_by_name(self, filename, dest):
        date = filename_to_datetime(filename)
        year = data.strftime("%Y")
        doy  = data.strftime("%j")
        url = os.path.join(LAADSData.base_url, product, str(year), filename)
        try:
            urlretrieve(url, dest)
        except:
            raise ValueError("File name " + str(filename) + " not found on server.")
