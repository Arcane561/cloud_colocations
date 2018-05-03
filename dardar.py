from bs4 import BeautifulSoup
from cache import CachedFile
import json
import numpy as np
import os
from urllib.request import urlopen, urlretrieve
from ftplib import FTP
from datetime import datetime
from pyhdf.SD import SD, SDC

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

from utils import ensure_extension

icare_ftp_url   = "ftp.icare.univ-lille1.fr"
icare_base_path = "/SPACEBORNE/MULTI_SENSOR/"

def filename_to_datetime(filename):
    parts = filename.split("_")
    return datetime.strptime(parts[2], "%Y%j%H%M%S")

def __ftp_listing_to_list__(path, t = int):
    with FTP(icare_ftp_url) as ftp:
        ftp.login(user = "simonpf", passwd = "dardar_geheim!")
        try:
            ftp.cwd(path)
        except:
            raise Exception("Can't find product folder " + path  +
                            ". Are you sure this is an ICARE multi"
                            + "sensor product?")
        ls = ftp.nlst()
    return [t(l) for l in ls]

def get_years(product):
    path = os.path.join(icare_base_path, product)
    return __ftp_listing_to_list__(path)

def get_days(product, year):
    to_date = lambda x: int(datetime.strptime(x, "%Y_%m_%d").strftime("%j"))
    path = os.path.join(icare_base_path, product, str(year))
    return __ftp_listing_to_list__(path, to_date)

def get_files(product, year, day):
    day_str = str(day)
    day_str = "0" * (3 - len(day_str)) + day_str
    date = datetime.strptime(str(year) + str(day_str), "%Y%j")
    path = os.path.join(icare_base_path, product, str(year),
                        date.strftime("%Y_%m_%d"))
    ls = __ftp_listing_to_list__(path, str)
    return [l for l in ls if l[-3:] == "hdf"]


    path = os.path.join(icare_base_path, product, str(year), str(day))
    return ICAREData.__ftp_listing_to_list__(path)

def get_days(product, year):
    to_date = lambda x: int(datetime.strptime(x, "%Y_%m_%d").strftime("%j"))
    path = os.path.join(icare_base_path, product, str(year))
    return ICAREData.__ftp_listing_to_list__(path, to_date)

def get_file(filename, dest):
    date = filename_to_datetime(filename)
    product = filename.split("_")[0].replace("-", "_")
    path = os.path.join(icare_ftp_url, icare_base_path, product,
                        str(date.year), date.strftime("%Y_%m_%d"))
    filename = ensure_extension(filename, "hdf")
    dest     = ensure_extension(dest, "hdf")

    print(filename, dest)
    with FTP(icare_ftp_url) as ftp:
        ftp.login(user = "simonpf", passwd = "dardar_geheim!")
        #try:
        print(filename, dest)
        ftp.cwd(path)
        with open(dest, 'wb') as f:
            ftp.retrbinary('RETR ' + filename, f.write)
        #except:
        #    raise Exception("Can't find file " + path  +
        #                    ". Are you sure this is an ICARE multi"
        #                    + "sensor product?")

def get_file_by_date(product, date):
    if type(date) is str:
        try:
            date = datetime.strptime(date, "%Y%j%H%M")
        except:
            raise Exception(date + " is not a compatible with date fomat " +
                            "%Y%j%H%M.")
    elif type(date) is not datetime:
        date = datetime(date)

    dates_available = get_files(product, date.year,
                                date.timetuple().tm_yday)
    dt = [date - filename_to_datetime(d)  for d in dates_available]
    dt_days    = np.array([t.days for t in dt])
    dt_seconds = np.array([t.seconds for t in dt])

    inds = (dt_days == 0)
    if sum(inds) == 0:
        raise Exception("Couldn't find any file close to the given date.")

    inds_sorted = np.argsort(dt_seconds)
    inds = inds_sorted[inds]

    return dates_available[inds[0]]

class ICAREFile(CachedFile):

    def __init__(self, product, date):
        if type(date) is str:
            try:
                date = datetime.strptime(date, "%Y%j%H%M")
            except:
                raise Exception(date + " is not a compatible with date fomat " +
                                "%Y%j%H%M.")
        elif type(date) is not datetime:
            date = datetime(date)

        self.date = date
        self.product     = product
        self.name = get_file_by_date(product, date)
        super().__init__(product, self.name)

        self.file_handle = SD(self.file, SDC.READ)
        datasets = self.file_handle.datasets()

    def get_file(self, dest):
       print("Downloading file: " + self.name)
       get_file(self.name, dest)
       self.file = dest

    def get_lats(self):
        return np.asarray(self.file_handle.select('CLOUDSAT_Latitude')[:])

    def get_lons(self):
        return np.asarray(self.file_handle.select('CLOUDSAT_Longitude')[:])

    def get_cloud_types(self):
        return np.max(np.asarray(self.file_handle.select('CLOUDSAT_Cloud_Scenario')[:, :]), axis = 1)

    def plot_footprint(self, ax = None):
        if ax is None:
            ax = plt.gca()

        lats = self.get_lats()
        lons = self.get_lons()

        print(lats)
        print(lons)

        lat_0 = lats[0]
        lon_0 = lons[0]
        # lon_0, lat_0 are the center point of the projection.
        # resolution = 'l' means use low resolution coastlines.
        m = Basemap(projection='ortho',lon_0 = lon_0, lat_0 = lat_0,
                    resolution='l', ax = ax)
        m.drawcoastlines(color = 'grey')
        # draw parallels and meridians.
        m.drawparallels(np.arange(-90.,120.,30.))
        m.drawmeridians(np.arange(0.,420.,60.))

        X, Y = m(lons, lats)
        ax.scatter(X, Y)
        plt.title("Full Disk Orthographic Projection")
        plt.show()





class ICAREData:

    ftp_url = "ftp.icare.univ-lille1.fr"
    base_path = "/SPACEBORNE/MULTI_SENSOR/"

    def __ftp_listing_to_list__(path, t = int):
        with FTP(ICAREData.ftp_url) as ftp:
            ftp.login(user = "simonpf", passwd = "dardar_geheim!")
            try:
                ftp.cwd(path)
            except:
                raise Exception("Can't find product folder " + path  +
                                ". Are you sure this is an ICARE multi"
                                + "sensor product?")
            ls = ftp.nlst()
        return [t(l) for l in ls]

    def get_years(product):
        path = os.path.join(ICAREData.base_path, product)
        return ICAREData.__ftp_listing_to_list__(path)

    def get_days(product, year):
        to_date = lambda x: int(datetime.strptime(x, "%Y_%m_%d").strftime("%j"))
        path = os.path.join(ICAREData.base_path, product, str(year))
        return ICAREData.__ftp_listing_to_list__(path, to_date)

    def __init__(self):
        self.ftp = FTP(ICAREData.ftp_url)
        self.ftp.login(user = "simonpf", passwd = "dardar_geheim!")
        self.ftp.cwd("SPACEBORNE")
        self.ftp.cwd("MULTI_SENSOR")

        ls = self.ftp.retrlines("NLST")
        print(ls)

