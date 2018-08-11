from bs4 import BeautifulSoup
from cache import get_cached_files, CachedFile
import json
import numpy as np
import os
from urllib.request import urlopen, urlretrieve
from ftplib import FTP
from datetime import datetime
from pyhdf.SD import SD, SDC
import settings

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
        ftp.login(user = settings.ftp_user, passwd = settings.ftp_password)
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
        ftp.login(user = settings.ftp_user, passwd = settings.ftp_password)
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

    cached_files = get_cached_files(product,
            product.replace("_", "-") + "*.hdf")
    cached_dates = [filename_to_datetime(f) for f in cached_files]
    print(product)
    dt_seconds = np.array([(date - cd).seconds for cd in cached_dates])
    dt_days    = np.array([(date - cd).days for cd in cached_dates])
    inds = [i for i,(s, d) in enumerate(zip(dt_seconds, dt_days)) if s == 0 and d == 0]
    if len(inds) > 0:
        return cached_files[inds[0]]

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

    def get_cloud_scenario(self):
        return np.asarray(self.file_handle.select('CLOUDSAT_Cloud_Scenario')[:, :])

    def get_cloud_scenario_caliop(self):
        return np.asarray(self.file_handle.select('CALIOP_Mask_Refined')[:, :])

    def get_cloudsat_reflectivity(self):
        return np.asarray(self.file_handle.select('CLOUDSAT_2B_GEOPROF_Radar_Reflectivity')[:, :])

    def get_caliop_backscatter(self):
        return np.asarray(self.file_handle.select('CALIOP_Total_Attenuated_Backscatter_532')[:, :])

    def get_cloudsat_reflectivity(self):
        return np.asarray(self.file_handle.select('CLOUDSAT_2B_GEOPROF_Radar_Reflectivity')[:, :])

    def get_height(self):
        return np.asarray(self.file_handle.select('CS_TRACK_Height')[:])

    def get_cloud_height(self, start = 0, end = -1):

        i = (start + end) // 2
        if end > 0:
            cloud_scenario = np.asarray(self.file_handle.select('CALIOP_Mask_Refined')[:, :])[i, :]
        else:
            cloud_scenario = np.asarray(self.file_handle.select('CALIOP_Mask_Refined')[i, :])


        inds = np.where(cloud_scenario == 2)[0]
        print(inds)
        if len(inds) > 1:
            z = np.asarray(self.file_handle.select('CS_TRACK_Height')[:])
            return z[inds[0]]
        else:
            return - 9999.0

    def get_mean_height(self, subsampling):
        z = np.asarray(self.file_handle.select('CS_TRACK_Height')[:])
        n_bins = z.shape[0]  // subsampling
        if z.shape[0] % subsampling > 0:
            n_bins += 1
        z_means = np.zeros(n_bins)

        si = 0
        for i in range(n_bins):
            if i < n_bins - 1:
                z_means[i] = np.mean(z[si:si + subsampling])
            else:
                z_means[i] = np.mean(z[si:])
            si += subsampling
        return z_means
        
    def get_subsampled_cloud_scenario(self, start = 0, end = -1, subsampling = 20):

        if end > 0:
            cloud_scenario = np.asarray(self.file_handle.select('CLOUDSAT_Cloud_Scenario')[:, :])[start:end, :]
        else:
            cloud_scenario = np.asarray(self.file_handle.select('CLOUDSAT_Cloud_Scenario'))[start :, :]

        n = cloud_scenario.shape[0]
        n_bins = cloud_scenario.shape[1]  // subsampling
        if cloud_scenario.shape[1] % subsampling > 0:
            n_bins += 1

        cloud_scenario_simple = np.zeros((n, n_bins))

        bins = np.arange(1, 11) - 0.5
        for i in range(n):
            si = 0
            for j in range(n_bins):
                if j < n_bins - 1:
                    hs, _ = np.histogram(cloud_scenario[i, si:si + subsampling], bins = bins)
                else:
                    hs, _ = np.histogram(cloud_scenario[i, si:], bins = bins)

                if hs.sum() == 0.0:
                    cloud_scenario_simple[i, j] = 0.0
                else:
                    cloud_scenario_simple[i, j] = np.argmax(hs)

                cloud_scenario_simple[i, j] = np.argmax(hs)
                si += subsampling

        return cloud_scenario_simple

    def get_subsampled_cloud_scenario_dardar(self, start = 0, end = -1, subsampling = 20):

        if end > 0:
            cloud_scenario = np.asarray(self.file_handle.select('CALIOP_Mask_Refined')[:, :])[start:end, :]
        else:
            cloud_scenario = np.asarray(self.file_handle.select('CALIOP_Mask_Refined'))[start :, :]

        n = cloud_scenario.shape[0]
        n_bins = cloud_scenario.shape[1]  // subsampling
        if cloud_scenario.shape[1] % subsampling > 0:
            n_bins += 1

        cloud_scenario_simple = np.zeros((n, n_bins))

        bins = np.arange(-1, 13) - 0.5
        for i in range(n):
            si = 0
            for j in range(n_bins):
                if j < n_bins - 1:
                    hs, _ = np.histogram(cloud_scenario[i, si:si + subsampling], bins = bins)
                else:
                    hs, _ = np.histogram(cloud_scenario[i, si:], bins = bins)
                cloud_scenario_simple[i, j] = np.argmax(hs)
                si += subsampling

        return cloud_scenario_simple

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
            ftp.login(user = settings.ftp_user, passwd = settings.ftp_password)
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
        self.ftp.login(user = settings.ftp_user, passwd = settings.ftp_password)
        self.ftp.cwd("SPACEBORNE")
        self.ftp.cwd("MULTI_SENSOR")

        ls = self.ftp.retrlines("NLST")
        print(ls)

