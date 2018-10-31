"""
The :code:`products` module contains code for the representation of data
products providing and abstract interface to lookup and download data
files.

Atributes:

    caliop(IcareProduct): The CALIOP 01kmCLay data product.
    modis(IcareProduct): The MODIS MYD021 data product.
    modis(IcareProduct): The MODIS MYD03 geolocation product.

"""
from ftplib   import FTP
from datetime import datetime, timedelta
import os
from . import settings
import shutil
import tempfile

def ensure_extension(path, ext):
    if not path[-len(ext):] == ext:
        path = path + ext
    return path

################################################################################
# File cache
################################################################################

class FileCache:
    """
    Simple file cache to avoid downloading files multiple times.

    Attributes:

        path(str): Path of folder containing the cache
    """
    def __init__(self, path = None):
        """
        Create a file cache.

        Arguments:

            path(str): Folder to use as file path. If not provided a
                temporary directory is created.
        """
        if path is None:
            self.path = tempfile.mkdtemp()
            self.temp = True
        else:
            self.path = path
            self.temp = False

    def get(self, filename):
        """
        Lookup file from cache.

        Arguments:

            filename(str): Filename to lookup.

        Returns:

            The full path of the file in the cache or None if
            the file is not found.

        """
        path = os.path.join(self.path, filename)
        if os.path.isfile(path):
            return path
        else:
            None

    def __del__(self):
        if self.temp:
            shutil.rmtree(self.path)

file_cache = FileCache()

################################################################################
# Icare product
################################################################################

class IcareProduct:
    """
    Base class for data products available from the ICARE ftp server.
    """
    base_url = "ftp.icare.univ-lille1.fr"

    def __init__(self, product_path, name_to_date):
        """
        Create a new product instance.

        Arguments:

        product_path(str): The path of the product. This should point to
            the folder that bears the product name and contains the directory
            tree which contains the data files sorted by date.

        name_to_date(function): Funtion to convert filename to datetime object.
        """
        self.product_path = product_path
        self.name_to_date = name_to_date
        self.cache = {}


    def __ftp_listing_to_list__(self, path, t = int):
        """
        Retrieve directory content from ftp listing as list.

        Arguments:

           path(str): The path from which to retrieve the ftp listing.

           t(type): Type constructor to apply to the elements of the
                listing. To retrieve a list of strings use t = str.

        Return:

            A list containing the content of the ftp directory.

        """
        if not path in self.cache:
            with FTP(IcareProduct.base_url) as ftp:
                ftp.login(user = settings.ftp_user, passwd = settings.ftp_password)
                try:
                    ftp.cwd(path)
                except:
                    raise Exception("Can't find product folder " + path  +
                                    "on the ICARE ftp server.. Are you sure this is"
                                    "a  ICARE multi sensor product?")
                ls = ftp.nlst()
            ls = [t(l) for l in ls]
            self.cache[path] = ls
        return self.cache[path]

    def get_files(self, year, day):
        """
        Return all files from given year and julian day. Files are returned
        in chronological order sorted by the file timestamp.

        Arguments:

            year(int): The year from which to retrieve the filenames.

            day(int): Day of the year of the data from which to retrieve the
                the filenames.

        Return:

            List of all HDF files available of this product on the given date.
        """
        day_str = str(day)
        day_str = "0" * (3 - len(day_str)) + day_str
        date = datetime.strptime(str(year) + str(day_str), "%Y%j")
        path = os.path.join(self.product_path, str(year),
                            date.strftime("%Y_%m_%d"))
        ls = self.__ftp_listing_to_list__(path, str)
        files = [l for l in ls if l[-3:] == "hdf"]
        return files

    def get_preceeding_file(self, filename):
        """
        Return filename of the file that preceeds the given filename in time.

        Arguments:

            filename(str): The name of the file of which to find the preceeding one.

        Returns:

            The filename of the file preceeding the file with the given filename.

        """
        t = self.name_to_date(filename)

        year = t.year
        day  = int((t.strftime("%j")))
        files = self.get_files(year, day)

        i = files.index(filename)

        if i == 0:
            dt = timedelta(days = 1)
            t_p = t - dt
            year = t_p.year
            day  = int((t_p.strftime("%j")))
            return self.get_files(year, day)[-1]
        else:
            return files[i - 1]


    def get_following_file(self, filename):
        """
        Return filename of the file that follows the given filename in time.

        Arguments:

            filename(str): The name of the file of which to find the following file.

        Returns:

            The filename of the file following the file with the given filename.

        """
        t = self.name_to_date(filename)

        year = t.year
        day  = int((t.strftime("%j")))
        files = self.get_files(year, day)

        i = files.index(filename)

        if i == len(files) - 1:
            dt = timedelta(days = 1)
            t_p = t + dt
            year = t_p.year
            day  = int((t_p.strftime("%j")))
            return self.get_files(year, day)[0]
        else:
            return files[i + 1]

    def get_files_in_range(self, t0, t1, t0_inclusive = False):
        """
        Get all files within time range.

        Retrieves a list of product files that include the specified
        time range.

        Arguments:

            t0(datetime.datetime): Start time of the time range

            t1(datetime.datetime): End time of the time range

            t0_inclusive(bool): Whether or not the list should start with
                the first file containing t0 (True) or the first file found
                with start time later than t0 (False).

        Returns:

            List of filename that include the specified time range.

        """
        dt = timedelta(days = 1)

        t = t0
        files = []

        while((t1 - t).total_seconds() > 0.0):

            year = t.year
            day  = int((t.strftime("%j")))

            fs = self.get_files(year, day)

            ts = [self.name_to_date(f) for f in fs]

            dts0 = [self.name_to_date(f) - t0 for f in fs]
            pos0 = [dt.total_seconds() > 0.0 for dt in dts0]

            dts1 = [self.name_to_date(f) - t1 for f in fs]
            pos1 = [dt.total_seconds() > 0.0 for dt in dts1]

            inds = [i for i, (p0, p1) in enumerate(zip(pos0, pos1)) if p0 and not p1]
            files += [fs[i] for i in inds]

            t += dt

        if t0_inclusive:
            f_p = self.get_preceeding_file(files[0])
            files = [f_p] + files

        if not pos1[-1]:
            files += [self.get_following_file(files[-1])]

        return files

    def get_file_by_date(self, t):
        """
        Get file with start time closest to a given date.

        Arguments:

            t(datetime): A date to look for in a file.

        Return:

            The filename of the file with the closest start time
            before the given time.
        """

        # Check last file from previous day
        dt = timedelta(days = 1)
        t_p = t - dt
        year = t_p.year
        day  = int((t_p.strftime("%j")))
        files = self.get_files(year, day - 1)[-1:]

        year = t.year
        day  = int(t.strftime("%j"))
        files += self.get_files(year, day)

        ts  = [self.name_to_date(f) for f in files]
        dts = [tf - t for tf in ts]

        indices = [i for i,dt in enumerate(dts) if dt.total_seconds() > 0.0]
        if len(indices) == 0:
            ind = len(dts) - 1
        else:
            ind = indices[0] - 1

        return files[ind]

    def download_file(self, filename):
        """
        Download a given product file.

        Arguments:

            filename(str): The name of the file to download.

            dest(str): Where to store the file.
        """
        cache_hit = file_cache.get(filename)
        if cache_hit:
            return cache_hit
        else:
            date = self.name_to_date(filename)
            path = os.path.join(self.product_path, str(date.year),
                                date.strftime("%Y_%m_%d"))
            filename = ensure_extension(filename, ".hdf")
            dest     = os.path.join(file_cache.path, filename)

            print("Downloading file ", filename)

            with FTP(self.base_url) as ftp:
                ftp.login(user = settings.ftp_user, passwd = settings.ftp_password)
                ftp.cwd(path)
                with open(dest, 'wb') as f:
                    ftp.retrbinary('RETR ' + filename, f.write)
        return dest

################################################################################
# Filename to date conversion
################################################################################

def modis_name_to_date(s):
    """Convert MODIS filename to date"""
    i = s.index(".")
    s = s[i + 1 :]
    j = s.index(".")
    s = s[: j + 5]
    return datetime.strptime(s, "A%Y%j.%H%M")

def caliop_name_to_date(s):
    """Convert CALIOP name to date"""
    i = s.index(".")
    j = s[i + 1:].index(".") - 4
    s = s[i + 1 : i + j].replace("-", ".")
    return datetime.strptime(s, "%Y.%m.%dT%H.%M")

################################################################################
# Data products
################################################################################

caliop     = IcareProduct("SPACEBORNE/CALIOP/01kmCLay.v4.10", caliop_name_to_date)
modis      = IcareProduct("SPACEBORNE/MODIS/MYD021KM", modis_name_to_date)
modis_geo  = IcareProduct("SPACEBORNE/MODIS/MYD03", modis_name_to_date)

cs = 'CAL_LID_L2_01kmCLay-Standard-V4-10.2010-01-05T00-50-26ZN.hdf'
t0 = datetime(2010, 1, 5, 0, 0, 0)
t1 = datetime(2010, 1, 5, 23, 55, 0)
cf = caliop.get_file_by_date(t0)

cfs = caliop.get_files_in_range(t0, t1)
