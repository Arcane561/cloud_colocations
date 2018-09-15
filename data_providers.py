import json
import os
from datetime import datetime
from ftplib import FTP
from urllib.request import urlopen, urlretrieve
from utils import ensure_extension
import settings

class LAADS:

    base_url = "http://ladsweb.modaps.eosdis.nasa.gov/archive/allData/6/"
    filename_to_datetime = None

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

    def __init__(self):
        pass

    def get_files(self, product, year, day):
        day_str = str(day)
        day_str = "0" * (3 - len(day_str)) + day_str

        url = LAADS.base_url + product + "/" + str(year) + "/" \
                + day_str + ".json"
        return LAADS.__json_to_list__(url, str)

    def get_file(self, filename, dest):
        parts = filename.split(".")
        product = parts[0]
        date    = datetime.strptime(parts[1][1:] + parts[2], "%Y%j%H%M")

        day_str = str(date.timetuple().tm_yday)
        day_str = "0" * (3 - len(day_str)) + day_str

        url = os.path.join(laads_base_url, product,
                           str(date.year), day_str, filename)
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

class ICARE:

    base_url   = "ftp.icare.univ-lille1.fr"
    product_paths = {"CALIOP" : "/SPACEBORNE/CALIOP/01kmCLay.v4.10/",
                     "MYD021KM"    : "/SPACEBORNE/MODIS/MYD021KM.006/",
                     "MYD03"       : "/SPACEBORNE/MODIS/MYD03.006/"}
    filename_to_datetime = None
    separator = "."

    def __ftp_listing_to_list__(path, t = int):
        with FTP(ICARE.base_url) as ftp:
            ftp.login(user = settings.ftp_user, passwd = settings.ftp_password)
            try:
                ftp.cwd(path)
            except:
                raise Exception("Can't find product folder " + path  +
                                ". Are you sure this is an ICARE multi"
                                + "sensor product?")
            ls = ftp.nlst()
        return [t(l) for l in ls]

    def __init__(self, product_path, date_format):
        self.product_path = product_path
        self.date_format  = date_format

    def get_files(self, product, year, day):
        day_str = str(day)
        day_str = "0" * (3 - len(day_str)) + day_str
        date = datetime.strptime(str(year) + str(day_str), "%Y%j")
        path = os.path.join(self.product_path, str(year),
                            date.strftime("%Y_%m_%d"))
        ls = ICARE.__ftp_listing_to_list__(path, str)
        return [l for l in ls if l[-3:] == "hdf"]

    def get_file(self, filename, dest):
        date = self.filename_to_datetime(filename)
        product = filename.split(self.separator)[0].replace("-", "_")
        product_path = self.product_paths[product]
        path = os.path.join(product_path, str(date.year),
                            date.strftime("%Y_%m_%d"))
        filename = ensure_extension(filename, "hdf")
        dest     = ensure_extension(dest, "hdf")

        print(filename, dest)
        with FTP(self.base_url) as ftp:
            ftp.login(user = settings.ftp_user, passwd = settings.ftp_password)
            #try:
            ftp.cwd(path)
            with open(dest, 'wb') as f:
                ftp.retrbinary('RETR ' + filename, f.write)
            #except:
            #    raise Exception("Can't find file " + path  +
            #                    ". Are you sure this is an ICARE multi"
            #                    + "sensor product?")
