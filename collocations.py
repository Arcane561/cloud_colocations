""" A-train collocation extraction.

Contains code that manages the extraction of collocations from
Caliop and Modis files.
"""
from products import Modis, ModisGeo, Caliop
from formats import Caliop01kmclay, ModisMyd03, ModisMyd021km
from netCDF4 import Dataset

import geopy.distance
import glob
import numpy as np
import os
import shutil
import tempfile

class OutputFile:
    """
    The output file format, that takes care of storing the extracted
    collocations in NetCDF4 format.
    """
    def __init__(self, base_path, year, day, dn):
        """
        Create a collocation file for a given Julia day and year.
        Files are sorted in a folder hierarchy by year and day. If
        these folders do not exist, they will be created.

        A collocation consists of MODIS data in a region extending
        :code:`dn` pixels in each orientation in along- and across-track
        direction from the central CALIOP profile as well as data from
        :code:`2 * dn + 1` CALIOP profiles centered around this center.

        Parameters:
            base_path(str): The base folder in which the collocation
                files will be stored.

            year(int): The year of the collocations
            day(int): The Julian day of the collocations.
            dn(int): Number of pixels the collocations should extend into
                the MODIS swath away from the pixel closest to the CALIOP
                profile.
        """

        f = os.path.join(base_path, str(year), str(day))
        if not os.path.exists(f):
            os.makedirs(f)
        self.root = Dataset(os.path.join(f, "collocations_{0}.nc".format(dn)),
                            "w")

        self.ci = self.root.createDimension("collocations", None)
        self.dn = self.root.createDimension("collocation_size", 2 * dn + 1)
        self.layers = self.root.createDimension("cloud_layers", 4)
        self.bands = self.root.createDimension("modis_bands", 7)

        self.ctp = self.root.createVariable("cloud_top_pressure", "f4",
                                            ("collocations",
                                             "collocation_size",
                                             "cloud_layers"))
        self.cbp = self.root.createVariable("cloud_base_pressure", "f4",
                                            ("collocations",
                                             "collocation_size",
                                             "cloud_layers"))

        self.cta = self.root.createVariable("cloud_top_altitude", "f4",
                                            ("collocations",
                                             "collocation_size",
                                             "cloud_layers"))
        self.cba = self.root.createVariable("cloud_base_altitude", "f4",
                                            ("collocations",
                                             "collocation_size",
                                             "cloud_layers"))
        self.caliop_lats = self.root.createVariable("caliop_lats", "f4",
                                                    ("collocations",
                                                     "collocation_size"))
        self.caliop_lons = self.root.createVariable("caliop_lons", "f4",
                                                    ("collocations",
                                                     "collocation_size"))

        self.modis_data = self.root.createVariable("modis_data", "f4",
                                                    ("collocations",
                                                     "modis_bands",
                                                     "collocation_size",
                                                     "collocation_size"))
        self.modis_lats = self.root.createVariable("modis_lats", "f4",
                                                    ("collocations",
                                                     "collocation_size",
                                                     "collocation_size"))
        self.modis_lons = self.root.createVariable("modis_lons", "f4",
                                                    ("collocations",
                                                     "collocation_size",
                                                     "collocation_size"))
        self.ci = 0

    def add(self, ctp, cbp, cta, cba, caliop_lats,
            caliop_lons, modis_data, modis_lats, modis_lons):
        """
        Add a collocation to the file.

        Parameters:
            ctp(numpy.ndarray): 2D array containing the cloud TOP PRESSURE
                for the first four cloud layers along the profiles in the
                collocation.
            cbp(numpy.ndarray): 2D array containing the cloud BASE PRESSURE
                for the first four cloud layers along the profiles in the
                collocation.
            cta(numpy.ndarray): 2D array containing the cloud TOP PRESSURE
                for the first four cloud layers along the profiles in the
                collocation.
            cba(numpy.ndarray): 2D array containing the cloud BASE PRESSURE
                for the first four cloud layers along the profiles in the
                collocation.
            cba(numpy.ndarray): 2D array containing the cloud BASE PRESSURE
                for the first four cloud layers along the profiles in the
                collocation.
            caliop_lats(numpy.ndarray): 1D array containing the latitudes
                of the Caliop profiles in the collocation.
            caliop_lons(numpy.ndarray): 1D array containing the longitudes
                of the Caliop profiles in the collocation.
            modis_data(numpy.ndarray): 3D array containing the MODIS
                of the 7 infra-red channels.
            modis_lats(numpy.ndarray): 2D array containing the MODIS
                latitudes for each pixel of the collocation.
            modis_lons(numpy.ndarray): 2D array containing the MODIS
                longitudes for each pixel of the collocation.
        """
        self.ctp[self.ci, :, :] = ctp
        self.cbp[self.ci, :, :] = cbp
        self.cta[self.ci, :, :] = cta
        self.cba[self.ci, :, :] = cba
        self.caliop_lats[self.ci, :] = caliop_lats.ravel()
        self.caliop_lons[self.ci, :] = caliop_lons.ravel()
        self.modis_data[self.ci, :, :, :] = modis_data
        self.modis_lats[self.ci, :, :] = modis_lats
        self.modis_lons[self.ci, :, :] = modis_lons
        self.ci += 1

    def close(self):
        self.root.close()

class Collocations:
    """
    Implements the general collocation-extraction workflow:

    1. Get names of all files for that day.
    2. Download the files.
    3. Extract collocations.
    4. Remove temporary files.

    """
    def __init__(self, year, day, products):
        """
        Create a collocation object for the given year, day and products.

        Creates a temporary folders, to which all data from this day
        will be downloaded.

        Retrieves file names from all files available for this day from
        the webserver.
        """
        self.products = products
        self.year = year
        self.day = day

        self.folder = tempfile.mkdtemp()
        self.files = [p.get_files(self.year, self.day) for p in self.products]

    def download_files(self):
        """
        Download all files to the temporary directory.
        """
        file_types = sum([[i] * len(self.files[i]) \
                            for i in range(len(self.files))], [])

        files = sum(self.files, [])
        dates = [self.products[file_types[i]].filename_to_date(files[i])
                    for i in range(len(files))]

        inds = np.argsort(dates)
        for i in inds:
            t = file_types[i]
            self.products[t].download_file(files[i], self.folder)

    def remove_folder(self):
        """
        Remove temporary folder if still existant.
        """
        if not self.folder is None:
            shutil.rmtree(self.folder)

    def find_collocation(lats_modis, lons_modis, lat, lon, margin, dmax):
        """
        Finds indices of the center of a collocation in a modis file for
        given coordinates of a Caliop profile.

        Parameters:
            lats_modis(numpy.ndarray): 2D array containing the latitudes
                of the Modis pixels.
            lons_modis(numpy.ndarray): 2D array containing the longitudes
                of the Modis pixels.
            lat(np.float): Latitude coordinate of the Caliop profile.
            long(np.float): Longitude of the Caliop profile.
            margin(int): Number of pixels the found center must be away
                from the limits of the modis swath.
            dmax(np.float): Maximum allowed distance between found center
                pixel and given coordinates.
        Returns:

            Tuple containing the first and second index of the closest
            pixel as well as the distance to the provided coordinates.
            If the distance is larger than dmax or not within the margins
            of the dimensions of the modis data, the indices returned are
            -1.
        """
        d = np.sqrt((lats_modis - lat) ** 2.0 + (lons_modis - lon) ** 2.0)

        i_min = np.argmin(d)
        i = i_min // lats_modis.shape[1]
        j = i_min % lats_modis.shape[1]


        d = geopy.distance.distance((lat, lon),
                                    (lats_modis[i, j], lons_modis[i, j]))

        if i < margin or j < margin or \
           i >= lats_modis.shape[0] - margin or \
           j >= lats_modis.shape[1] - margin:
            i = -1
            j = -1

        return i, j, d

    def find_collocations(collocation,
                          dn = 50,
                          maxdist = 1.0):
        """
        Implements the processing loop for the extraction of collocations.
        Reads consecutively through all caliop and modis files from the
        given day and extracts the collocations.

        Collocations are extracted so that they don't overlap.

        """

        path = "/home/simonpf/Dendrite/UserAreas/Simon/cloud_collocations"
        output_file = OutputFile(path, collocation.year, collocation.day, dn)

        caliop_files = collocation.files[2]
        modis_geo_files = collocation.files[1]
        modis_files = collocation.files[0]

        # Caliop sorted files indices and file index
        fis_c = np.argsort([collocation.products[2].filename_to_date(f) \
                            for f in caliop_files])
        fi_c = 0

        # Modis sorted file indices and file index
        fis_m_g = np.argsort([collocation.products[1].filename_to_date(f) \
                            for f in modis_geo_files])
        fis_m = np.argsort([collocation.products[0].filename_to_date(f) \
                            for f in modis_files])
        fi_m = 0

        d = 0.0
        d_old = 0.0


        f_c = Caliop01kmclay(os.path.join(collocation.folder,
                                        caliop_files[fis_c[fi_c]]))
        lats_c = f_c.get_latitudes()
        lons_c = f_c.get_longitudes()
        f_m = ModisMyd03(os.path.join(collocation.folder,
                                    modis_geo_files[fis_m_g[fi_m]]))
        f_md = ModisMyd021km(os.path.join(collocation.folder,
                                        modis_files[fis_m[fi_m]]))
        lats_m = f_m.get_latitudes()
        lons_m = f_m.get_longitudes()
        i_c = dn

        while fi_c < len(fis_c) and fi_m < len(fis_m) and fi_m < len(fis_m_g):


            d_old = d
            j_m, k_m, d = self.find_collocation(lats_m, lons_m, lats_c[i_c],
                                        lons_c[i_c], dn, 1.0)

            if d < maxdist and j_m > 0 and k_m > 0:
                try:
                    output_file.add(f_c.get_top_pressure(i_c, dn),
                                    f_c.get_base_pressure(i_c, dn),
                                    f_c.get_top_altitude(i_c, dn),
                                    f_c.get_base_altitude(i_c, dn),
                                    lats_c[i_c - dn : i_c + dn + 1],
                                    lons_c[i_c - dn : i_c + dn + 1],
                                    f_md.get_input_data(j_m, k_m, dn),
                                    lats_m[j_m - dn : j_m + dn + 1,
                                        k_m - dn : k_m + dn + 1],
                                    lons_m[j_m - dn : j_m + dn + 1,
                                        k_m - dn : k_m + dn + 1])
                except:
                    print("Error day {0}: {1} {2}".format(collocation.day,
                                                          j_m, k_m))

            # If distance increase go to next MODIS file.
            if d_old > 2.0 * maxdist and d > 2.0 * maxdist and d > d_old:
                fi_m += 1
                if fi_m < len(fis_m) and fi_m < len(fis_m_g):
                    f_m = ModisMyd03(os.path.join(collocation.folder,
                                                modis_geo_files[fis_m_g[fi_m]]))
                    f_md = ModisMyd021km(os.path.join(collocation.folder ,
                                                    modis_files[fis_m[fi_m]]))
                    lats_m = f_m.get_latitudes()
                    lons_m = f_m.get_longitudes()

                    if i_c > 2 * dn:
                        i_c -= 2 * dn
                continue

            i_c += 2 * dn

            if i_c > lats_c.size - dn:
                fi_c += 1
                if fi_c < len(fis_c):
                    f_c = Caliop01kmclay(os.path.join(collocation.folder,
                                                    caliop_files[fis_c[fi_c]]))
                    lats_c = f_c.get_latitudes()
                    lons_c = f_c.get_longitudes()
                    i_c = dn
                    print("Opening caliop file: ",
                          collocation.products[2].filename_to_date(
                              caliop_files[fis_c[fi_c]]
                          )
                    )
                continue

        output_file.close()
        print("Finished processing day {0}.".format(collocation.day))

    def find_collocations_overlapping(collocation,
                                      dn = 50,
                                      maxdist = 1.0):
        """
        This function extracts all consecutive collocations around all
        Caliop profiles. This is for comparison with the SMHI data.
        If no complete collocation can be extracted the modis data will
        be set to nan.
        """

        path = ("/home/simonpf/Dendrite/UserAreas/Simon/cloud_collocations/"
                "test_data")
        output_file = OutputFile(path, collocation.year, collocation.day, dn)

        caliop_files = collocation.files[2]
        modis_geo_files = collocation.files[1]
        modis_files = collocation.files[0]

        # Caliop sorted file indices and file index
        fis_c = np.argsort([collocation.products[2].filename_to_date(f) \
                            for f in caliop_files])
        fi_c = 0

        # Modis sorted file indices and file index
        fis_m_g = np.argsort([collocation.products[1].filename_to_date(f) \
                            for f in modis_geo_files])
        fis_m = np.argsort([collocation.products[0].filename_to_date(f) \
                            for f in modis_files])
        fi_m = 0

        d = 0.0
        d_old = 0.0


        f_c = Caliop01kmclay(os.path.join(collocation.folder,
                                        caliop_files[fis_c[fi_c]]))
        lats_c = f_c.get_latitudes()
        lons_c = f_c.get_longitudes()
        f_m = ModisMyd03(os.path.join(collocation.folder,
                                    modis_geo_files[fis_m_g[fi_m]]))
        f_md = ModisMyd021km(os.path.join(collocation.folder,
                                        modis_files[fis_m[fi_m]]))
        lats_m = f_m.get_latitudes()
        lons_m = f_m.get_longitudes()
        i_c = i_c

        while fi_c < len(fis_c) and fi_m < len(fis_m) and fi_m < len(fis_m_g):


            d_old = d
            j_m, k_m, d = self.find_collocation(lats_m, lons_m, lats_c[i_c],
                                        lons_c[i_c], 0, 1.0)

            try:
                output_file.add(f_c.get_top_pressure(i_c, dn),
                                f_c.get_base_pressure(i_c, dn),
                                f_c.get_top_altitude(i_c, dn),
                                f_c.get_base_altitude(i_c, dn),
                                lats_c[i_c - dn : i_c + dn + 1],
                                lons_c[i_c - dn : i_c + dn + 1],
                                f_md.get_input_data(j_m, k_m, dn),
                                lats_m[j_m - dn : j_m + dn + 1,
                                    k_m - dn : k_m + dn + 1],
                                lons_m[j_m - dn : j_m + dn + 1,
                                    k_m - dn : k_m + dn + 1])
            except:
                output_file.add(f_c.get_top_pressure(i_c, dn),
                                f_c.get_base_pressure(i_c, dn),
                                f_c.get_top_altitude(i_c, dn),
                                f_c.get_base_altitude(i_c, dn),
                                lats_c[i_c - dn : i_c + dn + 1],
                                lons_c[i_c - dn : i_c + dn + 1],
                                np.nan * np.zeros((2 * dn + 1,
                                                   2 * dn + 1, 7)),
                                lats_m[j_m - dn : j_m + dn + 1,
                                    k_m - dn : k_m + dn + 1],
                                lons_m[j_m - dn : j_m + dn + 1,
                                    k_m - dn : k_m + dn + 1])

            # If distance increase go to next MODIS file.
            if d_old > 2.0 * maxdist and d > 2.0 * maxdist and d > d_old:
                fi_m += 1
                if fi_m < len(fis_m) and fi_m < len(fis_m_g):
                    f_m = ModisMyd03(os.path.join(collocation.folder,
                                                modis_geo_files[fis_m_g[fi_m]]))
                    f_md = ModisMyd021km(os.path.join(collocation.folder ,
                                                    modis_files[fis_m[fi_m]]))
                    lats_m = f_m.get_latitudes()
                    lons_m = f_m.get_longitudes()

                    if i_c > 2 * dn:
                        i_c -= 2 * dn
                continue

            i_c += 1

            if i_c > lats_c.size - dn:
                fi_c += 1
                if fi_c < len(fis_c):
                    f_c = Caliop01kmclay(os.path.join(collocation.folder,
                                                    caliop_files[fis_c[fi_c]]))
                    lats_c = f_c.get_latitudes()
                    lons_c = f_c.get_longitudes()
                    i_c = dn
                    print("Opening caliop file: ",
                          collocation.products[2].filename_to_date(
                              caliop_files[fis_c[fi_c]]
                          )
                    )
                continue

        output_file.close()
        print("Finished processing day {0}.".format(collocation.day))

dn = 50

def process_day(year, day):
    c = Collocations(year, day, [Modis(), ModisGeo(), Caliop()])
    c.download_files()
    c.find_collocations(c)
    c.remove_folder()


#c = Collocations(2010, 100, [Modis(), ModisGeo(), Caliop()])
#c.download_files()

#files = glob.glob("temp/CAL*")
#cf = Caliop01kmclay(files[0])
#
#files = glob.glob("temp/MYD03*")
#mf = ModisMyd03(files[0])
#
#files = glob.glob("temp/MYD021*")
#mf_2 = ModisMyd021km(files[0])

#find_collocations(c)

