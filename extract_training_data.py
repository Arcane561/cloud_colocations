"""
Functions to extract training data from collocation data.
"""

from scipy.spatial import cKDTree
from netCDF4 import Dataset

import glob
import numpy as np
import os

base_path = "/home/simonpf/Dendrite/UserAreas/Simon/cloud_collocations/"
dest_path = "/home/simonpf/src/cloud_collocations/data/"

################################################################################
# Simple training data
################################################################################

class TrainingData:
    """
    The :class:`TrainingData` class extract the *simple* training data from
    the collocation data. The simple training data consists only of the
    MODIS input and the corresponding cloud top pressure at the center of
    the MODIS input.
    """
    def __init__(self, dn):
        """
        Create a :class:`TrainingData` object that extracts samples of
        input size :code:`2 * dn + 1`.

        Parameters:
            dn(int): The size of the input data to extract.
        """

        path = os.path.join(base_path, "2010/*/*" + str(dn) + ".nc")
        self.files = glob.glob(path)

        self.dn = dn

        self.of = Dataset(os.path.join(dest_path, "training_data_{0}.nc".format(dn)), "w")
        self.of.createDimension("collocations", None)
        self.of.createDimension("input_size", 2 * dn + 1)
        self.of.createDimension("channels", 7)
        self.x = self.of.createVariable("x", "f4", ("collocations",
                                               "channels",
                                               "input_size",
                                               "input_size"))
        self.y = self.of.createVariable("y", "f4", "collocations")

    def extract(self, n = -1):
        """
        Extract training data from collocation data.

        Parameters:
            n(int): Number of files to extract data from. If negative,
            all data from all files that were found will be extracted.

        """
        dn = self.dn

        if n < 0:
            fs = self.files
        else:
            fs = self.files[:n]
        fi = 0

        for cf in fs:
            print("Processing file {0}".format(cf))
            ds = Dataset(cf)
            nc = ds.dimensions["collocations"].size
            dn0 = (ds.dimensions["collocation_size"].size - 1) // 2

            self.x[fi : fi + nc, :, :, :] = ds.variables["modis_data"]\
                                            [:, :, dn0 - dn : dn0 + dn + 1,
                                             dn0 - dn : dn0 + dn + 1]
            self.y[fi : fi + nc] = ds.variables["cloud_top_altitude"][:, dn0, 0]
            fi += nc
            ds.close()

    def close(self):
        """
        Close the output file. This must be called to filnalize the NetCDF
        file containing the training data.
        """
        self.of.close()


################################################################################
# Complex training data
################################################################################

class ComplexTrainingData:
    """
    The :class:`ComplexTrainingData` class extract the complex training data
    from the collocation data. The complex training data consists of the MODIS
    input and all cloud top pressures along the CALIOP transect within the
    MODIS input.
    """
    def __init__(self, dn, ddn = 5):
        """
        Create a :class:`TrainingData` object that extracts samples of
        input size :code:`2 * dn + 1`.

        If a :code:`dn` is chosen that is smaller than the extracted
        collocations, multiple scenes from the input collocation will
        be extracted. This will be shifted across-track starting with
        an offset of :code:`-dn` up to :code:`dn` with steps of
        :code:`ddn`. This allows varying the position of the CALIOP
        profiles with respect to the MODIS input data.

        Parameters:
            dn(int): The size of the input data to extract.

            ddn(int): The step size for across-track translated subscenes.
        """
        path = os.path.join(base_path, "2010/*/*.nc")
        self.files = glob.glob(path)

        self.dn = dn
        self.ddn = ddn

        self.of = Dataset(os.path.join(dest_path, "complex_training_data_{0}.nc".format(dn)), "w")
        self.of.createDimension("collocations", None)
        self.of.createDimension("input_size", 2 * dn + 1)
        self.of.createDimension("channels", 7)
        self.x = self.of.createVariable("x", "f4", ("collocations",
                                                    "channels",
                                                    "input_size",
                                                    "input_size"))
        self.y = self.of.createVariable("y", "f4", ("collocations",
                                                    "input_size",
                                                    "input_size"))
        self.cm = self.of.createVariable("cloud_mask", "i4", ("collocations",
                                                              "input_size",
                                                              "input_size"))
        self.vm = self.of.createVariable("validity_mask", "i4", ("collocations",
                                                                 "input_size",
                                                                 "input_size"))
        self.x_lats = self.of.createVariable("x_lats", "f4",  ("collocations",
                                                               "input_size",
                                                               "input_size"))
        self.x_lons = self.of.createVariable("x_lons", "f4",  ("collocations",
                                                               "input_size",
                                                               "input_size"))
        self.y_lats = self.of.createVariable("y_lats", "f4",  ("collocations",
                                                               "input_size"))
        self.y_lons = self.of.createVariable("y_lons", "f4",  ("collocations",
                                                               "input_size"))


    def extract(self, n = -1):
        fi = 0

        if n < 0:
            fs = self.files
        else:
            fs = self.files[:n]

        for cf in fs:
            print("Processing file {0}".format(cf))
            ds = Dataset(cf)
            nc = ds.dimensions["collocations"].size
            dn0 = (ds.dimensions["collocation_size"].size - 1) // 2

            for i in range(nc):
                c_lats = ds.variables["caliop_lats"][i, :]
                c_lons = ds.variables["caliop_lons"][i, :]
                c_lons *= np.cos(np.radians(c_lats))
                c_tree = cKDTree(np.hstack([c_lats.reshape(-1, 1),
                                            c_lons.reshape(-1, 1)]),
                                 copy_data = True)

                m_lats = ds.variables["modis_lats"][i, :, :]
                m_lons = ds.variables["modis_lons"][i, :, :]
                m_lons *= np.cos(np.radians(m_lats))

                m_data  = ds.variables["modis_data"][i, :, :, :]
                ctp = ds.variables["cloud_top_pressure"][i, :, 0]

                for j in range(-(self.dn // self.ddn) * self.ddn,
                               self.dn,  self.ddn):
                    i_s, i_e = dn0 + - self.dn, dn0 + self.dn + 1
                    j_s, j_e = dn0 - self.dn + j, dn0 + self.dn + 1 + j
                    m_lats = ds.variables["modis_lats"][i, i_s : i_e, j_s : j_e]
                    m_lons = ds.variables["modis_lons"][i, i_s : i_e, j_s : j_e]
                    m_lons *= np.cos(np.radians(m_lats))
                    m_tree = cKDTree(np.hstack([m_lats.reshape(-1, 1),
                                                m_lons.reshape(-1, 1)]),
                                     copy_data = True)
                    c_lats = ds.variables["caliop_lats"][i, i_s : i_e]
                    c_lons = ds.variables["caliop_lons"][i, i_s : i_e]
                    c_lons *= np.cos(np.radians(c_lats))

                    print(i_s, i_e)
                    print(j_s, j_e)
                    self.x[fi, :, :, :] = m_data[:, i_s : i_e, j_s : j_e]

                    _, inds = c_tree.query(np.hstack([m_lats.reshape(-1, 1),
                                                      m_lons.reshape(-1, 1)]))
                    shape = (2 * self.dn + 1,) * 2
                    y = np.zeros(shape)


                    y = np.reshape(ctp[:][inds], shape)
                    print(inds)

                    cm = np.zeros(shape)
                    cm[y > 0] = 1.0

                    _, inds = m_tree.query(np.hstack([c_lats.reshape(-1, 1),
                                                      c_lons.reshape(-1, 1)]))
                    vm = np.zeros(shape[0] * shape[1])
                    vm[inds] = 1.0
                    vm = np.reshape(vm, shape)

                    self.y[fi, :, :] =  y
                    self.cm[fi, :, :] = cm
                    self.vm[fi, :, :] = vm
                    self.x_lats[fi, :, :] = m_lats
                    self.x_lons[fi, :, :] = m_lons
                    self.y_lats[fi, :] = c_lats
                    self.y_lons[fi, :] = c_lons

                    fi += 1
            ds.close()

    def close(self):
        """
        Close the output file. This must be called to filnalize the NetCDF
        file containing the training data.
        """
        self.of.close()

cd = TrainingData(50)
cd.extract(20)
cd.close()
