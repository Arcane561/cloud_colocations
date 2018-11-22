import os
import numpy as np

from netCDF4 import Dataset

def subsample_classes(classes):
    bins = np.arange(12) - 0.5
    cs, bins = np.histogram(classes, bins = bins)
    i = np.argmax(cs)
    return i


class TrainingDataFile:
    def __init__(self, path, bands, dn, ss = None, overwrite = False):
        self.path  = path
        self.bands = bands
        self.ss    = ss
        self.dn    = dn

        if self.ss is None:
            filename = "training_data_" + str(self.dn) + ".nc"
        else:
            filename = "training_data_ss_" + str(self.ss) + "_" + str(self.dn) + ".nc"

        self.filename = os.path.join(self.path, filename)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        print("loading ", self.filename)
        # Create new file.
        if not os.path.isfile(self.filename) or overwrite:

            root = Dataset(self.filename, "w")
            self.root = root
            self.dims = (root.createDimension("samples", None),
                         root.createDimension("bands", len(self.bands)),
                         root.createDimension("along_track", 2 * self.dn + 1),
                         root.createDimension("across_track", 2 * self.dn + 1))

            dims = ("samples", "bands", "along_track", "across_track")
            self.x = root.createVariable("input", "f4", dims)
            dims = ("samples",)
            self.cth = root.createVariable("cth", "f4", dims)
            self.ctp = root.createVariable("ctp", "f4", dims)
            self.cloud_class = root.createVariable("cloud_class", "i4", dims)

            self.sample_index = 0

        # Read existing file.
        else:
            root = Dataset(self.filename, "r")

            self.x = root.variables["input"]
            self.cth = root.variables["cth"]
            self.ctp = root.variables["ctp"]
            self.cloud_class = root.variables["cloud_class"]

            self.sample_index = self.ctp.shape[0]

    def __del__(self):
        if hasattr(self, "root"):
            self.root.close()

    def add_sample(self, input, cth, ctp, cloud_class):

        self.x[self.sample_index, :, :, :] = input
        self.cth[self.sample_index] = cth
        self.ctp[self.sample_index] = ctp
        self.cloud_class[self.sample_index] = cloud_class

        self.sample_index += 1
