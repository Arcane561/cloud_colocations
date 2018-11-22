from cloud_colocations.raw_data      import RawData
from cloud_colocations.training_data import TrainingDataFile, subsample_classes

import glob
import os

import numpy as np

result_path = "/home/simon/src/cloud_colocations/data"
data_path   = "/home/simon/src/cloud_colocations"
dn = 20

training_data       = TrainingDataFile(result_path, [30, 31], dn)
training_data_ss_5  = TrainingDataFile(result_path, [30, 31], dn, ss = 5)
training_data_ss_11 = TrainingDataFile(result_path, [30, 31], dn, ss = 11)

year = 2009
days = glob.glob(os.path.join(data_path, str(year), "*"))
for d in days:

    day = int(os.path.basename(d))
    print("processing day: ", d)

    data = RawData(year, day, dn, basepath = data_path)

    #
    # Original data
    #

    band_31 = data.modis_data.variables["band_31"][:, :, :]
    band_32 = data.modis_data.variables["band_32"][:, :, :]
    cth = data.caliop_data.variables["cth"][:, :]
    ctp = data.caliop_data.variables["ctp"][:, :]
    cloud_class = data.caliop_data.variables["cloud_class"][:, :]

    dn = (cloud_class.shape[1] - 1) // 2

    for i in range(band_31.shape[0]):
        x = np.stack([band_31[i, :, :],
                      band_32[i, :, :]])
        training_data.add_sample(x, cth[i], ctp[i], cloud_class[i])

    #
    # Subsampled 5
    #

    band_31 = data.modis_ss_5_data.variables["band_31"][:, :, :]
    band_32 = data.modis_ss_5_data.variables["band_32"][:, :, :]
    cth = data.caliop_ss_5_data.variables["cth"][:, :]
    ctp = data.caliop_ss_5_data.variables["ctp"][:, :]
    cloud_class = data.caliop_ss_5_data.variables["cloud_class"][:]

    for i in range(band_31.shape[0]):
        x = np.stack([band_31[i, :, :],
                      band_32[i, :, :]])
        cth_ = cth[i, 5 * dn]
        ctp_ = ctp[i, 5 * dn]
        cloud_class_ = subsample_classes(cloud_class[i, 5 * dn - 5 : 5 * dn + 6])
        training_data_ss_5.add_sample(x, cth_, ctp_, cloud_class_)

    #
    # Subsampled 11
    #

    band_31 = data.modis_ss_11_data.variables["band_31"][:, :, :]
    band_32 = data.modis_ss_11_data.variables["band_32"][:, :, :]
    cth = data.caliop_ss_11_data.variables["cth"][:, :]
    ctp = data.caliop_ss_11_data.variables["ctp"][:, :]
    cloud_class = data.caliop_ss_11_data.variables["cloud_class"][:]

    for i in range(band_31.shape[0]):
        x = np.stack([band_31[i, :, :],
                      band_32[i, :, :]])
        cth_ = cth[i, 11 * dn]
        ctp_ = ctp[i, 11 * dn]
        cloud_class_ = subsample_classes(cloud_class[i, 11 * dn - 11 : 11 * dn + 12])
        training_data_ss_11.add_sample(x, cth_, ctp_, cloud_class_)



