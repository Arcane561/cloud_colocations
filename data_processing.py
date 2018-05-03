import glob
import os
import numpy as np
from netCDF4 import Dataset

def extract_cloud_types(ds):

    if not type(ds) == Dataset:
        ds = Dataset(ds)

    root = ds.groups["output"]
    cs = root.variables["cloud_scenario"]
    print(cs.shape)
    n = cs.shape[0]
    y = np.zeros((n, 9))

    bins = np.arange(10) + 0.5

    for i in range(n):
        h, _ = np.histogram(cs[i, :, :], bins = bins)
        if h.sum() == 0.0:
            y[i, 0] = 1.0
        else:
            y[i, 1 + np.argmax(h)] = 1.0
    return y

def extract_5_by_5_neighborhood(ds):

    if not type(ds) == Dataset:
        ds = Dataset(ds)

    root = ds.groups["input"]
    modis = np.array(root.variables["modis"])
    x = modis[:-1, :, 10 - 2 : 10 + 3, 10 - 2 : 10 + 3]

    return x

def whiten(x):
    return (x - np.mean(x, axis = (0, 2, 3)).reshape(1, -1, 1, 1)) / np.std(x, axis = (0, 2, 3)).reshape(1, -1, 1, 1)

def preprocess(x):
    for i in range(x.shape[1]):
        inds = np.isnan(x[:, i, :, :])
        x[:, i, :, :][inds] = np.mean(x[:, i, :, :])
    return whiten(x)

def combine_training_files(training_data, new_file):
    files = glob.glob(os.path.join(training_data, "cloud_collocations_10_*.nc"))

    neighborhood = 10
    root = Dataset(new_file, "w")
    input  = root.createGroup("input")
    output = root.createGroup("output")
    input.createDimension("channels", 36)
    input.createDimension("ao", neighborhood * 2 + 1)
    input.createDimension("xo", neighborhood * 2 + 1)
    input.createDimension("samples", None)

    v_modis = input.createVariable("modis", "f4", ("samples", "channels", "ao", "xo"))
    v_lats  = input.createVariable("lats", "f4", ("samples", "ao", "xo"))
    v_lons  = input.createVariable("lons", "f4", ("samples", "ao", "xo"))

    output.createDimension("samples", None)
    output.createDimension("ao", neighborhood * 2 + 1)
    output.createDimension("z", 22)

    v_cth = output.createVariable("cth", "f4", ("samples"))
    v_cc  = output.createVariable("cloud_scenario", "f4", ("samples", "ao", "z"))
    n = 0

    for f in files:
        print(f)
        n0 = v_modis.shape[0]
        print(n0)

        root_2 = Dataset(f, "r")

        v = root_2.groups["input"].variables["modis"]
        n1 = v.shape[0]
        print(n0, n1)

        v_modis[n0 : n1, :, :, :] = v[:, :, :, :]

        v = root_2.groups["input"].variables["lats"]
        v_lats[n0 : n1, :, :] = v[:, :, :]

        v = root_2.groups["input"].variables["lons"]
        v_lons[n0 : n1, :, :] = v[:, :, :]

        v = root_2.groups["output"].variables["cth"]
        v_cth[n0 : n1] = v[:]

        v = root_2.groups["output"].variables["cloud_scenario"]
        v_cc[n0 : n1, :, :] = v[:, :, :]

        root_2.close()
    root.close()


