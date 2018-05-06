import dardar
import modis
import numpy as np
import numpy.ma as ma
from mpl_toolkits.basemap import Basemap
from scipy import spatial
import matplotlib.pyplot as plt
import seaborn as sns
import geopy.distance
import cache
import pickle
import os
sns.reset_orig()

cache.cache_folder  = "/home/simonpf/Dendrite/UserAreas/Simon/cloud_collocations/satellite_data/"
collocation_folder = "/home/simonpf/Dendrite/UserAreas/Simon/cloud_collocations/collocations/"


class Collocation:
    def __init__(self, dardar_file, modis_files, name = ""):
        self.dardar_file = dardar_file
        self.modis_files = modis_files
        self.collocations = np.zeros((0, 4), dtype = int)
        self.distances = np.zeros(0)
        self.name = name

    def add(self, dardar_index,
            modis_file_index,
            collocation_index_0,
            collocation_index_1,
            distance):
        self.collocations = np.append(self.collocations,
                                      np.array([dardar_index,
                                                modis_file_index,
                                                collocation_index_0,
                                                collocation_index_1]).reshape(1, 4),
                                      axis = 0)
        self.distances = np.append(self.distances, distance)

    def get_lats(self):
        lats_dardar = np.zeros(0)
        lats_dardar = dardar_file.get_lats()[self.collocations[:, 0]]

        lats_modis  = np.zeros(0)
        for i, mf in enumerate(self.modis_files):
            indices = self.collocations[:, 1] == i
            indices_0 = self.collocations[indices, 2]
            indices_1 = self.collocations[indices, 3]
            lats_modis = np.append(lats_modis,
                                   mf.get_lats()[indices_0, indices_1],
                                   axis = 0)
        return lats_dardar, lats_modis

    def get_lons(self):
        lons_dardar = np.zeros(0)
        lons_dardar = dardar_file.get_lons()[self.collocations[:, 0]]

        lons_modis  = np.zeros(0)
        for i, mf in enumerate(self.modis_files):
            indices = self.collocations[:, 1] == i
            indices_0 = self.collocations[indices, 2]
            indices_1 = self.collocations[indices, 3]
            lons_modis = np.append(lons_modis,
                                   mf.get_lons()[indices_0, indices_1],
                                   axis = 0)
        return lons_dardar, lons_modis

    def save(self):
        parts = os.path.splitext(self.dardar_file.name)
        filename = os.path.join(collocation_folder, self.name, parts[0] + ".pckl")
        folder = os.path.dirname(filename)
        print("saving collocations: " + filename)

        if not os.path.exists(folder):
            os.makedirs(folder)

        self.modis_geo_product = self.modis_files[0].geo_product
        self.modis_files = [m.file.name for m in self.modis_files]

        self.dardar_product = self.dardar_file.product
        self.dardar_file = self.dardar_file.date

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def load(dardar_file):
        parts = os.path.splitext(dardar_file)
        filename = os.path.join(collocation_folder, parts[0] + ".pckl")
        f = open(filename, "rb")
        obj = pickle.load(f)

        print(obj.modis_files)
        obj.modis_files = [modis.ModisFile(m, obj.modis_geo_product) for m in obj.modis_files]
        obj.dardar_file = dardar.ICAREFile(obj.dardar_product, obj.dardar_file)
        return obj

        
    def get_modis_orbit(self, subsampling, bands):
        rads_modis = None
        lats_modis = None
        lons_modis = None

        ds = subsampling

        for mf in self.modis_files:
           print(mf.file)
           if rads_modis is None:
               rads = np.stack([mf.get_radiances(band = b)[::ds, ::ds]
                               for b in bands])
               lats = mf.get_lats()[::ds, ::ds]
               lons = mf.get_lons()[::ds, ::ds]
           else:
               rads_modis = np.append(rads_modis,
                                      [mf.get_radiances(band = b)[::ds, ::ds]
                                          for b in bands],
                                      axis = 0)
               lats_modis = np.append(lats_modis,
                                      mf.get_lats()[ ::ds, ::ds],
                                      axis = 0)
               lons_modis = np.append(lons_modis,
                                      mf.get_lons()[::ds, ::ds],
                                      axis = 0)

        valid = np.logical_not(np.any(np.isnan(lats_modis), axis = 1))
        lons = lons_modis[valid, :]
        lats = lats_modis[valid, :]
        rads = rads_modis[valid, :]

    def plot_modis_footprint(self, index = 0, ax = None, width = 400, subsampling = 4):

        if ax is None:
            ax = plt.gca()

        mf = self.modis_files
        rads_modis = None
        lats_modis = None
        lons_modis = None

        for mf in self.modis_files:
           print(mf.file)
           if rads_modis is None:
               rads = mf.get_radiances()
               m = rads.shape[1] // 2
               dn = width
               ddn = subsampling
               rads_modis = mf.get_radiances()[:, m - dn : m + dn : ddn]
               lats_modis = mf.get_lats()[:, m - dn : m + dn : ddn]
               lons_modis = mf.get_lons()[:, m - dn : m + dn : ddn]
           else:
               rads_modis = np.append(rads_modis,
                                      mf.get_radiances()[:, m - dn : m + dn : ddn],
                                      axis = 0)
               lats_modis = np.append(lats_modis,
                                      mf.get_lats()[:, m - dn : m + dn : ddn],
                                      axis = 0)
               lons_modis = np.append(lons_modis,
                                      mf.get_lons()[:, m - dn : m + dn : ddn],
                                      axis = 0)

        m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
                    llcrnrlon=-180,urcrnrlon=180,resolution='i', ax = ax)
        m.drawcoastlines(color = 'grey')
        m.drawparallels(np.arange(-90.,91.,30.))
        m.drawmeridians(np.arange(-180.,181.,60.))
        plt.title("MODIS Matches")

        valid = np.logical_not(np.any(np.isnan(lats_modis), axis = 1))
        lons = lons_modis[valid, :]
        lons2 = np.rad2deg(np.unwrap(np.deg2rad(lons)))
        lats = lats_modis[valid, :]
        rads = rads_modis[valid, :]

        lons = lons[::10, :]
        lats = lats[::10, :]
        rads = rads[::10, :]

        X, Y = m(lons, lats)
        d_lon_0 = np.abs(np.diff(lons, axis = 0)) > 180.0
        d_lon_0 = np.logical_or(d_lon_0[:, 1:], d_lon_0[:, :-1])

        d_lon_1 = np.abs(np.diff(lons, axis = 1)) > 180.0
        d_lon_1 = np.logical_or(d_lon_1[1:, :], d_lon_1[:-1, :])

        rads_masked = ma.masked_array(0.5 * (rads[:-1, :-1] + rads[1:, 1:]),  \
                                    d_lon_0 + d_lon_1)
        lons_masked = ma.masked_array(lons[:-1, :-1], d_lon_0 + d_lon_1)

        ax.pcolormesh(X, Y, rads_masked)

    def plot_dardar_footprint(self, ax = None, subsampling = 10):

        df = self.dardar_file
        lats = df.get_lats()
        lons = df.get_lons()

        m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
                    llcrnrlon=-180,urcrnrlon=180,resolution='i', ax = ax)
        m.scatter(lons[::subsampling],
                  lats[::subsampling], color = "crimson", marker = "x")

class Match:
    def __init__(self, year, day):
        self.modis_files = modis.get_files("MYD021KM", year, day)
        self.modis_times = [modis.filename_to_datetime(df) \
                             for df in self.modis_files]
        self.dardar_files = dardar.get_files("DARDAR_MASK", year, day)
        self.dardar_times = [dardar.filename_to_datetime(df) \
                             for df in self.dardar_files]

        print("Found " + str(len(self.modis_files)) + " MODIS files.")
        print("Found " + str(len(self.dardar_files)) + " DARDAR files.")

    def get_dardar_file(self, index = 0):
        return dardar.ICAREFile("DARDAR_MASK", self.dardar_times[index])

    def get_dardar_matches(self, index = 0):
        dardar_time = self.dardar_times[index]
        modis_time_deltas_1 = [(mt - dardar_time) for mt in self.modis_times]
        dt_days_1 = np.array([dt.days for dt in modis_time_deltas_1])

        if index < len(self.dardar_times) - 1:
            modis_time_deltas_2 = [(mt - self.dardar_times[index + 1]) \
                                   for mt in self.modis_times]
            dt_days_2 = np.array([dt.days for dt in modis_time_deltas_2])
        else:
            dt_days_2 = -1.0

        inds = np.where((dt_days_1 >= 0) * (dt_days_2 < 0))[0]

        if inds.size == 0:
            return []
        else:
            inds_start = np.maximum(inds[0] - 1, 0)
            inds_end = inds[-1]
            return [modis.ModisFile(mf, "MYD03") \
                    for mf in self.modis_files[inds_start : inds_end]]

    def get_collocations(self, index = 0, step = 42):
        modis_trees = []

        modis_files = self.get_dardar_matches(index)
        for mf in modis_files:
            print(mf.file)
            lats = mf.get_lats()
            lons = mf.get_lons()

            nx = lats.shape[1]
            m = nx // 2
            dn = 400
            lats = lats[:, m - dn : m + dn]
            lons = lons[:, m - dn : m + dn]


            modis_trees += [spatial.cKDTree(np.hstack([lats.reshape(-1, 1),
                                                      lons.reshape(-1, 1)]))]

        dardar_file = self.get_dardar_file(index)
        lats = dardar_file.get_lats()
        lons = dardar_file.get_lons()

        nps = lats.size

        index = step
        mf_index = 0

        collocations = Collocation(dardar_file, modis_files)

        dp = -1.0
        dd = 0.0

        while (index < nps - step):
            lat = lats[index]
            lon = lons[index]

            _, i = modis_trees[mf_index].query(np.array([[lat, lon]]))
            i_0 = i // (2 * dn)
            i_1 = m + i % (2 * dn) - dn

            d = geopy.distance.distance((lat, lon), modis_trees[mf_index].data[i, :])

            if (mf_index < len(modis_trees) - 1):
                _, i_2 = modis_trees[mf_index + 1].query(np.array([[lat, lon]]))
                i_0_2 = i_2 // (2 * dn)
                i_1_2 = m + i_2 % (2 * dn) - 10

                d_2 = geopy.distance.distance((lat, lon), modis_trees[mf_index + 1].data[i_2, :])
            else:
                d_2 = d

            print("collocation distance: " + str(d) + " / " + str(d_2))
            if d_2 < d:
                print("d (" + str(d) + " smaller than d_2 (" + str(d_2) + "). -> Next MODIS file.")
                mf_index += 1
            else:
                collocations.add(index, mf_index, i_0, i_1, d)
                index += step


            #if d > 1.0:
            #    if last_miss > 0.0:
            #        dd = d - last_miss
            #        if dd > step / 2.0:
            #            print("distance increasing ... moving to next MODIS orbit")
            #            if mf_index < len(modis_trees) - 1:
            #                mf_index += 1
            #            else:
            #                index = nps
            #        else:
            #            print("distance decreasing ...")
            #            index += step
            #    else:
            #        last_miss = d
            #else:
            #    collocations.add(index, mf_index, i_0, i_1, d)
            #    index += step
            #    last_miss = -1

        return collocations



def plot_collocations(lats_modis, lons_modis, rads_modis, cs):
    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
                llcrnrlon=-180,urcrnrlon=180,resolution='c')
    m.drawcoastlines(color = 'grey')
    m.drawparallels(np.arange(-90.,91.,30.))
    m.drawmeridians(np.arange(-180.,181.,60.))
    plt.title("MODIS Matches")

    valid = np.logical_not(np.any(np.isnan(lats_modis), axis = 1))
    lons = lons_modis[valid, :]
    lons2 = np.rad2deg(np.unwrap(np.deg2rad(lons)))
    lats = lats_modis[valid, :]
    rads = rads_modis[valid, :]

    lons = lons[::10, :]
    lats = lats[::10, :]
    rads = rads[::10, :]

    X, Y = m(lons, lats)
    d_lon_0 = np.abs(np.diff(lons, axis = 0)) > 180.0
    d_lon_0 = np.logical_or(d_lon_0[:, 1:], d_lon_0[:, :-1])

    d_lon_1 = np.abs(np.diff(lons, axis = 1)) > 180.0
    d_lon_1 = np.logical_or(d_lon_1[1:, :], d_lon_1[:-1, :])

    rads_masked = ma.masked_array(0.5 * (rads[:-1, :-1] + rads[1:, 1:]),  \
                                d_lon_0 + d_lon_1)
    lons_masked = ma.masked_array(lons[:-1, :-1], d_lon_0 + d_lon_1)

    plt.pcolormesh(X, Y, rads_masked)

    lats_cs_dardar, lats_cs_modis = cs.get_lats()
    lons_cs_dardar, lons_cs_modis = cs.get_lons()
    m.scatter(lons_cs_dardar, lats_cs_dardar, c = "C1", marker = "o")
    m.scatter(lons_cs_modis, lats_cs_modis, c = "C2", marker = "x")



#m = Match(2008, 2)
#for i in range(1, len(m.dardar_files)):
#    cs = m.get_collocations(i)
#    cs.save()

#rads_modis = None
#lats_modis = None
#lons_modis = None
#
#for mf in modis_files:
#    print(mf.file)
#    if rads_modis is None:
#        rads = mf.get_radiances()
#        m = rads.shape[1] // 2
#        dn = 400
#        rads_modis = mf.get_radiances()[:, m - dn : m + dn]
#        lats_modis = mf.get_lats()[:, m - dn : m + dn]
#        lons_modis = mf.get_lons()[:, m - dn : m + dn]
#    else:
#        rads_modis = np.append(rads_modis,
#                               mf.get_radiances()[:, m - dn : m + dn],
#                               axis = 0)
#        lats_modis = np.append(lats_modis,
#                               mf.get_lats()[:, m - dn : m + dn],
#                               axis = 0)
#        lons_modis = np.append(lons_modis,
#                               mf.get_lons()[:, m - dn : m + dn],
#                               axis = 0)
#
## llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
## are the lat/lon values of the lower left and upper right corners
## of the map.
## resolution = 'c' means use crude resolution coastlines.
#plt.figure()
#plot_collocations(lats_modis, lons_modis, rads_modis, cs)
#
#rads_modis = None
#lats_modis = None
#lons_modis = None
#
#for mf in modis_files:
#    print(mf.file)
#    if rads_modis is None:
#        rads = mf.get_radiances()
#        m = rads.shape[1] // 2
#        dn = 400
#        rads_modis = mf.get_radiances()[:, m - dn : m + dn]
#        lats_modis = mf.get_lats()[:, m - dn : m + dn]
#        lons_modis = mf.get_lons()[:, m - dn : m + dn]
#    else:
#        rads_modis = np.append(rads_modis,
#                               mf.get_radiances(band = 10)[:, m - dn : m + dn],
#                               axis = 0)
#        lats_modis = np.append(lats_modis,
#                               mf.get_lats()[:, m - dn : m + dn],
#                               axis = 0)
#        lons_modis = np.append(lons_modis,
#                               mf.get_lons()[:, m - dn : m + dn],
#                               axis = 0)
#
## llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
## are the lat/lon values of the lower left and upper right corners
## of the map.
## resolution = 'c' means use crude resolution coastlines.
#plt.figure()
#plot_collocations(lats_modis, lons_modis, rads_modis, cs)
#
#rads_modis = None
#lats_modis = None
#lons_modis = None
#
#for mf in modis_files:
#    print(mf.file)
#    if rads_modis is None:
#        rads = mf.get_radiances()
#        m = rads.shape[1] // 2
#        dn = 400
#        rads_modis = mf.get_radiances()[:, m - dn : m + dn]
#        lats_modis = mf.get_lats()[:, m - dn : m + dn]
#        lons_modis = mf.get_lons()[:, m - dn : m + dn]
#    else:
#        rads_modis = np.append(rads_modis,
#                               mf.get_radiances(band = 26)[:, m - dn : m + dn],
#                               axis = 0)
#        lats_modis = np.append(lats_modis,
#                               mf.get_lats()[:, m - dn : m + dn],
#                               axis = 0)
#        lons_modis = np.append(lons_modis,
#                               mf.get_lons()[:, m - dn : m + dn],
#                               axis = 0)

# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
# are the lat/lon values of the lower left and upper right corners
# of the map.
# resolution = 'c' means use crude resolution coastlines.
#plt.figure()
#plot_collocations(lats_modis, lons_modis, rads_modis, cs)
