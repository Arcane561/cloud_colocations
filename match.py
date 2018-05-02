import dardar
import modis
import numpy as np
import numpy.ma as ma
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

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
        return dardar.ICAREFile("DARDAR_MASK", self.dardar_times[0])

    def get_dardar_matches(self, index = 0):
        dardar_time = self.dardar_times[index]
        modis_time_deltas_1 = [(mt - dardar_time) for mt in self.modis_times]
        dt_days_1 = np.array([dt.days for dt in modis_time_deltas_1])

        if index < len(self.dardar_times):
            modis_time_deltas_2 = [(mt - self.dardar_times[index + 1]) \
                                   for mt in self.modis_times]
            dt_days_2 = np.array([dt.days for dt in modis_time_deltas_2])
        else:
            dt_days_2 = -1.0

        inds = np.where((dt_days_1 >= 0) * (dt_days_2 < 0))[0]

        print(inds)
        inds_start = np.maximum(inds[0] - 1, 0)
        inds_end = inds[-1]
        return [modis.ModisFile(mf, "MYD03") \
                for mf in self.modis_files[inds_start : inds_end]]



m = Match(2007, 1)
modis_files = m.get_dardar_matches(0)
dardar_file = m.get_dardar_file(0)


rads_modis = None
lats_modis = None
lons_modis = None

for mf in modis_files:
    print(mf.file)
    if rads_modis is None:
        rads_modis = mf.get_radiances()
        lats_modis = mf.get_lats()
        lons_modis = mf.get_lons()
    else:
        rads_modis = np.append(rads_modis, mf.get_radiances(), axis = 0)
        lats_modis = np.append(lats_modis, mf.get_lats(), axis = 0)
        lons_modis = np.append(lons_modis, mf.get_lons(), axis = 0)

# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
# are the lat/lon values of the lower left and upper right corners
# of the map.
# resolution = 'c' means use crude resolution coastlines.
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

#bounds = np.where(np.max(np.abs(np.diff(lons2, axis = 0)), axis = 1) > 180.0)[0]
#l_b = 0
#for b in bounds:
#    m.pcolormesh(lons2[l_b:b:10], lats[l_b:b:10], rads[l_b:b:10, :], latlon = True)
#    l_b = b + 1

plt.pcolormesh(X, Y, rads_masked)

lats_dardar = dardar_file.get_lats()
lons_dardar = dardar_file.get_lons()
m.scatter(lons_dardar, lats_dardar)
plt.show()

