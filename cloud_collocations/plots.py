import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import Normalize

def grid_to_edges(grid):
    new_grid = np.zeros((grid.shape[0]+ 1, grid.shape[1] + 1))
    new_grid[1:-1, 1:-1] = 0.25 * (grid[1:, 1:] + grid[1:, :-1] +
                                   grid[:-1, 1:] + grid[:-1, :-1])

    new_grid[0, 1:-1] = (grid[0, :-1] + grid[0, 1:]) - new_grid[1, 1:-1]
    new_grid[-1, 1:-1] = (grid[-1, :-1] + grid[-1, 1:]) - new_grid[-2, 1:-1]
    new_grid[1:-1, 0] = (grid[1:, 0] + grid[:-1, 0]) - new_grid[1:-1, 1]
    new_grid[1:-1, -1] = (grid[:-1, -1] + grid[1:, -1]) - new_grid[1:-1, -2]

    new_grid[0, 0] = 0.5 * (new_grid[0, 1] + new_grid[1, 0])
    new_grid[-1, 0] = 0.5 * (new_grid[-1, 1] + new_grid[-2, 0])
    new_grid[0, -1] = 0.5 * (new_grid[1, -1] + new_grid[0, -2])
    new_grid[-1, -1] = 0.5 * (new_grid[-1, -2] + new_grid[-2, -1])

    new_grid[0, 0]   = new_grid[1, 0] + new_grid[0, 1] - new_grid[1, 1]
    new_grid[0, -1]  = new_grid[0, -2] + new_grid[1, -1] - new_grid[1, -2]
    new_grid[-1, 0]  = new_grid[-2, 0] + new_grid[-1, 1] - new_grid[-2, 1]
    new_grid[-1, -1]  = new_grid[-2, -1] + new_grid[-1, -2] - new_grid[-2, -2]

    return new_grid

def plot_composite(modis_data, i, bands = [1, 4, 3], ax = None):

    if ax is None:
        ax = plt.gca()

    x = np.zeros(modis_data["band_1"].shape[1:] + (3,))

    for j, b in enumerate(bands):
        band_name = "band_" + str(b)
        xx = modis_data[band_name][i, :, :]
        x_max = xx.max()
        x_min = xx.min()
        x[:, :, j] = (xx - x_min) / (x_max - x_min)

    lats = modis_data["lats"][i, :, :]
    lons = modis_data["lons"][i, :, :]

    lat_grid = grid_to_edges(lats)
    lon_grid = grid_to_edges(lons)

    cs = np.ones((lats.size, 3))
    for j in range(3):
        cs[:, j] = x[:, :, j].ravel()
    print(cs[0, :])

    m = np.tile(np.arange(x.shape[0]), (x.shape[1],1))
    m = np.ones(x[:, :, 0].shape)
    img = ax.pcolormesh(lon_grid, lat_grid, x[:, :, 0], facecolors = cs, edgecolor = None)
    img.set_array(None)

    return img

def plot_modis_granule_composite(modis_file,
                                 modis_geo_file,
                                 bands = [1, 4, 3], ax = None):
    if ax is None:
        ax = plt.gca()

    x  = modis_file.data[bands, :, :]
    xx = np.zeros(modis_file.data.shape[1:] + (3,))

    for j in range(len(bands)):
        x_max = np.nanmax(x[j, :, :])
        x_min = np.nanmin(x[j, :, :])
        xx[:, :, j] = (x[j, :, :] - x_min) / (x_max - x_min)

    lats = modis_geo_file.get_latitudes()
    lons = modis_geo_file.get_longitudes()

    lat_grid = grid_to_edges(lats)
    lon_grid = grid_to_edges(lons)

    cs = np.ones((lats.size, 3))
    for j in range(3):
        cs[:, j] = xx[:, :, j].ravel()
    print(cs[0, :])

    m = np.tile(np.arange(x.shape[0]), (x.shape[1],1))
    m = np.ones(x[:, :, 0].shape)
    img = ax.pcolormesh(lon_grid, lat_grid, xx[:, :, 0], facecolors = cs, edgecolor = None)
    img.set_array(None)

    return img

def plot_swath_cth(caliop_data, i, ax = None):
    if ax is None:
        ax = plt.gca()

    lats = caliop_data["lats"][i, :]
    lons = caliop_data["lons"][i, :]

    norm = Normalize(vmin = 0.0, vmax = 20.0)

    cth = caliop_data["cth"][i, :]

    ax.scatter(lons, lats, c = cth, cmap = "plasma", norm = norm)

def plot_swath_cloud_mask(caliop_data, i, ax = None):
    if ax is None:
        ax = plt.gca()

    lats = caliop_data["lats"][i, :]
    lons = caliop_data["lons"][i, :]

    norm = Normalize(vmin = 0.0, vmax = 1.0)

    cloud = caliop_data["feature_class"][i, :] == 2

    ax.scatter(lons, lats, c = cloud, cmap = "plasma", norm = norm)

