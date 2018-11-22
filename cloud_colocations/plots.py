import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from skimage import exposure

from matplotlib.colors import Normalize

#
# Ugly path for matplotlib 3.0.0
#

from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

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

def plot_composite(modis_data, i, bands = [1, 4, 3], figure = None, subplot = (1,)):


    if ax is None:
        ax = plt.gca()

    lats = modis_data["lats"][i, :, :]
    lons = modis_data["lons"][i, :, :]


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

    m = np.tile(np.arange(x.shape[0]), (x.shape[1],1))
    m = np.ones(x[:, :, 0].shape)
    img = ax.pcolormesh(lon_grid, lat_grid, x[:, :, 0], facecolors = cs, edgecolor = None)
    img.set_array(None)

    return img

def plot_modis_granule_composite(modis_file,
                                 modis_geo_file,
                                 bands = [1, 4, 3],
                                 figure = None,
                                 grid_spec = None):
    if grid_spec is None:
        ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree())
    else:
        ax = plt.subplot(grid_spec, projection = ccrs.PlateCarree())

    lats = modis_geo_file.get_latitudes()
    lons = modis_geo_file.get_longitudes()

    lon_min = np.min(lons)
    lon_max = np.max(lons)
    lat_min = np.min(lats)
    lat_max = np.max(lats)

    x  = modis_file.data[[b - 1 for b in bands], :, :]
    xx = np.zeros(modis_file.data.shape[1:] + (3,))

    for j in range(len(bands)):
        x_max = np.nanmax(x[j, :, :])
        x_min = np.nanmin(x[j, :, :])
        xx[:, :, j] = (x[j, :, :] - x_min) / (x_max - x_min)

    x = exposure.adjust_gamma(x, 0.01)

    lat_grid = grid_to_edges(lats)
    lon_grid = grid_to_edges(lons)

    cs = np.ones((lats.size, 3))
    for j in range(3):
        cs[:, j] = xx[:, :, j].ravel()

    m = np.tile(np.arange(x.shape[0]), (x.shape[1],1))
    m = np.ones(x[:, :, 0].shape)
    img = ax.pcolormesh(lon_grid, lat_grid, xx[:, :, 0], facecolors = cs, edgecolor = None)
    img.set_array(None)

    ax.coastlines(lw = 1, color = "black", resolution = "110m")

    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(True)

    ax.spines['right'].set_color("grey")
    ax.spines['left'].set_color("grey")
    ax.spines['bottom'].set_color("grey")
    ax.spines['top'].set_color("grey")

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth = 0)
    gl.xlabels_top   = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.xformatter = LONGITUDE_FORMATTER
    gl.ylabels_right = False
    #gl.xlabel_style = {'size': 15, 'color': 'gray'}
    #gl.xlabel_style = {'color': 'red', 'weight': 'bold'}

    return ax, img

def plot_scalar_field(x, lats, lons, figure = None,
                      norm = None,
                      grid_spec = None):

    if grid_spec is None:
        ax = plt.subplot(1, 1, 1)
    else:
        ax = plt.subplot(grid_spec, projection = ccrs.PlateCarree())


    if norm is None:
        norm = Normalize()

    lat_grid = grid_to_edges(lats)
    lon_grid = grid_to_edges(lons)

    img = ax.pcolormesh(lon_grid, lat_grid, x, edgecolor = None, norm = norm)

    ax.coastlines(lw = 1, color = "black", resolution = "110m")

    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(True)

    ax.spines['right'].set_color("grey")
    ax.spines['left'].set_color("grey")
    ax.spines['bottom'].set_color("grey")
    ax.spines['top'].set_color("grey")

    return ax, img

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

