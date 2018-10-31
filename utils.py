import os
import numpy as np

def ensure_extension(name, ext):
    if ext[0] != ".":
        ext = "." + ext
    name_base, name_ext = os.path.splitext(name)
    if name_ext != ext:
        return name + ext
    else:
        return name

def grid_to_edges_2d(grid):
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

#                            + (2.0 * new_grid[1, 0] - new_grid[2, 0]))
#    new_grid[0, -1] = 0.5 * ((2.0 * new_grid[0, -2] - new_grid[0, -3])
#                            + (2.0 * new_grid[-1, 1] - new_grid[-1, 2]))
#    new_grid[-1, 0] = 0.5 * ((2.0 * new_grid[-2, 0] - new_grid[-3, 0])
#                            + (2.0 * new_grid[-1, 1] - new_grid[-1, 2]))
#    new_grid[-1, -1] = 0.5 * ((2.0 * new_grid[-2, -1] - new_grid[-3, -1])
#                            + (2.0 * new_grid[-1, 1] - new_grid[-1, 2]))

    return new_grid



def reshape_and_normalize(x):
        x = np.array(np.transpose(x[0, :, :, :], (1, 2, 0)))
        x_min = np.min(x, axis = (0, 1), keepdims = True)
        x_max = np.max(x, axis = (0, 1), keepdims = True)
        x = (x - x_min) / (x_max - x_min)
        return x
