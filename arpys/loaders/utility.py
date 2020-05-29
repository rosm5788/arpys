import numpy as np
from scipy.optimize import minimize

def make_filename_list(imin, imax, prefix, postfix, num_figs):
    format_str = '{0:0>' + str(num_figs) + 'd}'
    return [prefix + format_str.format(i) + postfix for i in range(imin, imax + 1)]

def find_basis(x, y, threshold = 0.005):
    """
    Find a regular grid approximation to the input 2D grid with coordinates given by x, y
    Threshold is the smallest step size possible for delta or min (5 um here)
    Guess the offset from the min
    Guess the delta by picking unique values and taking the average distance
    Guess the number of points by taking (max - min)/delta + 1
    Optimize the min coordinate using an optimizer
    """
    x = np.array(x)
    y = np.array(y)
    # find duplicates
    pairs = np.array([x, y]).T
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    dx = np.round(np.mean(np.diff(x_unique))/threshold)*threshold
    dy = np.round(np.mean(np.diff(y_unique))/threshold)*threshold
    x_min = np.min(x)
    y_min = np.min(y)
    x_max = np.max(x)
    y_max = np.max(y)
    Nx = int(np.round((x_max - x_min)/dx)) + 1
    Ny = int(np.round((y_max - y_min)/dy)) + 1
    if Nx*Ny != x.size or Nx*Ny != y.size:
        print('Guessed {0} number of points, but raster only has {1} points'.format(Nx*Ny, x.size))
    def basis_gen(vals, dx, dy, Nx, Ny, x_target, y_target):
        x_gen = vals[0] + dx*np.arange(Nx)
        y_gen = vals[1] + dy*np.arange(Ny)
        X, Y = np.meshgrid(x_gen, y_gen)
        X = X.flatten()
        Y = Y.flatten()
        return np.sum((X - x_target)**2) + np.sum((Y - y_target)**2)
    res = minimize(basis_gen, np.array([x_min, y_min]), args=(dx, dy, Nx, Ny, x, y))
    x_min, y_min = tuple(res.x)
    x_min = np.round(x_min/threshold)*threshold
    dx = np.round(dx/threshold)*threshold
    Nx = int(np.round(Nx))
    y_min = np.round(y_min/threshold)*threshold
    dy = np.round(dy/threshold)*threshold
    Ny = int(np.round(Ny))
    return x_min, dx, Nx, y_min, dy, Ny
