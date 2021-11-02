import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from pyimagetool import ImageTool
import time

def requires_ef(func):
    def func_wrapper(*args, **kwargs):
        if getattr(args[0], 'ef') is None:
            raise AttributeError("E_F is not defined yet.")
        return func(*args, **kwargs)

    return func_wrapper


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Source: https://stackoverflow.com/a/29042041
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))
    compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


def fermi_fcn_bg(x, ef, kBT, y0, *coeffs):
    """
    Fermi function with a polynomial background term.
    coeffs example:
    coeffs[0]*x**2 + coeffs[1]*x + coeffs[2]
    this notation is consistent with that provided by numpy.polyfit
    """
    p = np.array(coeffs)
    dos = (p[np.newaxis, :] * x[:, np.newaxis] ** (np.arange(len(p))[::-1])).sum(axis=1)
    return dos / (np.exp((x - ef) / kBT) + 1) + y0


def fermi_fcn_linear_bg(x, ef, kBT, y0, a, b):
    return (a + b * x) / (np.exp((x - ef) / kBT) + 1) + y0
# This is wrong
def phi2k(phi, ke):
    return 0.512 * np.sqrt(ke) * np.sin(np.radians(phi))
# The is wrong
def k2phi(k, ke):
    return np.degrees(np.arcsin(k/(0.512 * np.sqrt(ke))))


# slit angle is alpha, phi0 is offset to normal emission, theta0 is perpendicular angle (distance to normal emission)
# slit orientation values 0=vertical slit, 1=horizontal slit, 2= deflectors
# Transforms angle along the slit to k along the slit
def alpha2k(alpha, ke, theta0=0, phi0=0, slit_orientation=0):
    # Vertical Slit
    if slit_orientation == 0:
        k = -1 * 0.512 * np.sqrt(ke) * (np.sin((np.pi / 180) * (alpha + phi0)) / 2.0 + np.sin((np.pi / 180)
                                        * (alpha - phi0)) / 2.0 - np.cos((np.pi / 180) * theta0) *
                                        (np.sin((np.pi / 180) * (alpha + phi0)) / 2.0 -
                                        np.sin((np.pi / 180) * (alpha - phi0)) / 2.0))
        return k
    # Horizontal Slit
    elif slit_orientation == 1:
        k = 0.512 * np.sqrt(ke) * np.sin((np.pi / 180) * (alpha - phi0))
        return k
    # Deflectors
    elif slit_orientation == 2:
        k = 0.512 * np.sqrt(ke) * ((alpha - phi0) / (np.sqrt((alpha - phi0) ** 2 + theta0**2))) * \
             np.sin((np.pi / 180) * np.sqrt((alpha - phi0) ** 2 + theta0**2))
        return k
    else:
        print('big error my guy')


# Finds the associated ky from beta angle (angle across the slit) and alpha angle, theta0 is perpendicular
# angle offset to normal
# emission, phi0 is the offset to normal emission parallel to the slit.
# slit orientation values 0=vertical slit, 1=horizontal slit, 2=deflectors
def beta2k(beta, alpha, ke, theta0=0, phi0=0, slit_orientation=0):
    # Vertical Slit
    if slit_orientation == 0:
        k = -1 * 0.512 * np.sqrt(ke) * np.sin(np.radians(beta-theta0)) * np.cos(np.radians(alpha-phi0))
        return k
    if slit_orientation == 1:
        k = 0.512 * np.sqrt(ke) * np.sin(np.radians(beta-theta0)) * np.cos(np.radians(alpha-phi0))
        return k
    if slit_orientation == 2:
        k = -1 * 0.512 * np.sqrt(ke) * ((beta - phi0) / np.sqrt((alpha - phi0)**2 + (beta - theta0)**2)) * \
            np.sin(np.radians(np.sqrt((alpha-phi0)**2 + (beta-theta0)**2)))


# "Inverse Mapping Functions" from Ishida, Shin paper. Calculates angular sector from momentum sector.
# Calculates "kz" from kx, ky and energy. This is kz momentum of electron, not kz in the crystal (No inner potential)
# alpha is the angle along the slit, kx is momentum along the slit, ky is momentum across the slit
# phi0 is the angular offset to normal emission along the slit
def k2alpha(kx, ky, ke, phi0=0, slit_orientation=0):
    k = 0.512 * np.sqrt(ke)
    kz = np.sqrt(k**2 - kx**2 - ky**2)
    if slit_orientation == 0:
        alpha = np.degrees(np.arcsin((-np.cos(np.radians(phi0))*kx + np.sin(np.radians(phi0))*kz)/k))
        return alpha
    if slit_orientation == 1:
        alpha = np.degrees(np.arcsin((np.sin(np.radians(phi0))*np.sqrt(k**2 - kx**2) + np.cos(phi0)*kx)/k))
        return alpha
    elif slit_orientation ==2:
        alpha = (kx / np.sqrt(k**2 - kz**2)) * np.degrees(np.arccos(kz/k))
        return alpha
    else:
        print("what's the slit orientation?")


def k2beta(kx, ky, ke, phi0=0, slit_orientation=0):
    k = 0.512 * np.sqrt(ke)
    kz = np.sqrt(k**2 - kx**2 - ky**2)
    if slit_orientation == 0:
        beta = np.degrees(np.arctan((-1*ky)/(np.sin(np.radians(phi0)) * ky + np.cos(np.radians(phi0))*kz)))
        return beta
    if slit_orientation == 1:
        beta = np.degrees(np.arctan(ky/kz))
        return beta
    if slit_orientation == 2:
        beta = (-1 * ky / np.sqrt(k**2 - kz**2)) * np.degrees(np.arccos(kz/k))
        return beta
@xr.register_dataarray_accessor("arpes")
class Arpes:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.is_kinetic = True
        self.ef = None
        self.it = None

    # These k conversion equations are the "forward" k-conversions from angle to k-space This can be used with
    # the irregularly spaced k-conversion codes directly. K-space conversion depends on the orientation of the slit
    # so the user must select which orientation the experiment was using. This could in practice be automated
    # by setting an attribute in the associated beamline loader, but beamlines like MAESTRO can use both slit
    # orientations, so it's up to the user to set correctly.
    # slit_orientation values 0=vertical slit, 1=horizontal slit, 2=deflector+vertical slit,
    # 3=deflector+horizontal slit, phi0 = tilt offset along slit, theta0 = perpendicular offset across slit
    @staticmethod
    def forward_k_conversion(ke, alpha, beta, phi0=0, theta0=0, azimuth=0, slit_orientation=0):
        # Vertical Slit K Conversion, kx along slit
        if slit_orientation == 0:
            kx = 0.512 * np.sqrt(ke) * ((np.sin(np.radians(azimuth))*np.sin(np.radians(beta-theta0)) +
                                         np.cos(np.radians(azimuth))*np.sin(np.radians(phi0)) *
                                         np.cos(np.radians(beta-theta0)))*np.cos(np.radians(alpha)) -
                                        np.cos(np.radians(azimuth))*np.cos(np.radians(phi0)) *
                                        np.sin(np.radians(alpha)))

            ky = 0.512 * np.sqrt(ke) * ((-1*np.cos(np.radians(azimuth)) * np.sin(np.radians(beta-theta0)) +
                                         np.sin(np.radians(azimuth))*np.sin(np.radians(phi0)) *
                                         np.cos(np.radians(beta-theta0)))*np.cos(np.radians(alpha)) -
                                        np.sin(np.radians(azimuth))*np.cos(np.radians(phi0)) *
                                        np.sin(np.radians(alpha)))
        # Horizontal Slit K Conversion, kx along slit (swap kx and ky from Ishida paper)
        elif slit_orientation == 1:
            kx = 0.512 * np.sqrt(ke) * ((-1*np.cos(np.radians(azimuth))*np.sin(np.radians(phi0)) +
                                         np.sin(np.radians(azimuth))*np.sin(np.radians(beta-theta0)) *
                                         np.cos(np.radians(phi0))) * np.cos(np.radians(alpha)) +
                                        (np.cos(np.radians(azimuth)) * np.cos(np.radians(phi0)) +
                                         np.sin(np.radians(azimuth)) * np.sin(np.radians(beta-theta0)) *
                                         np.sin(np.radians(phi0))) * np.sin(np.radians(alpha)))
            ky = 0.512 * np.sqrt(ke) * ((np.sin(np.radians(azimuth)) * np.sin(np.radians(phi0)) +
                                         np.cos(np.radians(azimuth))*np.sin(np.radians(beta-theta0)) *
                                         np.cos(np.radians(phi0))) * np.cos(np.radians(alpha)) -
                                        (np.sin(np.radians(azimuth)) * np.cos(np.radians(phi0)) -
                                         np.cos(np.radians(azimuth)) * np.sin(np.radians(beta-theta0)) *
                                         np.sin(np.radians(phi0))) * np.sin(np.radians(alpha)))
        # Vertical Slit Deflector K Conversion, kx along slit
        elif slit_orientation == 2:
            kx = 0.512 * np.sqrt(ke) * ((-1 * np.radians(alpha) * np.cos(np.radians(azimuth)) *
                                         np.cos(np.radians(phi0)) + np.radians(beta) * np.sin(np.radians(azimuth)) *
                                         np.cos(np.radians(theta0)) - np.radians(beta) * np.cos(np.radians(azimuth)) *
                                         np.sin(np.radians(phi0)) * np.sin(np.radians(theta0))) *
                                        np.sinc(np.sqrt(np.radians(alpha)**2 + np.radians(beta)**2))
                                        + (np.sin(np.radians(azimuth)) * np.sin(np.radians(theta0))
                                           + np.cos(np.radians(azimuth)) * np.sin(np.radians(phi0))
                                            * np.cos(np.radians(theta0))) *
                                        np.cos(np.sqrt(np.radians(alpha)**2 + np.radians(beta)**2)))

            ky = 0.512 * np.sqrt(ke) * ((-1 * np.radians(alpha) * np.sin(np.radians(azimuth)) *
                                         np.cos(np.radians(phi0)) - np.radians(beta) * np.cos(np.radians(azimuth)) *
                                         np.cos(np.radians(theta0)) - np.radians(beta) * np.sin(np.radians(azimuth)) *
                                         np.sin(np.radians(phi0)) * np.sin(np.radians(theta0))) *
                                        np.sinc(np.sqrt(np.radians(alpha)**2 + np.radians(beta)**2)) -
                                        (np.cos(np.radians(azimuth)) * np.sin(np.radians(theta0)) -
                                         np.sin(np.radians(azimuth)) * np.sin(np.radians(phi0)) *
                                         np.cos(np.radians(theta0))) *
                                        np.cos(np.sqrt(np.radians(alpha)**2 + np.radians(beta)**2)))

        # Horizontal Slit Deflector K Conversion
        elif slit_orientation == 3:
            kx = 0.512 * np.sqrt(ke) * ((-1 * np.radians(beta) * np.sin(np.radians(azimuth)) *
                                         np.cos(np.radians(phi0)) + np.radians(alpha) * np.cos(np.radians(azimuth)) *
                                         np.cos(np.radians(theta0)) + np.radians(alpha) * np.sin(np.radians(azimuth)) *
                                         np.sin(np.radians(phi0)) * np.sin(np.radians(theta0))) *
                                        np.sinc(np.sqrt(np.radians(alpha)**2 + np.radians(beta)**2)) -
                                        (np.cos(np.radians(azimuth)) * np.sin(np.radians(theta0)) -
                                         np.sin(np.radians(azimuth)) * np.sin(np.radians(phi0)) *
                                         np.cos(np.radians(theta0))) *
                                        np.cos(np.sqrt(np.radians(alpha)**2 + np.radians(beta)**2)))

            ky = 0.512 * np.sqrt(ke) * ((-1 * np.radians(beta) * np.cos(np.radians(azimuth)) *
                                         np.cos(np.radians(phi0)) - np.radians(alpha) * np.sin(np.radians(azimuth)) *
                                         np.cos(np.radians(theta0)) + np.radians(alpha) * np.cos(np.radians(azimuth)) *
                                         np.sin(np.radians(phi0)) * np.sin(np.radians(theta0))) *
                                        np.sinc(np.sqrt(np.radians(alpha)**2 + np.radians(beta)**2)) +
                                        (np.sin(np.radians(azimuth)) * np.sin(np.radians(theta0)) +
                                         np.cos(np.radians(azimuth)) * np.sin(np.radians(phi0)) *
                                         np.cos(np.radians(theta0))) *
                                        np.cos(np.sqrt(np.radians(alpha)**2 + np.radians(beta)**2)))
        else:
            kx = None
            ky = None
            print('Slit Orientation not set correctly')

        return kx, ky
    @staticmethod
    def calc_inverse_rotation_matrix(phi0=0, theta0=0, azimuth=0):
        trot_inv = np.array([[np.cos(np.radians(theta0)) * np.cos(np.radians(azimuth))   , np.cos(np.radians(phi0)) *
                             np.sin(np.radians(azimuth)), -1*np.sin(np.radians(phi0))],
                             [np.sin(np.radians(theta0)) * np.sin(np.radians(phi0)) * np.cos(np.radians(azimuth)) -
                              np.cos(np.radians(theta0)) * np.sin(np.radians(azimuth)),
                              np.sin(np.radians(theta0)) * np.sin(np.radians(phi0)) * np.sin(np.radians(azimuth)) +
                              np.cos(np.radians(theta0)) * np.cos(np.radians(azimuth)),
                              np.sin(np.radians(theta0)) * np.cos(np.radians(phi0))],
                             [np.cos(np.radians(theta0)) * np.sin(np.radians(phi0)) * np.cos(np.radians(azimuth)) +
                              np.sin(np.radians(theta0)) * np.sin(np.radians(azimuth)),
                              np.cos(np.radians(theta0)) * np.sin(np.radians(phi0)) * np.sin(np.radians(azimuth)) -
                              np.sin(np.radians(theta0)) * np.cos(np.radians(azimuth)), np.cos(np.radians(theta0)) *
                              np.cos(np.radians(phi0))]])
        return trot_inv

    @staticmethod
    def reverse_k_conversion(ke, kx, ky, phi0=0, theta0=0, azimuth=0, slit_orientation=0):
        k = 0.512 * np.sqrt(ke)
        kz = np.sqrt(k**2 - kx**2 - ky**2)

        # Vertical Slit
        if slit_orientation == 0:
            alpha = np.degrees(np.arcsin((np.sin(np.radians(phi0)) * kz - (np.cos(np.radians(phi0)) *
                                                                           (np.cos(np.radians(azimuth)) * kx +
                                                                            np.sin(np.radians(azimuth)) * ky))) / k))
            beta = theta0 + np.degrees(np.arctan((np.sin(np.radians(azimuth)) * kx - np.cos(np.radians(azimuth)) * ky) /
                                                 (np.sin(np.radians(phi0)) * np.cos(np.radians(azimuth)) * kx +
                                                  np.sin(np.radians(phi0))*np.sin(np.radians(azimuth)) * ky +
                                                  np.cos(np.radians(phi0)) * kz)))
        # Horizontal Slit - I've switched kx and ky from the Ishida paper to keep kx along the slit,
        # and ky across the slit
        elif slit_orientation == 1:
            alpha = np.degrees(np.arcsin((np.sin(np.radians(phi0))*np.sqrt(k**2 - (np.sin(np.radians(azimuth))*ky -
                                                                                   np.cos(np.radians(azimuth))*kx)**2) -
                                          np.cos(np.radians(phi0))*(np.sin(np.radians(azimuth))*ky -
                                                                    np.cos(np.radians(azimuth))*kx))/k))
            beta = theta0 + np.degrees(np.arctan((np.cos(np.radians(azimuth))*ky + np.sin(np.radians(azimuth)) * kx)
                                                 / kz))
        # Vertical Slit Deflectors
        elif slit_orientation == 2:
            trot_inv = Arpes.calc_inverse_rotation_matrix(phi0=phi0, theta0=theta0, azimuth=azimuth)
            
            alpha = (-180/np.pi)*np.arccos(((trot_inv[2,0] * kx) + (trot_inv[2,1] * ky) + (trot_inv[2,2] * kz))/k)*((trot_inv[0,0] * kx) + (trot_inv[0,1] * ky) + (trot_inv[0,2] * kz))/np.sqrt(k**2 - ((trot_inv[2,0] * kx) + (trot_inv[2,1] * ky)+(trot_inv[2,2] * kz))**2)
            beta = (-180/np.pi)*np.arccos(((trot_inv[2,0] * kx) + (trot_inv[2,1] * ky) + (trot_inv[2,2] * kz)) / k)*(((trot_inv[1,0] * kx) + (trot_inv[1,1] * ky) + (trot_inv[1,2] * kz))/np.sqrt(k**2 - ((trot_inv[2,0] * kx) + (trot_inv[2,1] * ky)+(trot_inv[2,2] * kz))**2))
            
        # Horizontal Slit Deflectors switching ky -> kx from the Ishida paper to keep kx along the slit
        elif slit_orientation == 3:
            trot_inv = Arpes.calc_inverse_rotation_matrix(phi0=phi0, theta0=theta0, azimuth=azimuth)
            alpha = np.degrees(np.arccos((trot_inv[2,0] * ky + trot_inv[2,1] * kx + trot_inv[2,2] * kz)/k) *
                               ((trot_inv[1,0] * ky + trot_inv[1,1] * kx + trot_inv[1,2] * kz) /
                                np.sqrt(k**2 - (trot_inv[2,0] * ky + trot_inv[2,1]*kx + trot_inv[2,2] * kz)**2)))
            beta = -1 * np.degrees(np.arccos((trot_inv[2,0] * ky + trot_inv[2,1] * kx + trot_inv[2,2] * kz) / k) *
                                  ((trot_inv[0,0] * ky + trot_inv[0,1] * kx + trot_inv[0,2] * kz) /
                                   np.sqrt(k**2 - (trot_inv[2,0] * ky + trot_inv[2,1] * kx + trot_inv[2,2] * kz)**2)))
        else:
            alpha = None
            beta = None
            print('slit_orientation not set properly')

        return alpha, beta, ke,


    @requires_ef
    def map_isoenergy_k_irreg(self, ke=None, be=None, binwidth=1, phi0=0, theta0=0, azimuth=0, slit_orientation=0):
        if ke is None:
            ke = be + self.ef
        iso_e = self._obj.arpes.sel_kinetic(ke - binwidth, ke + binwidth).sum('energy')
        alpha, T = np.meshgrid(iso_e.arpes.slit, iso_e.arpes.perp, indexing='ij')
        kx, ky = self.forward_k_conversion(ke, alpha, T, phi0=phi0, theta0=theta0, azimuth=azimuth,
                                           slit_orientation=slit_orientation)
        iso_e = iso_e.assign_coords({'kx': (('slit', 'perp'), kx), 'ky': (('slit', 'perp'), ky)})
        return iso_e

    # Uses an irregularly spaced k-mesh, very quick, good for just plotting.
    # If you want a regularly spaced grid, use spectra_k_reg() to get a regularly spaced rectilinear grid
    # in be vs. k (This is slower)
    # Uses forward k-conversion due to irregularly spaced k-mesh, needs theta_offset  for vertical slit which is
    # assumed to be the offset perpendicular to the slit from normal emission.
    # slit_orientation values 0=vertical slit, 1=horizontal slit, 2=deflector(single cuts)
    @requires_ef
    def spectra_k_irreg(self, phi0=0, theta0=0, slit_orientation=0):
        KE, alpha = np.meshgrid(self._obj.arpes.energy, self._obj.arpes.slit, indexing='ij')
        kx, ky = self.forward_k_conversion(KE, alpha, 0, phi0=phi0, theta0=theta0, slit_orientation=slit_orientation)

        self._obj = self._obj.assign_coords(
            {'kx': (('energy', 'slit'), kx), 'binding': (('energy', 'slit'), KE - self.ef)})

        return self._obj

    # Uses a regularly spaced grid and utilizes linear interpolation via scipy RegularGridInterpolator
    # This function is a bit of a mess, using both forward and reverse k-conversion and some fancy footwork with
    # meshgrids and flattening and the such like. Probably needs work to optimize and make faster
    @requires_ef
    def map_k_reg(self, phi0=0, theta0=0, azimuth=0, slit_orientation=0):
        copy = self._obj.copy()
        # Fix the ordering of dimensions to energy,slit,perp
        copy = copy.transpose('energy', 'slit', 'perp')
        interp_object = RegularGridInterpolator((copy.energy.values, copy.slit.values, copy.perp.values),
                                                copy.values, bounds_error=False, fill_value=0)

        kxmin, ky = Arpes.forward_k_conversion(np.nanmax(copy.energy.values),
                                               np.nanmin(copy.slit.values), 0, phi0=phi0, theta0=theta0,
                                               azimuth=azimuth, slit_orientation=slit_orientation)
        kxmax, ky = Arpes.forward_k_conversion(np.nanmax(copy.energy.values),
                                               np.nanmax(copy.slit.values), 0, phi0=phi0, theta0=theta0,
                                               azimuth=azimuth, slit_orientation=slit_orientation)
        kx, kymin = Arpes.forward_k_conversion(np.nanmax(copy.energy.values), 0,
                                               np.nanmin(copy.perp.values), phi0=phi0, theta0=theta0,
                                               azimuth=azimuth, slit_orientation=slit_orientation)
        kx, kymax = Arpes.forward_k_conversion(np.nanmax(copy.energy.values), 0,
                                               np.nanmax(copy.perp.values), phi0=phi0, theta0=theta0,
                                               azimuth=azimuth, slit_orientation=slit_orientation)

        kx_new = np.sort(np.linspace(kxmin, kxmax, num=copy.slit.size, endpoint=True))
        ky_new = np.sort(np.linspace(kymin, kymax, num=copy.perp.size, endpoint=True))
        energy_new = np.linspace(np.nanmin(copy.energy.values), np.nanmax(copy.energy.values),
                                 num=copy.energy.size, endpoint=True)

        # Using sparse grid fails when you go to flatten things out for interpolation
        energy_grid, kxx, kyy = np.meshgrid(energy_new, kx_new, ky_new, indexing='ij', sparse=False)
        # Timing this function
        t0 = time.time()
        alpha, beta, energy = Arpes.reverse_k_conversion(energy_grid, kxx, kyy, phi0=phi0, theta0=theta0,
                                                         azimuth=azimuth, slit_orientation=slit_orientation)
        t1 = time.time()
        time_elapsed = t1 - t0
        print("Generation of interpolation points time elapsed = " + str(np.around(time_elapsed, decimals=3)) + "s")
        # Flatten out the grids to create a list of interpolation points which we will effectively loop through
        alpha_flat = alpha.reshape(-1, order='C')
        beta_flat = beta.reshape(-1, order='C')
        energy_flat = energy.reshape(-1, order='C')
        points_stacked = np.stack((energy_flat, alpha_flat, beta_flat))

        print("Calling interpolation on " + str(alpha_flat.size) + " points")
        # Do the interpolation
        t2 = time.time()
        interpolation_output = interp_object(points_stacked.T)
        t3 = time.time()
        print("Interpolation time elapsed = " + str(np.around(t3-t2, decimals=3)) + "s")

        interpolation_reshaped = interpolation_output.reshape((alpha.shape[0], alpha.shape[1], alpha.shape[2]),
                                                              order='C')
        return xr.DataArray(interpolation_reshaped, dims=['binding', 'kx', 'ky'],
                            coords={'binding': energy_new - self.ef, 'kx': kx_new, 'ky': ky_new}, attrs=copy.attrs)

    # This is much slower and relies on scipy RegularGridInterpolator and utilizes linear interpolation
    # to produce the rectilinear energy vs. k grid. This may not be strictly mathematically exact in preservation
    # of spectral weight through the transformation. This is roughly equivalent to "TransformToK" in our Igor code
    # Also assumes that the xarray has been constructed with dimensions energy vs. slit not slit vs. energy
    @requires_ef
    def spectra_k_reg(self, phi0):
        copy = self._obj.copy()
        interp_object = RegularGridInterpolator((copy.energy.values, copy.slit.values),
                                                copy.values, bounds_error=False, fill_value=0)
        lowk = phi2k(np.nanmin(copy.slit.values) - phi0, np.nanmax(copy.energy.values))
        highk = phi2k(np.nanmax(copy.slit.values) - phi0, np.nanmax(copy.energy.values))
        numk = copy.slit.size

        lowe = np.nanmin(copy.energy.values)
        highe = np.nanmax(copy.energy.values)
        nume = copy.energy.size

        kx = np.linspace(lowk, highk, num=numk, endpoint=True)
        ke = np.linspace(lowe, highe, num=nume, endpoint=True)
        be = ke - self.ef

        output = np.empty(copy.shape)

        i = 0
        j = 0
        for energy in ke:
            for k in kx:
                counts = interp_object([energy, k2phi(k + phi2k(phi0, energy), energy)])
                output[i, j] = counts
                j += 1
            j = 0
            i += 1
        return xr.DataArray(output,dims=['binding','kx'],coords={'binding':be,'kx':kx},attrs=copy.attrs)

    # This thing is monstrous, please put me out of my misery
    # Here theta0 is the distance to normal emission where this cut was taken
    @requires_ef
    def spectra_k_reg_2(self, phi0=0, theta0=0, slit_orientation=0):
        copy = self._obj.copy()

        # Embedding the cut into a higher dimensional space (3d for 2d spectra)
        # Padding spectra with Theta+-0.1
        empty_space = np.zeros(copy.values.shape)
        empty_xr = xr.DataArray(empty_space, dims=copy.dims, coords=copy.coords, attrs=copy.attrs)
        higher_dimensioned_spectra = xr.concat([empty_xr, copy, empty_xr], 'perp')
        higher_dim_thetas = [theta0-0.1, theta0, theta0+0.1]
        higher_dimensioned_spectra = higher_dimensioned_spectra.assign_coords({'perp': higher_dim_thetas})
        higher_dimensioned_spectra = higher_dimensioned_spectra.transpose("energy", "slit", "perp")
        interp_object = RegularGridInterpolator((higher_dimensioned_spectra.energy.values,
                                                 higher_dimensioned_spectra.slit.values, higher_dim_thetas),
                                                higher_dimensioned_spectra.values, bounds_error=False, fill_value=0)

        kxmin, ky = Arpes.forward_k_conversion(np.nanmax(higher_dimensioned_spectra.energy.values),
                                               np.nanmin(higher_dimensioned_spectra.slit.values), theta0,
                                               phi0=phi0, theta0=0,
                                               azimuth=0, slit_orientation=slit_orientation)
        kxmax, ky = Arpes.forward_k_conversion(np.nanmax(higher_dimensioned_spectra.energy.values),
                                               np.nanmax(higher_dimensioned_spectra.slit.values), theta0,
                                               phi0=phi0, theta0=0,
                                               azimuth=0, slit_orientation=slit_orientation)

        if theta0 <= 0:
            kx, kymin = Arpes.forward_k_conversion(np.nanmax(higher_dimensioned_spectra.energy.values), 0,
                                               np.nanmin(higher_dimensioned_spectra.perp.values),
                                               phi0=phi0, theta0=0,
                                               azimuth=0, slit_orientation=slit_orientation)
            kx, kymax = Arpes.forward_k_conversion(np.nanmax(higher_dimensioned_spectra.energy.values),
                                               np.nanmax(np.abs(higher_dimensioned_spectra.slit.values - phi0)),
                                               np.nanmax(higher_dimensioned_spectra.perp.values),
                                               phi0=phi0, theta0=0,
                                               azimuth=0, slit_orientation=slit_orientation)
        else:
            kx, kymin = Arpes.forward_k_conversion(np.nanmax(higher_dimensioned_spectra.energy.values),
                                               np.nanmax(np.abs(higher_dimensioned_spectra.slit.values - phi0)),
                                               np.nanmin(higher_dimensioned_spectra.perp.values),
                                               phi0=phi0, theta0=0,
                                               azimuth=0, slit_orientation=slit_orientation)
            kx, kymax = Arpes.forward_k_conversion(np.nanmax(higher_dimensioned_spectra.energy.values), 0,
                                               np.nanmax(higher_dimensioned_spectra.perp.values),
                                               phi0=phi0, theta0=0,
                                               azimuth=0, slit_orientation=slit_orientation)

        # What even is this
        numky = 100
        kx_new = np.sort(np.linspace(kxmin, kxmax, num=higher_dimensioned_spectra.slit.size, endpoint=True))
        ky_new = np.sort(np.linspace(kymin, kymax, num=numky, endpoint=True))
        print(ky_new)
        energy_new = np.linspace(np.nanmin(higher_dimensioned_spectra.energy.values),
                                 np.nanmax(higher_dimensioned_spectra.energy.values),
                                 num=higher_dimensioned_spectra.energy.size, endpoint=True)

        # Using sparse grid fails when you go to flatten things out for interpolation
        energy_grid, kxx, kyy = np.meshgrid(energy_new, kx_new, ky_new, indexing='ij', sparse=False)
        # Timing this function
        t0 = time.time()
        alpha, beta, energy = Arpes.reverse_k_conversion(energy_grid, kxx, kyy, phi0=phi0, theta0=0,
                                                         azimuth=0, slit_orientation=slit_orientation)
        t1 = time.time()
        time_elapsed = t1 - t0
        print("Generation of interpolation points time elapsed = " + str(np.around(time_elapsed, decimals=3)) + "s")
        # Flatten out the grids to create a list of interpolation points which we will effectively loop through
        alpha_flat = alpha.reshape(-1, order='C')
        beta_flat = beta.reshape(-1, order='C')
        energy_flat = energy.reshape(-1, order='C')
        points_stacked = np.stack((energy_flat, alpha_flat, beta_flat))

        print("Calling interpolation on " + str(alpha_flat.size) + " points")
        # Do the interpolation
        t2 = time.time()
        interpolation_output = interp_object(points_stacked.T)
        t3 = time.time()
        print("Interpolation time elapsed = " + str(np.around(t3 - t2, decimals=3)) + "s")

        interpolation_reshaped = interpolation_output.reshape((alpha.shape[0], alpha.shape[1], alpha.shape[2]),
                                                              order='C')
        higher_dimension_xr = xr.DataArray(interpolation_reshaped, dims=['binding', 'kx', 'ky'],
                                           coords={'binding': energy_new - self.ef, 'kx': kx_new, 'ky': ky_new},
                                           attrs=copy.attrs)
        flattened = higher_dimension_xr.sum('ky')
        return flattened


    # Kz maps should always be in binding energy, will need to shift off using a fixed work-function to recover
    # kinetic energy for k conversion
    def map_isoenergy_kz_k_irreg(self, be=0, workfunc=4.2, binwidth=0.1, phi0=0, inner_potential=15):
        iso_e = self._obj.arpes.sel_kinetic(be-binwidth, be+binwidth).sum('energy')
        T, F = np.meshgrid(iso_e.arpes.photon_energy, iso_e.arpes.slit, indexing='ij')
        kx = 0.512 * np.sqrt(be + T - workfunc) * np.sin(np.pi / 180 * (F - phi0))
        kz = 0.512 * np.sqrt((be + T - workfunc)*np.cos(np.pi/180 * (F-phi0))**2 + inner_potential)

        iso_e = iso_e.assign_coords({'kx':(('photon_energy','slit'), kx), 'kz':(('photon_energy', 'slit'), kz)})
        return iso_e

    # Currently can only take spectra at gamma point (in angle space) before k-converting
    def spectra_kz_k_irreg(self, binwidth=1, workfunc=4.2, phi0=0, inner_potential=15):
        cut = self._obj.arpes.sel_slit(phi0-binwidth, phi0+binwidth).sum('slit')
        F, BE = np.meshgrid(self._obj.arpes.photon_energy, self._obj.arpes.energy, indexing='ij')
        kz = 0.512 * np.sqrt((F + BE - workfunc) + inner_potential)

        cut = cut.assign_coords({'kz': (('photon_energy', 'energy'), kz)})
        return cut

    @requires_ef
    def cut_from_map_slit_k_irreg(self, theta0=0, phi0=0):
        cut = self._obj.sel({'perp':theta0}, method='nearest')
        KE, F = np.meshgrid(cut.arpes.energy, cut.arpes.slit, indexing='ij')
        kx = 0.512 * np.sqrt(KE) * np.sin(np.pi / 180 * (F - phi0))
        cut = cut.assign_coords({'kx': (('energy','slit'), kx), 'binding': (('energy','slit'), KE - self.ef)})
        return cut

    @requires_ef
    def cut_from_map_perp_k_irreg(self, theta0=0, phi0=0):
        cut = self._obj.sel({'slit': phi0}, method='nearest')
        KE, F = np.meshgrid(cut.arpes.energy, cut.arpes.perp, indexing='ij')
        kx = 0.512 * np.sqrt(KE) * np.sin(np.pi / 180 * (F - theta0))
        cut = cut.assign_coords({'kx': (('energy', 'perp'), kx), 'binding': (('energy', 'perp'), KE - self.ef)})
        return cut

    def normalize(self):
        self._obj.values = (self._obj.values - np.min(self._obj.values)) / (
                    np.max(self._obj.values) - np.min(self._obj.values))

    def set_gamma(self, slit=None, perp=None):
        if slit is not None:
            self._obj.coords['slit'].values -= slit
        if perp is not None:
            self._obj.coords['perp'].values -= perp

    def guess_high_symmetry(self):
        x = self._obj.values
        x_r = np.flip(x)
        cor = np.correlate(x, x_r, mode='full')
        idx = (np.argmax(cor) + 1) / 2
        return np.interp(idx, np.arange(self._obj.coords['slit'].size), self._obj.coords['slit'].values)

    def plot_spectra(self, kspace=False):
        if kspace:
            self._obj.plot(x='kx', y='binding')
        else:
            self._obj.plot(x='slit', y='energy')

    def plot_map(self, kspace=False):
        if self._obj.ndim == 2:
            if 'slit' in self._obj.coords and 'perp' in self._obj.coords:
                if kspace:
                    if 'kx' not in self._obj.coords or 'ky' not in self._obj.coords:
                        raise ValueError('You have not computed kspace yet!')
                    self._obj.plot(x='kx', y='ky')
                else:
                    self._obj.plot(x='slit', y='perp')
        else:
            raise ValueError('Your data is more than two dimensions. Cannot plot a map.')

    @requires_ef
    def waterfall(self, spacing=0.2, linear=True, kspace=False, fig=None, ax=None, **lineparams):
        if fig is None and ax is None:
            fig, ax = plt.subplots(1)
        edc_range = (np.max(self._obj, axis=self._obj.dims.index('slit')).values -
                     np.min(self._obj, axis=self._obj.dims.index('slit')).values)
        if linear:
            spacing = spacing * np.max(edc_range)
        offset = 0
        for i in range(self._obj['energy'].size):
            mdc = self._obj.isel({'energy': i}).values + offset
            if kspace:
                ax.plot(self._obj['kx'].isel({'energy': i}).values, mdc, **lineparams)
            else:
                ax.plot(self._obj.coords['slit'].values, mdc, **lineparams)
            if linear:
                offset += spacing
            else:
                offset += spacing * edc_range[i]
        return fig, ax

    @property
    def kinetic(self):
        return self._obj.coords['energy'].values

    @requires_ef
    @property
    def binding(self):
        return self._obj.coords['energy'].values - self.ef

    @property
    def energy(self):
        return self._obj.coords['energy'].values

    @property
    def slit(self):
        return self._obj.coords['slit'].values

    @property
    def perp(self):
        return self._obj.coords['perp'].values

    @property
    def photon_energy(self):
        return self._obj.coords['photon_energy'].values

    @requires_ef
    def sel_binding(self, *args):
        if len(args) == 2:
            return self._obj.sel({'energy': slice(args[0] + self.ef, args[1] + self.ef)})
        else:
            raise ValueError('binding only accepts min and max')

    def sel_kinetic(self, *args):
        if len(args) == 2:
            return self._obj.sel({'energy': slice(args[0], args[1])})
        else:
            return self._obj.sel({'energy': args[0]})

    def sel_slit(self, *args):
        if len(args) > 1:
            return self._obj.sel({'slit': slice(args[0], args[1])})
        else:
            return self._obj.sel({'slit': args[0]})

    def sel_perp(self, *args):
        if len(args) > 1:
            return self._obj.sel({'perp': slice(args[0], args[1])})
        else:
            return self._obj.sel({'perp': args[0]})

    def guess_ef(self):
        edc = self._obj.copy()
        if edc.ndim > 1:
            for dim_label in edc.dims:
                if dim_label != 'energy':
                    edc = edc.sum(dim_label)
        if edc.size > 150:
            factor = int(np.ceil(edc.size / 100))
            edc = edc.arpes.downsample({'energy': factor})
        edc_y = edc.values
        edc_x = edc.coords['energy'].values
        return edc_x[np.argmin(np.diff(edc_y))]

    @staticmethod
    def guess_fermi_params(edc, bg_order=1):
        """
        downsamples the data so that it contains at most 150 points
        estimate Ef by minimizing the first derivative
        fit the data below Ef to a polynomial of order bg_order
        guess that the resolution is 4.4 meV (TODO: make this more robust by finding width of peak in derivative)
        estimate y0 by suggesting the average of the final few values
        """
        if edc.size > 150:
            factor = int(np.ceil(edc.size / 100))
            edc = edc.arpes.downsample({'energy': factor})
        edc_y = edc.values
        edc_x = edc.coords['energy'].values
        i = np.argmin(np.diff(edc_y))
        ef = edc_x[i]
        p = np.polyfit(edc_x[:(i - 5)], edc_y[:(i - 5)], bg_order)
        kBT = 0.001
        y0 = np.mean(edc_y[-5:])
        return (ef, kBT, y0) + tuple(p)

    @staticmethod
    def fermi_fit(edc, bg_order=1):
        return curve_fit(fermi_fcn_bg, edc.arpes.energy, edc.values, p0=Arpes.guess_fermi_params(edc, bg_order))

    @staticmethod
    def print_fit_params(popt):
        formatter = 'Ef: {0} (eV)\n10-90: {1} (meV)\ny0: {2}\n'
        for i in range(3, len(popt)):
            formatter += 'bg x**{0}: '.format(len(popt) - i - 1)
            formatter += '{' + str(i) + '}\n'
        vals = list(popt)
        vals[1] *= 4400
        print(formatter.format(*vals))

    @staticmethod
    def dewarp_curve(p, x):
        if len(p) == 1:
            return p[0]
        elif len(p) == 2:
            return p[0] * x + p[1]
        elif len(p) == 3:
            return p[0] * x ** 2 + p[1] * x + p[2]
        elif len(p) == 4:
            return p[0] * x ** 3 + p[1] * x ** 2 + p[2] * x + p[3]

    @staticmethod
    def make_dewarp_curve(angles, ef):
        p = np.polyfit(angles, ef, deg=2)
        return partial(Arpes.dewarp_curve, p)

    @staticmethod
    def dewarp_spectra(spectra, dewarp):
        ef_pos = dewarp(spectra.coords['slit'].values)
        ef_min = np.min(ef_pos)
        ef_max = np.max(ef_pos)
        de = spectra.coords['energy'].values[1] - spectra.coords['energy'].values[0]
        px_to_remove = int(round((ef_max - ef_min) / de))
        dewarped = np.empty((spectra.coords['energy'].size - px_to_remove, spectra.coords['slit'].size))
        for i in range(spectra.coords['slit'].size):
            rm_from_bottom = int(round((ef_pos[i] - ef_min) / de))
            rm_from_top = spectra.coords['energy'].size - (px_to_remove - rm_from_bottom)
            dewarped[:, i] = spectra.values[rm_from_bottom:rm_from_top, i]
        bottom_energy_offset = int(round((ef_max - ef_min) / de))
        energy = spectra.coords['energy'].values[bottom_energy_offset:]
        return xr.DataArray(dewarped, coords={'energy': energy, 'slit': spectra.coords['slit'].values},
                            dims=['energy', 'slit'], attrs=spectra.attrs)

    @staticmethod
    def create_dewarp(spectra):
        N = spectra.coords['slit'].size
        slit_angles = spectra.coords['slit'].values
        ef_pos = np.empty(N)
        for i in range(N):
            edc = spectra.isel({'slit': i}).values
            energy = spectra.coords['energy'].values
            params = Arpes.guess_fermi_params(spectra.isel({'slit': i}), 2)
            params, pcov = curve_fit(fermi_fcn_bg, energy, edc, p0=params)
            ef_pos[i] = params[0]
        dewarp = Arpes.make_dewarp_curve(slit_angles, ef_pos)
        return dewarp

    def downsample(self, dsf, operation='mean'):
        """
        dsf is a dict of dims and the amount to downsample by (i.e. 2, 3, 4, etc)
        operation can be sum or mean
        downsampled coordinates are always averaged
        """
        dat = self._obj.copy()
        new_shape = list(dat.shape)
        new_coords = {key: dat[key].values for key in dat.dims}
        dat_mat = dat.values
        for label, ds in dsf.items():
            i = dat.dims.index(label)
            extra = new_shape[i] % ds
            if extra == 0:
                extra = None
            else:
                extra *= -1
            new_shape[i] = new_shape[i] // ds
            new_coords[label] = new_coords[label][:extra].reshape((-1, ds)).mean(axis=1)
            slicer = [slice(None, None)] * dat_mat.ndim
            slicer[i] = slice(None, extra)
            dat_mat = dat_mat[tuple(slicer)]
        mat = bin_ndarray(dat_mat, new_shape, operation=operation)
        return xr.DataArray(mat, coords=new_coords, dims=dat.dims, attrs=dat.attrs)

    def plot(self, layout=ImageTool.LayoutComplete):
        self.it = ImageTool(self._obj, layout=layout)
        self.it.show()
        return self.it
