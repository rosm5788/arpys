import numpy as np
import xarray as xr
from scipy.optimize import curve_fit as cfit
import sys


def fermi_fcn_linear_bg(x, ef, kBT, y0, a, b):
    return (a + b * x) / (np.exp((x - ef) / kBT) + 1) + y0


def fit_fermi_func(edc, p0):
    energies = edc.arpes.energy
    return cfit(fermi_fcn_linear_bg, energies, edc.sel({'energy': energies}, method='nearest'), p0=p0)

def normalize_hvscan(hvscan):
    #break hvscan into each photon energy
    hv_cuts = []
    for hv in hvscan['photon_energy']:
        hv_cut = hvscan.sel({'photon_energy': hv}, method='nearest')
        sum = hv_cut.sum('slit').sum('energy')
        area = hv_cut.energy.size * hv_cut.slit.size
        average = sum/area
        hv_cut_normed = hv_cut/average
        hv_cuts.append(hv_cut_normed)

    hvscan_normed = xr.concat(hv_cuts, 'photon_energy')
    hvscan_normed = hvscan_normed.assign_coords({'photon_energy':hvscan['photon_energy']})
    return hvscan_normed

# pass in usual hvscan object (xarray with slit,energy (in binding),photon energy axes),
# energy_window should be a slice object to cut down the energy window,
# initial_guess is a p0 object for scipy.optimize.curve_fit, p0=[ef,kbT,y0,a,b]
def align_ef_hvscan(hvscan, energy_window, initial_guess):
    hv_cuts = []
    # fit each cut and align ef to 0 binding
    for hv in hvscan['photon_energy']:
        hv_cut = hvscan.sel({'photon_energy': hv}, method='nearest')
        edc = hv_cut.sel({'energy': energy_window}).sum('slit')
        popt, pcov = fit_fermi_func(edc, initial_guess)
        new_ef = popt[0]
        new_be = hv_cut['energy'] - new_ef
        hv_cuts.append(hv_cut.assign_coords({'energy': new_be}))

    hv_cuts_interped = []
    hv_cuts_interped.append(hv_cuts[0])
    for scan_no in np.arange(1,len(hv_cuts)):
        hv_cuts_interped.append(hv_cuts[scan_no].interp_like(hv_cuts[0]))

    hv_cuts_aligned = xr.concat(hv_cuts_interped, 'photon_energy')
    hv_cuts_aligned = hv_cuts_aligned.assign_coords({'photon_energy': hvscan['photon_energy']})
    return hv_cuts_aligned