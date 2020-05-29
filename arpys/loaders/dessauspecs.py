import xarray as xr
import numpy as np
try:
    import igor.binarywave as igor
except ImportError as e:
    import warnings
    warnings.warn('You cannot import SPECS from Dessau without igor module')

def load_specs_ibw(filename):
    dat = igor.load(filename)
    mat = dat['wave']['wData']
    wavenote = load_wave_note(dat)
    ke = float(wavenote['Kinetic Energy'])
    pe = float(wavenote['Pass Energy'])
    e_delta = (0.000060241-0.00000030146*pe)*pe*(1920/mat.shape[0])
    e_offset = ke - e_delta*mat.shape[0]*(937/1920)
    a_offset = -16.430
    a_delta = 0.0255102*(1200/mat.shape[1])
    energy = e_offset + e_delta*np.arange(mat.shape[0])
    angle = a_offset + a_delta*np.arange(mat.shape[1])
    return xr.DataArray(mat, coords={'energy': energy, 'slit': angle}, dims=('energy', 'slit'), attrs=wavenote)

def load_wave_note(wave):
    wavenote_dict = {}
    for x in wave['wave']['note'].decode('utf-8').split('\n'):
        if len(x) == 0:
            continue
        split = x.split(':')
        if len(split) == 1:
            continue
        elif len(split) > 2:
            split = [split[0], ':'.join(split[1:])]
        key, val = split
        wavenote_dict[key] = val
    return wavenote_dict

def load_map(file_list, dewarp=None):
    if dewarp is None:
        wave_list = [load_specs_ibw(filename) for filename in file_list]
    else:
        wave_list = [dewarp_spectra(load_specs_ibw(filename), dewarp) for filename in file_list]
    axis0_label = wave_list[0].dims[0]
    axis1_label = wave_list[0].dims[1]
    axis0 = wave_list[0].coords[axis0_label].values
    axis1 = wave_list[0].coords[axis1_label].values
    phi = np.array([float(w.attrs['Phi']) for w in wave_list])
    mat = np.empty((wave_list[0].shape[0], wave_list[1].shape[1], phi.size))
    for i in range(phi.size):
        mat[:, :, i] = wave_list[i].values
    return xr.DataArray(mat, coords={axis0_label: axis0, axis1_label: axis1, 'perp': phi}, dims=(axis0_label, axis1_label, 'perp'))

def load_scan(file_list):
    wave_list = [load_specs_ibw(filename) for filename in file_list]
    axis0_label = wave_list[0].dims[0]
    axis1_label = wave_list[0].dims[1]
    axis0 = wave_list[0].coords[axis0_label].values
    axis1 = wave_list[0].coords[axis1_label].values
    scan = np.arange(len(wave_list))
    mat = np.empty((wave_list[0].shape[0], wave_list[1].shape[1], scan.size))
    for i in range(scan.size):
        mat[:, :, i] = wave_list[i].values
    return xr.DataArray(mat, coords={axis0_label: axis0, axis1_label: axis1, 'scan': scan}, dims=(axis0_label, axis1_label, 'scan'))
