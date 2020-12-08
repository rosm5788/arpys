from nexusformat.nexus import *
import numpy as np
import xarray as xr
import arpys


def load_diamond_consolidated(filename):
    f = nxload(filename)
    counts = np.array(f.entry1.instrument.analyser.data)
    sapolar = np.array(f.entry1.instrument.manipulator.sapolar)
    energies = np.array(f.entry1.instrument.analyser.energies)
    slit_angle = np.array(f.entry1.instrument.analyser.angles)

    axis_labels = ['slit', 'energy']

    if len(sapolar) == 1:  # alternatively, if counts.shape[0] == 1
        vals = np.copy(counts[0, :, :])
        coords = {'slit': slit_angle, 'energy': energies}

    else:
        vals = np.copy(counts)
        coords = {'perp': sapolar, 'slit': slit_angle, 'energy': energies}
        # append perp label to beginning of axis_labels if the file is a fermi map
        axis_labels.insert(0, 'perp')

    return xr.DataArray(vals, dims=axis_labels, coords=coords)


def load_diamond_fermi(filename):
    f = nxload(filename)
    counts = np.array(f.entry1.instrument.analyser.data)
    sapolar = np.array(f.entry1.instrument.manipulator.sapolar)
    energies = np.array(f.entry1.instrument.analyser.energies)
    slit_angle = np.array(f.entry1.instrument.analyser.angles)

    axis_labels = ['perp', 'slit', 'energy']
    coords = {'perp': sapolar, 'slit': slit_angle, 'energy': energies}

    vals = np.copy(counts)

    return xr.DataArray(vals, dims=axis_labels, coords=coords)


def load_diamond_single(filename):
    f = nxload(filename)
    counts = np.array(f.entry1.instrument.analyser.data)
    sapolar = np.array(f.entry1.instrument.manipulator.sapolar)
    energies = np.array(f.entry1.instrument.analyser.energies)
    slit_angle = np.array(f.entry1.instrument.analyser.angles)

    axis_labels = ['slit', 'energy']
    coords = {'slit': slit_angle, 'energy': energies}

    vals = np.copy(counts[0, :, :])

    return xr.DataArray(vals, dims=axis_labels, coords=coords)
