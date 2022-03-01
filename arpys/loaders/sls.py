import numpy as np
import xarray as xr
import h5py


def load_sls_adress_fermi(filename):
    """
    Loader for swiss light source endstation Adress. The file structure is comprised of
    dictionary keys which also have associated attributes. The "Matrix" key has attributes
    "IGORWaveNote", " IGORWaveScaling", and "IGORWaveUnits". The scaling is given in the
    form of a 2D array where each array is comprised of the bin size and the starting
    value.
    """
    f = h5py.File(filename, 'r')
    scaling = f['Matrix'].attrs['IGORWaveScaling']
    coord_names = f['Matrix'].attrs['IGORWaveUnits']
    data = np.array(f['Matrix'])
    slit_endpoint = scaling[1][1] + data.shape[0] * scaling[1][0]
    slit_angle = np.linspace(scaling[1][1], slit_endpoint, num=data.shape[0])

    energy_endpoint = scaling[2][1] + data.shape[1] * scaling[2][0]
    energy = np.linspace(scaling[2][1], energy_endpoint, num=data.shape[1])

    perp_endpoint = scaling[3][1] + data.shape[2] * scaling[3][0]
    perp = np.linspace(scaling[3][1], perp_endpoint, num=data.shape[2])
    axis_labels = ['slit', 'energy', 'perp']
    coords = {'slit': slit_angle,
              'energy': energy,
              'perp': perp}
    final_dataarray = xr.DataArray(data, dims=axis_labels, coords=coords)
    f.close()
    return final_dataarray


def load_sls_adress_hvscan(filename):
    f = h5py.File(filename, 'r')
    scaling = f['Matrix'].attrs['IGORWaveScaling']
    coord_names = f['Matrix'].attrs['IGORWaveUnits']
    data = np.array(f['Matrix'])
    slit_endpoint = scaling[1][1] + data.shape[0] * scaling[1][0]
    slit_angle = np.linspace(scaling[1][1], slit_endpoint, num=data.shape[0])
    binding_endpoint = scaling[2][1] + data.shape[1] * scaling[2][0]
    binding = np.linspace(scaling[2][1], binding_endpoint, num=data.shape[1])
    energy_endpoint = scaling[3][1] + data.shape[2] * scaling[3][0]
    energy = np.linspace(scaling[3][1], energy_endpoint, num=data.shape[2])
    axis_labels = ['slit', 'energy', 'photon_energy']
    coords = {'slit': slit_angle,
              'energy': binding,
              'photon_energy': energy}
    dataarray = xr.DataArray(data, dims=axis_labels, coords=coords)
    transposed_scan = dataarray.transpose('energy', 'slit', 'photon_energy')
    return transposed_scan
