import numpy as np
import xarray as xr
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from .utility import find_basis


# Boolean typed attributes break netCDF saving functionality used by some group members, this changes
# boolean variables to strings ("True" and "False")
def fix_boolean_attributes(xarray):
    for attr in xarray.attrs:
        if type(xarray.attrs[attr]) == np.bool_:
            xarray.attrs[attr] = str(xarray.attrs[attr])
    return xarray
# 02/2020 and before?? Confirmed to work with data from 02/2020
def load_ssrl_52_prehistoric(filename):
    conv = {'X': 'x', 'Z': 'z', 'ThetaX': 'slit', 'ThetaY': 'perp', 'Theta Y': 'perp', 'Kinetic Energy': 'energy'}
    f = h5py.File(filename, 'r')
    counts = np.array(f['Data']['Count']).T
    if counts.ndim == 4:
        # for some reason, transpose the last two axes?
        counts = np.transpose(counts, axes=[0, 1, 3, 2])
    delta = []
    offset = []
    axis_labels = []
    for i in range(len(counts.shape)):
        axis_key = 'Axes' + str(i)
        try:
            axis_label = conv[f['Data'][axis_key].attrs['Label']]
            axis_labels.append(axis_label)
        except KeyError as e:
            print('Could not understand [{0}] as an arpys axis!'.format(f['Data'][axis_key].attrs['Label']))
            return None
        delta.append(f['Data'][axis_key].attrs['Delta'])
        offset.append(f['Data'][axis_key].attrs['Offset'])
    vals = np.copy(counts)
    if 'Time' in f['Data']:
        exposure = np.array(f['Data']['Time']).T
        select = exposure > 0
        vals[select] = counts[select] / exposure[select]
        vals[~select] = 0
    coords = {}
    for delt, offs, label, size in zip(delta, offset, axis_labels, counts.shape):
        coords[label] = np.arange(size) * delt + offs
    attrs = dict(f['Beamline'].attrs).copy()
    attrs.update(dict(f['Endstation'].attrs))
    attrs.update(dict(f['Details'].attrs))
    f.close()
    return xr.DataArray(vals, dims=axis_labels, coords=coords, attrs=attrs)


# Prior to 10/2020
def load_ssrl_52_old(filename):
    conv = {'X': 'x', 'Z': 'z', 'ThetaX': 'slit', 'ThetaY': 'perp', 'Theta Y': 'perp', 'Kinetic Energy': 'energy'}
    f = h5py.File(filename, 'r')
    counts = np.array(f['data']['counts'])
    if counts.ndim == 4:
        # for some reason, transpose the last two axes?
        counts = np.transpose(counts, axes=[0, 1, 3, 2])
    delta = []
    offset = []
    axis_labels = []
    for i in range(len(counts.shape)):
        axis_key = 'axis' + str(i)
        try:
            axis_label = conv[f['data'][axis_key].attrs['label']]
            axis_labels.append(axis_label)
        except KeyError as e:
            print('Could not understand [{0}] as an arpys axis!'.format(f['data'][axis_key].attrs['label']))
            return None
        delta.append(f['data'][axis_key].attrs['delta'])
        offset.append(f['data'][axis_key].attrs['offset'])
    vals = np.copy(counts)
    if 'exposure' in f['data']:
        exposure = np.array(f['data']['exposure'])
        select = exposure > 0
        vals[select] = counts[select] / exposure[select]
        vals[~select] = 0
    coords = {}
    for delt, offs, label, size in zip(delta, offset, axis_labels, counts.shape):
        coords[label] = np.arange(size)*delt + offs
    attrs = dict(f['beamline'].attrs).copy()
    attrs.update(dict(f['endstation'].attrs))
    attrs.update(dict(f['analyzer'].attrs))
    f.close()
    return xr.DataArray(vals, dims=axis_labels, coords=coords, attrs=attrs)

#After 10/2020
def load_ssrl_52(filename):
    conv = {'X': 'x', 'Z': 'z', 'ThetaX': 'slit', 'ThetaY': 'perp', 'Theta Y': 'perp', 'Kinetic Energy': 'energy'}
    f = h5py.File(filename, 'r')
    counts = np.array(f['Data']['Count'])
    if counts.ndim == 4:
        # for some reason, transpose the last two axes?
        counts = np.transpose(counts, axes=[0, 1, 3, 2])
    delta = []
    offset = []
    axis_labels = []
    for i in range(len(counts.shape)):
        axis_key = 'Axes' + str(i)
        try:
            axis_label = conv[f['Data'][axis_key].attrs['Label']]
            axis_labels.append(axis_label)
        except KeyError as e:
            print('Could not understand [{0}] as an arpys axis!'.format(f['data'][axis_key].attrs['label']))
            return None
        delta.append(f['Data'][axis_key].attrs['Delta'])
        offset.append(f['Data'][axis_key].attrs['Offset'])
    vals = np.copy(counts)
    if 'Time' in f['Data']:
        exposure = np.array(f['Data']['Time'])
        select = exposure > 0
        vals[select] = counts[select] / exposure[select]
        vals[~select] = 0
    coords = {}
    for delt, offs, label, size in zip(delta, offset, axis_labels, counts.shape):
        coords[label] = np.arange(size) * delt + offs
    attrs = dict(f['Beamline'].attrs).copy()
    attrs.update(dict(f['Manipulator'].attrs))
    attrs.update(dict(f['Measurement'].attrs))
    attrs.update(dict(f['Temperature'].attrs))
    attrs.update(dict(f['UserSettings'].attrs))
    attrs.update(dict(f['UserSettings']['AnalyserSlit'].attrs))
    f.close()
    xarray = xr.DataArray(vals, dims=axis_labels, coords=coords, attrs=attrs)
    boolean_fixed = fix_boolean_attributes(xarray)
    return boolean_fixed

def load_ssrl_52_raster(filename):
    conv = {'Kinetic Energy': 'energy', 'ThetaX': 'slit', 'ThetaY': 'perp', 'Theta Y': 'perp', 'X': 'x', 'Y': 'y', 'Z': 'z'}
    with h5py.File(filename, 'r') as f:
        counts = np.array(f['Data']['Count'])
        if counts.ndim == 3:
            counts = np.transpose(counts, (2, 1, 0))
        elif counts.ndim == 4:
            counts = np.transpose(counts, (2, 3, 0, 1))
        delta = []
        offset = []
        axis_labels = []
        for i in range(counts.ndim):
            axis_key = 'Axes' + str(i)
            try:
                print(f['Data'][axis_key].attrs['Label'])
                axis_label = conv[f['Data'][axis_key].attrs['Label']]
                axis_labels.append(axis_label)
            except KeyError as e:
                print('Could not understand [{0}] as an arpys axis!'.format(axis_label))
                return None
            delta.append(f['Data'][axis_key].attrs['Delta'])
            offset.append(f['Data'][axis_key].attrs['Offset'])
        exposure = None
        if 'Exposure' in f['Data']:
            exposure = np.array(f['Data']['Exposure'])
        elif 'Time' in f['Data']:
            exposure = np.array(f['Data']['Time'])
        if exposure is not None:
            if exposure.ndim == 3:
                exposure = np.transpose(exposure, (2, 1, 0))
            select = exposure > 0
            counts[select] = counts[select] / exposure[select]
            counts[~select] = 0
        coords = {}
        for i in range(counts.ndim):
            coords[axis_labels[i]] = np.arange(counts.shape[i])*delta[i] + offset[i]
        attrs = dict(f['Beamline'].attrs)
        attrs.update(dict(f['Endstation'].attrs))
        return xr.DataArray(counts, dims=axis_labels, coords=coords, attrs=attrs)

def load_ssrl_52_raster_list(filenames, x_min=None, dx=None, Nx=None, z_min=None, dz=None, Nz=None, gaps=None):
    x = []
    z = []
    with h5py.File(filenames[0], 'r') as f:
        Nenergy, Nslit = f['Data']['Count'].shape
        delta_energy = f['Data']['Axes0'].attrs['Delta']
        offset_energy = f['Data']['Axes0'].attrs['Offset']
        delta_slit = f['Data']['Axes1'].attrs['Delta']
        offset_slit = f['Data']['Axes1'].attrs['Offset']
        energy = offset_energy + delta_energy*np.arange(Nenergy)
        slit = offset_slit + delta_slit*np.arange(Nslit)
        attrs = dict(f['Beamline'].attrs)
        attrs.update(dict(f['Endstation'].attrs))
    for filename in filenames:
        with h5py.File(filename, 'r') as f:
            x.append(f['Endstation'].attrs['X'])
            z.append(f['Endstation'].attrs['Z'])
    x = np.array(x)
    z = np.array(z)
    plt.plot(x, z, '+')
    plt.show()
    if x_min is None or z_min is None:
        x_min, dx, Nx, z_min, dz, Nz = find_basis(x, z, threshold=0.005)
    x_coord = x_min + dx*np.arange(Nx)
    z_coord = z_min + dz*np.arange(Nz)
    counts = np.empty((Nenergy, Nslit, Nx*Nz))
    if gaps is not None:
        if not isinstance(gaps[0], list):
            gaps = [gaps]
        for gap in gaps:
            pos = gap[0] + Nz*gap[1]
            print(pos)
            filenames.insert(pos, '')
    for i, filename in enumerate(filenames):
        if filename == '':
            counts[:, :, i] = np.zeros((Nenergy, Nslit))
        else:
            with h5py.File(filename, 'r') as f:
                counts[:, :, i] = np.array(f['Data']['Count'])
    counts = counts.reshape((Nenergy, Nslit, Nz, Nx))
    counts = np.transpose(counts, (0, 1, 3, 2))
    dims = ('energy', 'slit', 'x', 'z')
    coords = {'energy': energy, 'slit': slit, 'x': x_coord, 'z': z_coord}
    return xr.DataArray(counts, dims=dims, coords=coords, attrs=attrs)


def load_ssrl_52_photonEscan(filename):
    conv = {'X': 'x', 'Z': 'z', 'ThetaX': 'slit', 'ThetaY': 'perp', 'Theta Y': 'perp', 'Kinetic Energy': 'energy'}
    f = h5py.File(filename, 'r')
    # 3d dataset, kinetic energy, angle, photon energy
    counts = np.array(f['Data']['Count'])
    I0 = np.abs(np.array(f['MapInfo']['Beamline:I0']))

    xaxis_offsets = np.array(f['MapInfo']['Measurement:XAxis:Offset'])
    xaxis_maxs = np.array(f['MapInfo']['Measurement:XAxis:Maximum'])
    xaxis_size = counts.shape[0]

    yaxis_offsets = np.array(f['MapInfo']['Measurement:YAxis:Offset'])
    yaxis_deltas = np.array(f['MapInfo']['Measurement:YAxis:Delta'])
    yaxis_size = counts.shape[1]

    zaxis_offset = f['Data']['Axes2'].attrs['Offset']
    zaxis_delta = f['Data']['Axes2'].attrs['Delta']
    zaxis_size = counts.shape[2]
    zaxis_max = zaxis_size*zaxis_delta + zaxis_offset
    zaxis_coord = np.linspace(zaxis_offset, zaxis_max, num=zaxis_size)

    photon_energy_scan_dataarrays = []

    #Slice by slice along z (photon energy)
    for photon_energy_slice in np.arange(zaxis_size):
        ekslice = counts[:, :, photon_energy_slice] / I0[photon_energy_slice]
        kinetic_coords = np.linspace(xaxis_offsets[photon_energy_slice],xaxis_maxs[photon_energy_slice], num=xaxis_size)
        angle_coords = np.arange(yaxis_size)*yaxis_deltas[photon_energy_slice] + yaxis_offsets[photon_energy_slice]
        dims = ('energy', 'slit')
        coords = {'energy':kinetic_coords, 'slit':angle_coords}
        ekslice_dataarray = xr.DataArray(ekslice, dims=dims, coords=coords)

        #Cut down on window to find ef with initial guess, will always need tuning if mono drifts too much...
        photon_energy = zaxis_coord[photon_energy_slice]
        workfunc = 4.365
        efguess = photon_energy - workfunc
        maxkinetic = np.nanmax(kinetic_coords)
        effinder = ekslice_dataarray.sel({'energy': slice(efguess-1.0, maxkinetic)})
        ef = effinder.arpes.guess_ef()
        binding_coords = kinetic_coords - ef

        newcoords = {'energy': binding_coords, 'slit': angle_coords}
        ekslice_binding = xr.DataArray(ekslice,dims=dims, coords=newcoords)
        photon_energy_scan_dataarrays.append(ekslice_binding)

    aligned_eks = []
    first_ek = photon_energy_scan_dataarrays[0]
    aligned_eks.append(first_ek)

    for i in np.arange(1,len(photon_energy_scan_dataarrays)):
        interped = photon_energy_scan_dataarrays[i].interp_like(first_ek)
        aligned_eks.append(interped)

    aligned_photonE_scan = xr.concat(aligned_eks,'photon_energy')
    aligned_photonE_scan = aligned_photonE_scan.assign_coords(coords={'photon_energy': zaxis_coord})
    return aligned_photonE_scan


def load_ssrl_54_single(filename):
    file = open(filename)
    all_lines = file.readlines()
    dimension1coord = np.fromstring(all_lines[8][18:], sep=' ')
    dimension2coord = np.fromstring(all_lines[12][18:], sep=' ')

    dims = ['energy', 'slit']
    coords = {'energy': dimension1coord, 'slit': dimension2coord}

    info1 = all_lines[16:43]
    attrs = {}
    for line in info1:
        x = line.split('=')
        attrs[x[0]] = x[-1].strip()

    userinfo = all_lines[45:81]
    for line in userinfo:
        x = line.split('=')
        attrs[x[0]] = x[-1].strip()

    dataframe = pd.read_csv(filename, skiprows=85, delim_whitespace=True, header=None).drop([0], axis=1)
    raw_data = dataframe.to_numpy().astype('float64')

    return xr.DataArray(raw_data, dims=dims, coords=coords, attrs=attrs)


def load_ssrl_54_fmap(filenames):
    cuts = []
    perps = []
    for file in filenames:
        single_cut = load_ssrl_54_single(file)
        single_theta = float(single_cut.attrs['T'])
        perps.append(single_theta)
        cuts.append(single_cut)

    perps = np.array(perps)
    concatenated = xr.concat(cuts, 'perp')
    concatenated = concatenated.assign_coords(coords={'perp':perps})

    return concatenated
