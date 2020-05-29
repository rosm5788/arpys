import numpy as np
import xarray as xr
import h5py
import matplotlib.pyplot as plt

def load_ssrl_52(filename):
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