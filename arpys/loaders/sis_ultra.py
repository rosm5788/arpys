import xarray as xr
import numpy as np
import h5py as h5


def load_sistem_h5(filename):
    f = h5.File(filename,'r')
    counts = f['Electron Analyzer']['Image Data']
    ndims = len(counts.shape)

    conv = {'X': 'x', 'Y': 'y', 'Z': 'z', 'Binding Energy': 'binding', 'Y-Scale': 'slit', 'Tilt': 'perp',
            'Kinetic Energy': 'energy'}
    coords = {}
    dims = []
    for dim in np.arange(ndims):
        axislabel = conv[f['Electron Analyzer']['Image Data'].attrs[
            'Axis' + str(dim) + '.Description'].decode('utf-8').strip()]
        axisoffset = f['Electron Analyzer']['Image Data'].attrs['Axis' + str(dim) + '.Scale'][0]
        axisdelta = f['Electron Analyzer']['Image Data'].attrs['Axis' + str(dim) + '.Scale'][1]
        axis_size = counts.shape[dim]
        coords[axislabel] = np.arange(axis_size) * axisdelta + axisoffset
        dims.append(axislabel)

    attrs = dict(counts.attrs)
    for key in list(f['Other Instruments']):
        attrs[key] = np.float64(f['Other Instruments'][key])
    return xr.DataArray(counts, dims=dims, coords=coords, attrs=attrs)


# Will read all SES .zip type deflector maps, need to first extract .zip to a separate folder and pass in the bin file
# of the spectrum and the associated .ini file that will define the coordinates THIS WILL ONLY WORK FOR MAPS.
def load_ses_deflector_map(binfilename, inifilename):
    conv = {'Energy [eV]': 'energy', 'Thetax [deg]': 'slit', 'Thetay [deg]': 'perp'}
    # Read .ini file and extract all key value pairs
    ini_dictionary = {k: v.strip() for k, v in (l.split('=') for l in open(inifilename).readlines()[1:])}

    # Will read entire .bin file into 1d array
    with open(binfilename) as f:
        data_unshaped = np.fromfile(f, dtype=np.dtype('f4'))

    # Final data shape is "depth, height, width"
    data = np.reshape(data_unshaped, (int(ini_dictionary['depth']), int(ini_dictionary['height']),
                                      int(ini_dictionary['width'])))
    dims = [conv[ini_dictionary['depthlabel']], conv[ini_dictionary['heightlabel']], conv[ini_dictionary['widthlabel']]]

    depth_offset = float(ini_dictionary['depthoffset'])
    depth_size = int(ini_dictionary['depth'])
    depth_delta = float(ini_dictionary['depthdelta'])
    depth_coord = np.linspace(depth_offset, depth_size*depth_delta + depth_offset, endpoint=False, num=depth_size)

    height_offset = float(ini_dictionary['heightoffset'])
    height_size = int(ini_dictionary['height'])
    height_delta = float(ini_dictionary['heightdelta'])
    height_coord = np.linspace(height_offset, height_size*height_delta + height_offset, endpoint=False, num=height_size)

    width_offset = float(ini_dictionary['widthoffset'])
    width_size = int(ini_dictionary['width'])
    width_delta = float(ini_dictionary['widthdelta'])
    width_coord = np.linspace(width_offset, width_size*width_delta + width_offset, endpoint=False, num=width_size)

    coords = {dims[0]: depth_coord, dims[1]: height_coord, dims[2]: width_coord}

    return xr.DataArray(data, dims=dims, coords=coords)

