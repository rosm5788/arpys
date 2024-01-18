import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd

try:
    import igor.binarywave as igor
except ImportError as e:
    import warnings
    warnings.warn('You cannot import HiSOR data without igor module')

def load_hisor_ibw(filename):
    wave = igor.load(filename)
    data = wave['wave']['wData']
    note = wave['wave']['note']
    sfA = wave['wave']['wave_header']['sfA']
    sfB = wave['wave']['wave_header']['sfB']
    ndim = wave['wave']['wave_header']['nDim']

    KE = np.linspace(sfB[0], sfA[0] * ndim[0] + sfB[0], num=ndim[0], endpoint=False)
    angles = np.linspace(sfB[1], sfA[1] * ndim[1] + sfB[1], num=ndim[1], endpoint=False)
    attrs = load_wave_note(note)
    return xr.DataArray(data, coords={'energy': KE, 'slit': angles}, dims=('energy', 'slit'),attrs=attrs)


def load_wave_note(note):
    wavenote_dict = {}
    for x in str(note).split('\\r'):
        if len(x) == 0:
            continue
        split = x.split('=')
        if len(split) == 1:
            continue
        elif len(split) > 2:
            split = [split[0], '='.join(split[1:])]
        key, val = split
        wavenote_dict[key] = val
    return wavenote_dict


# REGION_INI refers to the ini file with "Spectrum_" prefix, MAIN_INI does not have the prefix. This is from unzipped
# SES FS Maps (.bin and .ini files)
def load_ses_map(REGION_INI, MAIN_INI, FS_PATH):

    #Read REGION_INI
    with open(REGION_INI) as region_ini:
        widthoffset = 0
        for line in region_ini:
            l = line
            match l:
                case str(x) if x.startswith("widthoffset="):
                    widthoffset = float(x.split("=")[1])
                case str(x) if x.startswith("widthdelta="):
                    widthdelta = float(x.split("=")[1])
                case str(x) if x.startswith("width="):
                    widthnum = int(x.split("=")[1])
                case str(x) if x.startswith("heightoffset="):
                    heightoffset = float(x.split("=")[1])
                case str(x) if x.startswith("heightdelta="):
                    heightdelta = float(x.split("=")[1])
                case str(x) if x.startswith("height="):
                    heightnum = int(x.split("=")[1])
                case str(x) if x.startswith("depthoffset="):
                    depthoffset = float(x.split("=")[1])
                case str(x) if x.startswith("depthdelta="):
                    depthdelta = float(x.split("=")[1])
                case str(x) if x.startswith("depth="):
                    depthnum = int(x.split("=")[1])
                case str(x) if x.startswith("widthlabel="):
                    widthlabel = str(x.split("=")[1].strip())
                case str(x) if x.startswith("heightlabel="):
                    heightlabel = str(x.split("=")[1].strip())
                case str(x) if x.startswith("depthlabel="):
                    depthlabel = str(x.split("=")[1]).strip()

    match str(widthlabel):
        case "Energy [eV]":
            energy = np.linspace(widthoffset, widthoffset + widthnum * widthdelta, num=widthnum, endpoint=False)
        case "Thetax [deg]":
            slit = np.linspace(widthoffset, widthoffset + widthnum * widthdelta, num=widthnum, endpoint=False)
        case "Thetay [deg]":
            perp = np.linspace(widthoffset, widthoffset + widthnum * widthdelta, num=widthnum, endpoint=False)

    match heightlabel:
        case "Energy [eV]":
            energy = np.linspace(heightoffset, heightoffset + heightnum * heightdelta, num=heightnum, endpoint=False)
        case "Thetax [deg]":
            slit = np.linspace(heightoffset, heightoffset + heightnum * heightdelta, num=heightnum, endpoint=False)
        case "Thetay [deg]":
            perp = np.linspace(heightoffset, heightoffset + heightnum * heightdelta, num=heightnum, endpoint=False)

    match depthlabel:
        case "Energy [eV]":
            energy = np.linspace(depthoffset, depthoffset + depthnum * depthdelta, num=depthnum, endpoint=False)
        case "Thetax [deg]":
            slit = np.linspace(depthoffset, depthoffset + depthnum * depthdelta, num=depthnum, endpoint=False)
        case "Thetay [deg]":
            perp = np.linspace(depthoffset, depthoffset + depthnum * depthdelta, num=depthnum, endpoint=False)

    # Read MAIN_INI for attributes and metadata
    attrs = read_main_ini(MAIN_INI)

    # Reshape the binary file into the correct shape (this may break on other sets of data, watch out)
    binaryfile = np.fromfile(FS_PATH,dtype=np.float32)
    data = np.reshape(binaryfile,(widthnum, heightnum, depthnum),order='F')

    return xr.DataArray(data,coords={'slit':slit,'energy':energy,'perp':perp},dims=('energy','slit','perp'),attrs=attrs)


def read_main_ini(MAIN_INI):
    attrs = {}
    with open(MAIN_INI) as main_ini:
        for line in main_ini:
            split = line.split("=")
            if len(split) > 1:
                key = split[0]
                value = split[1]
                attrs[key] = value
    return attrs

def load_raster_map(LOGFILE):
    logfilepath = Path(LOGFILE)
    prefix = str(logfilepath.parent)

    logfile_dataframe = pd.read_csv(logfilepath, sep="\t", header=None, names=['filename', 'polar', 'tilt', 'progx', 'progy', 'progz', 'azimuth', 'encx', 'ency', 'encz'], index_col=False)
    lowx = logfile_dataframe['progx'].min()
    highx = logfile_dataframe['progx'].max()

    sort_backward = np.lexsort((logfile_dataframe['progx'], logfile_dataframe['progz']))
    deltax = logfile_dataframe['progx'][sort_backward[1]] - logfile_dataframe['progx'][sort_backward[0]]
    lenx = int((highx - lowx) / deltax) + 1

    lowz = logfile_dataframe['progz'].min()
    highz = logfile_dataframe['progz'].max()
    deltaz = np.abs(logfile_dataframe['progz'][sort_backward[1] + 2] - logfile_dataframe['progz'][sort_backward[1] + 1])

    lenz = int(np.ceil((np.abs(lowz) - np.abs(highz)) / deltaz)) + 1

    x_linspace = np.round(np.linspace(lowx, highx, num=lenx, endpoint=True), decimals=3)
    z_linspace = np.round(np.linspace(lowz, highz, num=lenz, endpoint=True), decimals=3)

    data_array = []
    for index, row in logfile_dataframe.iterrows():
        filename = Path(prefix) / str(row['filename'])
        progx = row['progx']
        progy = row['progy']
        progz = row['progz']
        data = load_hisor_ibw(filename)
        data.attrs.update({'progx': progx, 'progy': progy, 'progz': progz})
        data_array.append(data)

    slit = data_array[0].slit
    energy = data_array[0].energy
    xa_empty = xr.DataArray(coords={'slit': slit, 'energy': energy, 'x': x_linspace, 'z': z_linspace},
                            dims=('slit', 'energy', 'x', 'z'))

    # Pain
    for data in data_array:
        progx = data.progx
        progz = data.progz
        xa_empty.loc[{'slit': slit, 'energy': energy, 'x': progx, 'z': progz}] = data

    return xa_empty