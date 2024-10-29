#this has been adapted from Dushyant's Hisor Loaders
import xarray as xr
xr.set_options(keep_attrs=True)

import numpy as np
from pathlib import Path
import pandas as pd
from io import StringIO
import io
from zipfile import ZipFile

try:
    import igor.binarywave as igor
    import igor.igorpy as igorpy
except ImportError as e:
    import warnings
    warnings.warn('You cannot import HiSOR data without igor module')

# NSLS outputs all .pxt or .ibw files into a directory. the .pxt files appear to be
# easier to manage (since their names are easier to mimic with the generate filenames function)
# if you want an ibw version, copy-paste this function (adding _ibw) and use load_nsls_ibw
def load_hvscan(filelist,attempt_alignment=False):
    scanlist = []
    photon_energies = []
    for file in filelist:
        scan = load_nsls_pxt(file)
        scanlist.append(scan)
        photon_energies.append(float(scan.attrs['Excitation Energy']))
    
    first_binding = scanlist[0]['energy'] - np.float64(scanlist[0].attrs['Excitation Energy']) + 4.4
    scanlist[0]['energy'] = first_binding

    binding_list = []
    binding_list.append(scanlist[0])
    for i in range (1,len(scanlist)):
        binding_guess = scanlist[i]['energy'] - np.float64(scanlist[i].attrs['Excitation Energy']) + 4.4
        binding_slice = scanlist[i].assign_coords({'energy':binding_guess})
        binding_list.append(binding_slice.interp_like(binding_list[0]))

    #attempts better alignment by guessing the fermi level for each cut
    #might negate some mono drift 
    if attempt_alignment == True:
        for cut in binding_list:
            new_ef = cut.sel(energy=slice(-1,1)).arpes.guess_ef()
            cut = cut.assign_coords(energy = cut.energy.values - new_ef)

    hvscan = xr.concat(binding_list,dim='photon_energy')
    hvscan.coords['photon_energy'] = photon_energies
    hvscan = hvscan.rename({'energy':'binding'})
    return hvscan

def load_nsls_ibw(filename):
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

def load_nsls_pxt(filename):
    file = igorpy.load(filename, initial_byte_order='<')
    igor_wave = file.children[0]

    axis_labels = ['energy','slit']
    coords = dict(zip(axis_labels, igor_wave.axis))
    notes = list(filter(None, str(igor_wave.notes).replace("\\r", "\\n").split("\\n")))[1:-2]
    note_dictionary = []
    for note in notes:
        line = note.split("=")
        if len(line) == 2:
            note_dictionary.append((line[0], line[1]))
    note_dictionary = dict(note_dictionary)

    return xr.DataArray(
        igor_wave.data,
        coords=coords,
        dims=axis_labels,
        attrs=note_dictionary
    )
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
    binaryfile = np.fromfile(FS_PATH, dtype=np.float32)
    data = np.reshape(binaryfile,(widthnum, heightnum, depthnum),order='F')

    return xr.DataArray(data,coords={'slit':slit,'energy':energy,'perp':perp},dims=('energy','slit','perp'),attrs=attrs)

#Loads SES map directly from .zip file into memory, no need to extract .zip files first into folders
# USE THIS WHEN READING FERMI MAPS
def load_ses_map_zip(zipfile):
    input_zip = ZipFile(zipfile)
    _region_ini = ""
    _fs_path = ""
    _main_ini = ""
    for name in input_zip.namelist():
        if name.endswith('.bin'):
            _fs_path = name
        elif name.startswith('Spectrum') and name.endswith('.ini'):
            _region_ini = name
        else:
            _main_ini = name


    #Read REGION_INI
    with io.TextIOWrapper(input_zip.open(_region_ini), encoding="utf-8") as region_ini:
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
    attrs = read_main_ini_zipped(input_zip, _main_ini)

    # Reshape the binary file into the correct shape (this may break on other sets of data, watch out)
    with input_zip.open(_fs_path, mode='r') as FS_PATH:
        data = FS_PATH.read()
        binaryfile = np.frombuffer(data, dtype=np.float32)

    data = np.reshape(binaryfile,(widthnum, heightnum, depthnum),order='F')

    return xr.DataArray(data,coords={'slit':slit,'energy':energy,'perp':perp},dims=('energy','slit','perp'),attrs=attrs)

def read_main_ini_zipped(zipfile, MAIN_INI):
    attrs = {}
    with io.TextIOWrapper(zipfile.open(MAIN_INI), encoding="utf-8") as main_ini:
        for line in main_ini:
            split = line.split("=")
            if len(split) > 1:
                key = split[0]
                value = split[1]
                attrs[key] = value
    return attrs

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
    deltax = np.round(logfile_dataframe['progx'][sort_backward[1]] - logfile_dataframe['progx'][sort_backward[0]],decimals=3)
    rangex = np.round(highx - lowx,decimals=3)
    dividex = np.round(rangex/deltax,decimals=3)
    lenx = int(dividex) + 1

    lowz = logfile_dataframe['progz'].min()
    highz = logfile_dataframe['progz'].max()
    deltaz = np.round(np.abs(logfile_dataframe['progz'][sort_backward[1] + 2] - logfile_dataframe['progz'][sort_backward[1] + 1]),decimals=3)


    rangez = np.round(np.abs(lowz) - np.abs(highz),decimals=3)
    dividez = np.round(rangez / deltaz, decimals=3)
    lenz = int(dividez) + 1


    x_linspace = np.round(np.linspace(lowx, highx, num=lenx, endpoint=True), decimals=3)
    z_linspace = np.round(np.linspace(lowz, highz, num=lenz, endpoint=True), decimals=3)
    #z_linspace = np.round(np.arange(lowz,highz,deltaz),decimals=3)


    data_array = []
    for index, row in logfile_dataframe.iterrows():
        filename = Path(prefix) / str(row['filename'])
        progx = row['progx']
        progy = row['progy']
        progz = row['progz']
        data = load_hisor_ibw(filename)
        data.attrs.update({'progx': progx, 'progy': progy, 'progz': progz})
        data_array.append(data)

    slit = data_array[0].arpes.downsample({'energy':2,'slit':2}).slit
    energy = data_array[0].arpes.downsample({'energy':2,'slit':2}).energy
    xa_empty = xr.DataArray(coords={'slit': slit, 'energy': energy, 'x': x_linspace, 'z': z_linspace},
                            dims=('slit', 'energy', 'x', 'z'))

    # Pain
    for data in data_array:
        progx = data.progx
        progz = data.progz
        xa_empty.loc[{'slit': slit, 'energy': energy, 'x': progx, 'z': progz}] = data.arpes.downsample({'energy':2,'slit':2})

    return xa_empty

def load_raster_1d(logfile):
    logfilepath = Path(logfile)
    prefix = str(logfilepath.parent)
    logfile_dataframe = pd.read_csv(logfilepath, sep="\t", header=None, names=['filename', 'polar', 'tilt', 'progx', 'progy', 'progz', 'azimuth', 'encx', 'ency', 'encz'], index_col=False)
    lowx = logfile_dataframe['progx'].min()
    highx = logfile_dataframe['progx'].max()
    deltax = logfile_dataframe['progx'][1] - logfile_dataframe['progx'][0]
    lenx = int((highx - lowx) / deltax) + 1
    x_linspace = np.round(np.linspace(lowx, highx, num=lenx, endpoint=True), decimals=3)

    data_array = []
    for index, row in logfile_dataframe.iterrows():
        filename = Path(prefix) / str(row['filename'])
        progx = row['progx']
        progy = row['progy']
        progz = row['progz']
        data = load_hisor_ibw(filename)
        data.attrs.update({'progx': progx, 'progy': progy, 'progz': progz})
        data_array.append(data)

    slit = data_array[0].arpes.downsample({'energy': 2, 'slit': 2}).slit
    energy = data_array[0].arpes.downsample({'energy': 2, 'slit': 2}).energy
    xa_empty = xr.DataArray(coords={'slit': slit, 'energy': energy, 'x': x_linspace},
                            dims=('slit', 'energy', 'x'))
    for data in data_array:
        progx = data.progx
        xa_empty.loc[{'slit': slit, 'energy': energy, 'x': progx}] = data.arpes.downsample({'energy':2,'slit':2})

    return xa_empty
