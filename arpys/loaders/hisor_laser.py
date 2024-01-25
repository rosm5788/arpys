import xarray as xr
xr.set_options(keep_attrs=True)

import numpy as np
from pathlib import Path
import pandas as pd
from io import StringIO

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

# Reads the "Signal" part of the .txt. file which contains the Spin-EDCs for each detector
def read_signals(filename):
    with open(filename) as f:
        iterator = 0
        l = f.readline()
        while (not (l.startswith('[Signal'))):
            l = f.readline()  # Dump lines until you get to "Signal"
        l = f.readlines()[:-1]

    one_string = ""
    for line in l:
        one_string = one_string + line + '\n'

    #signals_list = []
    #for line in l:
    #    signal_split = np.array(line.strip().split("    "), dtype=float)
    #    signals_list.append(signal_split)

    #signals = np.array(signals_list)
    stringio = StringIO(one_string)
    signals_pd = pd.read_csv(stringio, delim_whitespace=True,header=None, names=['energy', 'white', 'black'], index_col=False)
    signals = signals_pd.to_numpy()
    return signals


# Reads the energy scale of the Spin-EDCs which is stored as "Dimension 1 scale" in the .txt file
def read_coords(filename):
    with open(filename) as f:
        for l in f:
            if l.startswith("Dimension 1 scale"):
                coord = np.array(l.split("=")[1].strip().split(" "), dtype=float)
    return coord

# Reads the "ThetaX" coordinate for spin mapping, scanning ThetaX will generate 2d cut
# Not actually needed anymore if attrs being read properly...
def read_thetax(filename):
    with open(filename) as f:
        for l in f:
            if l.startswith("ThetaX"):
                thetax = float(l.split("=")[1].strip())
    return thetax


# Reads single '.txt' spin EDC, stores three separate channels in Dataset as 'energy_readout', 'white', 'black'
# 'white' detector can be spin-x or spin-z, depending on set up
# 'black' detector can be spin-y or spin-z, depending on set up
# You need to remember magnetization for data, MAGNETIZATION IS NOT SET IN METADATA

def read_single_spin(filename):
    signals = read_signals(filename)
    signals = signals.T
    coord = read_coords(filename)
    #thetax = read_thetax(filename)
    attrs = read_metadata_txt(filename)
    dataset = xr.Dataset(
        {'energy_readout': ('energy', signals[0]), 'white': ('energy', signals[1]), 'black': ('energy', signals[2])},
        coords=dict(energy=coord), attrs=attrs)
    return dataset

def read_metadata_txt(filename):
    attrs = {}
    with open(filename) as f:
        for l in f:
            if len(l) == 0:
                continue
            split = l.split('=')
            if len(split) == 1:
                continue
            elif len(split) > 2:
                split = [split[0], '='.join(split[1:])]
            key, val = split
            attrs[key] = val.strip()
    return attrs

# Needs sorted filelist which is sorted by ThetaX to generate properly
# TODO: Fix reading metadata and adding as attrs to Dataset object
def read_spin_map(filelist):
    thetaxs = []
    data_array = []
    for scan in filelist:
        single = read_single_spin(scan)
        thetaxs.append(float(single.ThetaX.strip()))
        data_array.append(single)
    thetax_variable = xr.Variable('ThetaX', thetaxs)
    full_concat = xr.concat(data_array, thetax_variable)
    return full_concat

# Will read full set of spin map scans regardless of polarity, and will figure out putting together + and - scans
def read_spin_map_full(filelist):
    thetaxs_positive = []
    thetaxs_negative = []
    data_array_positive = []
    data_array_negative = []
    for scan in filelist:
        single = read_single_spin(scan)
        if single.polarity == "+":
            thetaxs_positive.append(float(single.ThetaX.strip()))
            if single.SpinChannel == "White":
                data_array_positive.append(single['white'])
            elif single.SpinChannel == "Black":
                data_array_positive.append(single['black'])
        elif single.polarity == "-":
            thetaxs_negative.append(float(single.ThetaX.strip()))
            if single.SpinChannel == "White":
                data_array_negative.append(single['white'])
            elif single.SpinChannel == "Black":
                data_array_negative.append(single['black'])
        else:
            print("polarity wasn't read, please check if you are reading 'modified' files")

    if len(thetaxs_positive) > 0 and len(thetaxs_negative) > 0:
        thetax_variable_positive = xr.Variable('ThetaX', thetaxs_positive)
        concat_positive = xr.concat(data_array_positive, thetax_variable_positive)
        thetax_variable_negative = xr.Variable('ThetaX', thetaxs_negative)
        concat_negative = xr.concat(data_array_negative, thetax_variable_negative)
        full_dataset = xr.Dataset(data_vars=dict(positive=concat_positive,
                                                 negative=concat_negative))

    elif len(thetaxs_positive) > 0 and len(thetaxs_negative) == 0:
        thetax_variable_positive = xr.Variable('ThetaX', thetaxs_positive)
        concat_positive = xr.concat(data_array_positive, thetax_variable_positive)
        full_dataset = xr.Dataset(data_vars=dict(positive=concat_positive))

    elif len(thetaxs_negative) > 0 and len(thetaxs_positive) == 0:
        thetax_variable_negative = xr.Variable('ThetaX', thetaxs_negative)
        concat_negative = xr.concat(data_array_negative, thetax_variable_negative)
        full_dataset = xr.Dataset(data_vars=dict(negative=concat_negative))
    else:
        print('by jove everything be empty')
        full_dataset = 0
    return full_dataset

# Normalize spin maps - pass in positive and negative channels explicitly - Dataset['white'] or Dataset['black']
# Returns tuple (Iup, Idown)
def normalize_spin_maps(positive_map, negative_map, sherman_coeff):

    #align positive to negative grid
    positive_aligned = positive_map.reindex_like(negative_map)

    A = (positive_aligned - negative_map) / (positive_aligned + negative_map)
    P = A / sherman_coeff
    Iup = (1 + P) * (positive_aligned + negative_map) * (1/2)
    Idown = (1 - P) * (positive_aligned + negative_map) * (1/2)

    return Iup, Idown
