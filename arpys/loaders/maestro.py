import numpy as np
import xarray as xr
xr.set_options(keep_attrs=True)
from astropy.io import fits
import arpys
import h5py
import os

def load_maestro_h5_XPS(filename):
    with h5py.File(filename) as file:
        spectra_name = list(file['1D_Data'].keys())[0]
        spectrum_data = file['1D_Data'][spectra_name]
        scale_offsets = file['1D_Data'][spectra_name].attrs['scaleOffset'] # This is a tuple of the initial axis value for (pixel,energy)
        scale_deltas = file['1D_Data'][spectra_name].attrs['scaleDelta'] # This is a tuple of the change in each pixel for (pixel,energy)

        spectrum_array = spectrum_data[:,0]
        attrs = load_maestro_h5_attrs(file)
        energy_vals = np.linspace(scale_offsets[0],scale_offsets[0]+scale_deltas[0]*(spectrum_array.shape[0]-1),spectrum_array.shape[0],True)
        spectrum_xarray = xr.DataArray(spectrum_array,{'energy':energy_vals},attrs=attrs)
    return spectrum_xarray

#For loading 1D XPS data
def load_maestro_fits_XPS(filename):
    with fits.open(filename) as fits_object:
        data_type = fits_object[1].data.dtype.names[-1]
        data = fits_object[1].data[data_type].T
        axis_length = len(data)

        tfields = str(fits_object[1].header['TFIELDS'])
        axis_names = fits_object[1].header['TDESC' + tfields]
        initial_axis_value = float(fits_object[1].header['TRVAL' + tfields])
        axis_delta = float(fits_object[1].header['TDELT' + tfields])

        axis_name = axis_names.replace("(","").replace(")","").split(",")[0]
        conv = {'pixel': 'slit', 'eV': 'energy', 'pixels': 'slit'}
        axis_name_converted = conv[axis_name] # I figure it's best not to hardcode calling it energy just in case
        dims = [axis_name_converted]
        
        coords = {}
        axis_full = np.linspace(initial_axis_value, (axis_length - 1)*axis_delta + initial_axis_value, num=axis_length)
        coords[axis_name_converted] = axis_full

        attrs = read_maestro_fits_attrs(fits_object)
        return xr.DataArray(data[:,0], dims=dims, coords=coords, attrs=attrs)

def load_maestro_h5_single(filename):
    with h5py.File(filename) as file:
        spectra_name = list(file['2D_Data'].keys())[0]
        spectrum_data = file['2D_Data'][spectra_name]
        scale_offsets = file['2D_Data'][spectra_name].attrs['scaleOffset'] # This is a tuple of the initial axis value for (pixel,energy)
        scale_deltas = file['2D_Data'][spectra_name].attrs['scaleDelta'] # This is a tuple of the change in each pixel for (pixel,energy)
        try:
            is_swept = {"S": True, "F": False}[spectra_name[0]]
        except:
            raise KeyError("Congratulations, you've discovered a new edge case! Please tell Alex about this")

        spectrum_array = spectrum_data[:,:,0]
        attrs = load_maestro_h5_attrs(file)

        if is_swept:
            energy_vals = np.linspace(scale_offsets[1],scale_offsets[1]+scale_deltas[1]*(spectrum_array.shape[0]-1),spectrum_array.shape[0],True)
            # This assumes each pixel is 0.045 deg in thetax and that the detector is centered at thetax=0
            slit_vals = np.linspace(-(spectrum_array.shape[1]-1)*0.045/2,(spectrum_array.shape[1]-1)*0.045/2,spectrum_array.shape[1],endpoint=True)
            spectrum_xarray = xr.DataArray(spectrum_array,{'energy':energy_vals,'slit':slit_vals},attrs=attrs)
        else:
            energy_vals = np.linspace(scale_offsets[0],scale_offsets[0]+scale_deltas[0]*(spectrum_array.shape[1]-1),spectrum_array.shape[1],True)
            # This assumes each pixel is 0.045 deg in thetax and that the detector is centered at thetax=0
            slit_vals = np.linspace(-(spectrum_array.shape[0]-1)*0.045/2,(spectrum_array.shape[0]-1)*0.045/2,spectrum_array.shape[0],endpoint=True)
            spectrum_xarray = xr.DataArray(spectrum_array,{'slit':slit_vals,'energy':energy_vals},attrs=attrs)
        spectrum_xarray = spectrum_xarray.transpose('slit','energy')
    return spectrum_xarray

# For loading a single cut, NOT FOR LOADING FERMI MAPS OR PHOTON ENERGY SCANS
def load_maestro_fits_single(filename):
    with fits.open(filename) as fits_object:
        data_type = fits_object[1].data.dtype.names[-1]
        data = fits_object[1].data[data_type].T

        tfields = str(fits_object[1].header['TFIELDS'])
        axis_names = fits_object[1].header['TDESC' + tfields]
        try:
            axis_lengths = eval(fits_object[1].header['TDIM' + tfields])
        except KeyError:
            print("This might have failed because this data is 1D, try the XPS loader")
            raise
        initial_axis_values = eval(fits_object[1].header['TRVAL' + tfields])
        axis_deltas = eval(fits_object[1].header['TDELT' + tfields])
        axis_names_list = axis_names.replace("(","").replace(")","").split(",")

        conv = {'pixel': 'slit', 'eV': 'energy', 'pixels': 'slit'}
        dims = []
        coords = {}

        zipped = zip(axis_names_list, axis_lengths, initial_axis_values, axis_deltas)

        for axis_name, axis_length, initial_axis_value, axis_delta in zipped:
            axis_name_converted = conv[axis_name]
            if axis_name_converted == "slit": # This assumes the center of slit is at thetax=0 and each pixel is 0.045 deg apart
                axis_full = np.linspace(-(axis_length-1)*0.045/2, (axis_length-1)*0.045/2, num=axis_length)
            else:
                axis_full = np.linspace(initial_axis_value, (axis_length - 1) * axis_delta + initial_axis_value, num=axis_length)
            dims.append(axis_name_converted)
            coords[axis_name_converted] = axis_full

        attrs = read_maestro_fits_attrs(fits_object)
        return xr.DataArray(data[:, :, 0], dims=dims, coords=coords, attrs=attrs)

def load_maestro_h5_map(filename): 
    with h5py.File(filename) as file:
        spectra_name = list(file['2D_Data'].keys())[0]
        map_data = file['2D_Data'][spectra_name]
        scale_offsets = file['2D_Data'][spectra_name].attrs['scaleOffset'] # This is a tuple of the initial axis value for (pixel,energy)
        scale_deltas = file['2D_Data'][spectra_name].attrs['scaleDelta'] # This is a tuple of the change in each pixel for (pixel,energy)
        try:
            is_swept = {"S": True, "F": False}[spectra_name[0]]
        except:
            raise KeyError("Congratulations, you've discovered a new edge case! Please tell Alex about this")

        map_array = np.zeros(map_data.shape, dtype=map_data.dtype)
        for i in range(map_data.shape[2]): # Apparently hdf5 reads chunked data really slowly so it goes WAY faster if you do it like this
            map_array[:,:,i] = map_data[:,:,i]

        try: # This checks whether or not it's a deflector or a beta compensated map
            perp_vals = file['0D_Data']['Slit Defl']
        except:
            perp_vals = file['0D_Data']['Beta']
        attrs = load_maestro_h5_attrs(file)

        if is_swept:
            energy_vals = np.linspace(scale_offsets[1],scale_offsets[1]+scale_deltas[1]*(map_array.shape[0]-1),map_array.shape[0],True)
            # This assumes each pixel is 0.045 deg in thetax and that the detector is centered at thetax=0
            slit_vals = np.linspace(-(map_array.shape[1]-1)*0.045/2,(map_array.shape[1]-1)*0.045/2,map_array.shape[1],endpoint=True)
            map_xarray = xr.DataArray(map_array,{'energy':energy_vals,'slit':slit_vals,'perp':perp_vals},attrs=attrs)
        else:
            energy_vals = np.linspace(scale_offsets[0],scale_offsets[0]+scale_deltas[0]*(map_array.shape[1]-1),map_array.shape[1],True)
            # This assumes each pixel is 0.045 deg in thetax and that the detector is centered at thetax=0
            slit_vals = np.linspace(-(map_array.shape[0]-1)*0.045/2,(map_array.shape[0]-1)*0.045/2,map_array.shape[0],endpoint=True)
            map_xarray = xr.DataArray(map_array,{'slit':slit_vals,'energy':energy_vals,'perp':perp_vals},attrs=attrs)
        map_xarray = map_xarray.transpose('perp','slit','energy')
    return map_xarray

# For reading fermi maps ONLY, NOT FOR LOADING INDIVIDUAL SPECTRA OR PHOTON ENERGY SCANS
def load_maestro_fits_map(filename, is_deflector=True):
    with fits.open(filename) as fits_object:
        data_type = fits_object[1].data.dtype.names[-1]
        data = fits_object[1].data

        # data is now a list of "fits records" each of which is a slice in slit_defl or theta
        tfields = str(fits_object[1].header['TFIELDS'])
        axis_names = fits_object[1].header['TDESC' + tfields]
        axis_names_list = axis_names.replace("(", "").replace(")", "").split(",")
        axis_lengths = eval(fits_object[1].header['TDIM' + tfields])
        initial_axis_values = eval(fits_object[1].header['TRVAL' + tfields])
        axis_deltas = eval(fits_object[1].header['TDELT' + tfields])

        conv = {'pixel': 'slit', 'eV': 'energy', 'pixels': 'slit'}
        dims = ['perp']
        coords = {}

        zipped = zip(axis_names_list, axis_lengths, initial_axis_values, axis_deltas)
        for axis_name, axis_length, initial_axis_value, axis_delta in zipped:
            axis_name_converted = conv[axis_name]
            if axis_name_converted == "slit": # This assumes the center of slit is at thetax=0 and each pixel is 0.045 deg apart
                axis_full = np.linspace(-(axis_length-1)*0.045/2, (axis_length-1)*0.045/2, num=axis_length)
            else:
                axis_full = np.linspace(initial_axis_value, (axis_length - 1) * axis_delta + initial_axis_value, num=axis_length)
            dims.append(axis_name_converted)
            coords[axis_name_converted] = axis_full

        fermi_map = []
        perp_vals = []
        if is_deflector:
            for single_record in data:
                fermi_map.append(single_record.field(data_type).T)
                perp_vals.append(single_record.field('Slit Defl'))
        else:
            for single_record in data:
                fermi_map.append(single_record.field(data_type).T)
                perp_vals.append(single_record.field('beta'))
        fermi_map = np.array(fermi_map)
        perp_vals = np.array(perp_vals)

        coords['perp'] = perp_vals
        attrs = read_maestro_fits_attrs(fits_object)
        return xr.DataArray(fermi_map, dims=dims, coords=coords, attrs=attrs)

def load_maestro_h5_hvscan(filename): # Use this photon energy scan loader for any data 2025 on
    with h5py.File(filename) as file:
        spectra_name = list(file['2D_Data'].keys())[0]
        hvscan_data = file['2D_Data'][spectra_name]
        scale_offsets = file['2D_Data'][spectra_name].attrs['scaleOffset'] # This is a tuple of the initial axis value for (pixel,energy)
        scale_deltas = file['2D_Data'][spectra_name].attrs['scaleDelta'] # This is a tuple of the change in each pixel for (pixel,energy)
        try:
            is_swept = {"S": True, "F": False}[spectra_name[0]]
        except:
            raise KeyError("Congratulations, you've discovered a new edge case! Please tell Alex about this")
        

        hvscan_array = np.zeros(hvscan_data.shape, dtype=hvscan_data.dtype)
        for i in range(hvscan_data.shape[2]): # Apparently hdf5 reads chunked data really slowly so it goes WAY faster if you do it like this
            hvscan_array[:,:,i] = hvscan_data[:,:,i]

        photon_energies = file['0D_Data']['mono_eV'][:]
        attrs = load_maestro_h5_attrs(file)

        if is_swept:
            energy_vals = np.linspace(scale_offsets[1],scale_offsets[1]+scale_deltas[1]*(hvscan_array.shape[0]-1),hvscan_array.shape[0],True)
            # This assumes each pixel is 0.045 deg in thetax and that the detector is centered at thetax=0
            slit_vals = np.linspace(-(hvscan_array.shape[1]-1)*0.045/2,(hvscan_array.shape[1]-1)*0.045/2,hvscan_array.shape[1],endpoint=True)
            hvscan_xarray = xr.DataArray(hvscan_array,{'energy':energy_vals,'slit':slit_vals,'photon_energy':photon_energies},attrs=attrs)
        else:
            energy_vals = np.linspace(scale_offsets[0],scale_offsets[0]+scale_deltas[0]*(hvscan_array.shape[1]-1),hvscan_array.shape[1],True)
            # This assumes each pixel is 0.045 deg in thetax and that the detector is centered at thetax=0
            slit_vals = np.linspace(-(hvscan_array.shape[0]-1)*0.045/2,(hvscan_array.shape[0]-1)*0.045/2,hvscan_array.shape[0],endpoint=True)
            hvscan_xarray = xr.DataArray(hvscan_array,{'slit':slit_vals,'energy':energy_vals,'photon_energy':photon_energies},attrs=attrs)
        hvscan_xarray = hvscan_xarray.transpose('photon_energy','slit','energy')
    return hvscan_xarray

# For reading hvscans ONLY, NOT FOR LOADING INDIVIDUAL SPECTRA OR FERMI MAPS
# Doesn't work on data from at least 05/2025 on
def load_maestro_fits_hvscan(filename):
    with fits.open(filename) as fits_object:
        data_type = fits_object[1].data.dtype.names[-1]
        data = fits_object[1].data

        # data is now a list of "fits records" each of which is a slice in photon energy
        tfields = str(fits_object[1].header['TFIELDS'])
        axis_names = fits_object[1].header['TDESC' + tfields]
        axis_names_list = axis_names.replace("(", "").replace(")", "").split(",")
        axis_lengths = eval(fits_object[1].header['TDIM' + tfields])
        initial_axis_values = eval(fits_object[1].header['TRVAL' + tfields])
        axis_deltas = eval(fits_object[1].header['TDELT' + tfields])

        conv = {'pixel': 'slit', 'eV': 'energy', 'pixels': 'slit'}
        dims = []
        coords = {}

        zipped = zip(axis_names_list, axis_lengths, initial_axis_values, axis_deltas)
        for axis_name, axis_length, initial_axis_value, axis_delta in zipped:
            axis_name_converted = conv[axis_name]
            if axis_name_converted == "slit": # This assumes the center of slit is at thetax=0 and each pixel is 0.045 deg apart
                axis_full = np.linspace(-(axis_length-1)*0.045/2, (axis_length-1)*0.045/2, num=axis_length)
            else:
                axis_full = np.linspace(initial_axis_value, (axis_length - 1) * axis_delta + initial_axis_value, num=axis_length)
            dims.append(axis_name_converted)
            coords[axis_name_converted] = axis_full

        hv_scan = []
        photon_energies = []
        # align in binding
        for single_record in data:
            photon_energy = single_record.field('mono_eV')
            single_scan_da = xr.DataArray(single_record.field(data_type).T, dims=dims, coords=coords)
            #single_scan_aligned = align_binding(single_scan_da, photon_energy)

            hv_scan.append(single_scan_da)
            photon_energies.append(photon_energy)

        attrs = read_maestro_fits_attrs(fits_object)

        # interpolate each slice onto common energy/angle axes
        hv_scan_interped = [hv_scan[0]]
        for scan_no in np.arange(1,len(hv_scan)):
            slice_interpolated = hv_scan[scan_no].interp_like(hv_scan[0], method='linear')
            hv_scan_interped.append(slice_interpolated)

        # Sort by increasing photon energy so image tool doesn't freak out
        zipped_photonscan = zip(hv_scan_interped,photon_energies)
        sorted_scan = sorted(zipped_photonscan, key=lambda x: x[1])
        hv_scan_sorted = []
        photon_energies_sorted = []
        for scan, photon_energy in sorted_scan:
            hv_scan_sorted.append(scan)
            photon_energies_sorted.append(photon_energy)

        # concatenate and return
        photon_energy_scan = xr.concat(hv_scan_sorted, 'photon_energy')
        photon_energy_scan.attrs = attrs
        return photon_energy_scan.assign_coords({'photon_energy': photon_energies_sorted})




def align_binding(single_dataarray, photon_energy):
    initial_binding_energies = single_dataarray.arpes.energy
    initial_kinetics = initial_binding_energies + photon_energy - 4.2

    single_dataarray_ke = single_dataarray.assign_coords({'energy': initial_kinetics})
    initial_ef_guess = photon_energy - 4.2
    maxkinetic = np.nanmax(initial_kinetics)
    actual_ef = single_dataarray_ke.sel({'energy': slice(initial_ef_guess-0.1, maxkinetic)}).arpes.guess_ef()

    aligned_binding_energies = initial_kinetics - actual_ef
    return single_dataarray_ke.assign_coords({'energy': aligned_binding_energies})


# For reading attributes from fits file
def read_maestro_fits_attrs(fits_object):
    photon_energy = np.float32(fits_object[0].header['BL_E'])
    cryostat_a = np.float32(fits_object[1].data['Cryostat_A'])
    cryostat_b = np.float32(fits_object[1].data['Cryostat_B'])
    cryostat_c = np.float32(fits_object[1].data['Cryostat_C'])
    cryostat_d = np.float32(fits_object[1].data['Cryostat_D'])
    write_time = fits_object[1].header['WRITE_T']
    x_pos = fits_object[0].header['PMOTOR0']
    y_pos = fits_object[0].header['PMOTOR1']
    z_pos = fits_object[0].header['PMOTOR2']
    theta_pos = fits_object[0].header['PMOTOR3']
    beta_pos = fits_object[0].header['PMOTOR4']
    phi_pos = fits_object[0].header['PMOTOR5']
    alpha_pos = fits_object[0].header['PMOTOR6']
    slit_deflect_pos = fits_object[0].header['PMOTOR9']

    attrs = {"BL Energy": photon_energy,
             "Cryostat A": cryostat_a,
             "Cryostat B": cryostat_b,
             "Cryostat C": cryostat_c,
             "Cryostat D": cryostat_d,
             "Write Time": write_time,
             "X": x_pos,
             "Y": y_pos,
             "Z": z_pos,
             "Theta": theta_pos,
             "Beta": beta_pos,
             "Phi": phi_pos,
             "Alpha": alpha_pos,
             "Slit Deflector": slit_deflect_pos}
    return attrs

def load_maestro_h5_attrs(h5_object): # Reads attributes from .h5 files
    attrs = {}
    try:
        attrs['Pre-Comment'] = h5_object['Comments']['PreScan'][:][0][0].decode("ascii")
    except KeyError:
        attrs['Pre-Comment'] = None
    try: # Check if there's a postscan comment
        attrs['Post-Comment'] = h5_object["Comments"]["PostScan"][:][0][0].decode("ascii")
    except KeyError:
        attrs['Post-Comment'] = None
    attrs['Start Time'] = h5_object['Headers']['Main'][:][5][2].decode("ascii").replace("'","")
    attrs['Photon Energy'] = float(h5_object['Headers']['Beamline'][:][0][2])
    try: # Swept and Fixed modes have different headers
        scan_attrs = h5_object['Headers']['DAQ_Swept'][:] # They added a few attributes between beamtimes so I couldn't go by index
        scan_attrs_dict = {scan_attrs[i][0].decode('ascii'):scan_attrs[i][2] for i in range(len(scan_attrs))}
        attrs['Lens Mode'] = scan_attrs_dict['SSlnm0'].decode('ascii').replace("'","")
        attrs['Analyzer Slit'] = (scan_attrs_dict['SS_ESlitN'] if 'SS_ESlitN' in scan_attrs_dict else scan_attrs_dict['SS_ESliN']).decode('ascii').replace("'","") # Why would they change this?
        attrs['Pass Energy'] = int(scan_attrs_dict['SSpe_0'])
        attrs['Swept Min Energy'] = float(scan_attrs_dict['SSe0_0'])
        attrs['Swept Max Energy'] = float(scan_attrs_dict['SSe1_0'])
        try:
            attrs['Analyzer Energy Res'] = float(scan_attrs_dict['SSer_0'])
        except KeyError:
            pass
    except KeyError:
        attrs['Lens Mode'] = h5_object['Headers']['DAQ_Fixed'][:][9][2].decode('ascii').replace("'","")
        attrs['Analyzer Slit'] = h5_object['Headers']['DAQ_Fixed'][:][7][2].decode('ascii').replace("'","")
        attrs['Pass Energy'] = int(h5_object['Headers']['DAQ_Fixed'][:][10][2])
        attrs['Fixed Energy'] = float(h5_object['Headers']['DAQ_Fixed'][:][12][2])
    
    if 'Analyzer Energy Res' not in attrs:
        try: # Fixed mode doesn't output energy res for some reason
            analyzer_slit_size = float(attrs['Analyzer Slit'][4:7]) # Grabs the number from the string
        except:
            analyzer_slit_size = np.nan # In case there's an edge case I don't know about
        attrs['Analyzer Energy Res'] = analyzer_slit_size/400 * attrs['Pass Energy'] # Formula for R4000 taken from https://www.helmholtz-berlin.de/pubbin/igama_output?modus=datei&did=147

    attrs['EPU Polarization'] = float(h5_object['Headers']['Beamline'][:][3][2])
    attrs['Exit Slit Vertical'] = float(h5_object['Headers']['Beamline'][:][44][2])
    attrs['Exit Slit Horizontal'] = float(h5_object['Headers']['Beamline'][:][46][2])
    attrs['EPU Harmonic'] = float(h5_object['Headers']['Beamline'][:][82][2])
    attrs['EPU Grating'] = h5_object['Headers']['Beamline'][:][81][3].decode('ascii')
    attrs['Beam Energy Res'] = float(h5_object['Headers']['Beamline'][:][15][2])
    attrs['Total Energy Res'] = np.sqrt(attrs['Beam Energy Res']**2 + attrs['Analyzer Energy Res']**2)

    for i in range(7):
        attrs[h5_object['Headers']['Motors_Logical'][:][i][3].decode('ascii')] = float(h5_object['Headers']['Motors_Logical'][:][i][2])
    attrs['Deflector Angle'] = float(h5_object['Headers']['Motors_Logical'][:][9][2])

    attrs['Cryostat A'] = h5_object['0D_Data']['Cryostat_A'][:]
    attrs['Cryostat B'] = h5_object['0D_Data']['Cryostat_B'][:]
    attrs['Cryostat C'] = h5_object['0D_Data']['Cryostat_C'][:]
    attrs['Cryostat D'] = h5_object['0D_Data']['Cryostat_D'][:]
    return attrs

def print_maestro_logbook(folder): # Goes through a folder, gets all the .h5 files and prints the pre and post-scan comments
    h5_files = [file for file in os.listdir(folder) if file.endswith(".h5")]
    if len(h5_files) == 0:
        raise OSError("No .h5 files found in that folder")
    for file in h5_files:
        with h5py.File(folder+"\\"+file,"r") as scan:
            print("---------------------------------------")
            try:
                data = scan["Comments"]["PreScan"]
                print(file,"taken at",data[:][0][2].decode("ascii"))
                print(data[:][0][0].decode("ascii"))
            except KeyError:
                print(file,"taken at",scan['Headers']['Main'][:][5][2].decode("ascii").replace("'",""))
            try: 
                data2 = scan["Comments"]["PostScan"]
                print(data2[:][0][0].decode("ascii"))
            except KeyError:
                pass