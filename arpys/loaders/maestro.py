import numpy as np
import xarray as xr
from astropy.io import fits
import arpys


# For loading a single cut, NOT FOR LOADING FERMI MAPS OR PHOTON ENERGY SCANS
def load_maestro_fits_single(filename):
    with fits.open(filename) as fits_object:
        data_type = fits_object[1].data.dtype.names[-1]
        data = fits_object[1].data[data_type].T

        tfields = str(fits_object[1].header['TFIELDS'])
        axis_names = fits_object[1].header['TDESC' + tfields]
        axis_lengths = eval(fits_object[1].header['TDIM' + tfields])
        initial_axis_values = eval(fits_object[1].header['TRVAL' + tfields])
        axis_deltas = eval(fits_object[1].header['TDELT' + tfields])
        axis_names_list = axis_names.replace("(","").replace(")","").split(",")

        conv = {'pixel': 'slit', 'eV': 'energy', 'pixels': 'slit'}
        dims = []
        coords = {}

        zipped = zip(axis_names_list, axis_lengths, initial_axis_values, axis_deltas)

        for axis_name, axis_length, initial_axis_value, axis_delta in zipped:
            axis_name_converted = conv[axis_name]
            if axis_name_converted == "slit":
                axis_full = np.linspace(-15, axis_length*0.045 - 15, num=axis_length)
            else:
                axis_full = np.linspace(initial_axis_value, axis_length*axis_delta + initial_axis_value, num=axis_length)
            dims.append(axis_name_converted)
            coords[axis_name_converted] = axis_full

        attrs = read_maestro_fits_attrs(fits_object)
        return xr.DataArray(data[:, :, 0], dims=dims, coords=coords, attrs=attrs)

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
            if axis_name_converted == "slit":
                axis_full = np.linspace(-15, axis_length * 0.045 - 15, num=axis_length)
            else:
                axis_full = np.linspace(initial_axis_value, axis_length * axis_delta + initial_axis_value, num=axis_length)
            dims.append(axis_name_converted)
            coords[axis_name_converted] = axis_full

        fermi_map = []
        perp_vals = []
        if is_deflector:
            for single_record in data:
                fermi_map.append(single_record.field(data_type).T)
                perp_vals.append(single_record.field('Slit Defl'))
        else: # Need to make sure this is working. I need a beta map or something to show if it works
            for single_record in data:
                fermi_map.append(single_record.field(data_type).T)
                perp_vals.append(single_record.field('beta'))
        fermi_map = np.array(fermi_map)
        perp_vals = np.array(perp_vals)

        coords['perp'] = perp_vals
        attrs = read_maestro_fits_attrs(fits_object)
        return xr.DataArray(fermi_map, dims=dims, coords=coords, attrs=attrs)


# For reading hvscans ONLY, NOT FOR LOADING INDIVIDUAL SPECTRA OR FERMI MAPS
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
            if axis_name_converted == "slit":
                axis_full = np.linspace(-15, axis_length * 0.045 - 15, num=axis_length)
            else:
                axis_full = np.linspace(initial_axis_value, axis_length * axis_delta + initial_axis_value, num=axis_length)
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

