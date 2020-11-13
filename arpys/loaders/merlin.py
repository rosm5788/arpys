import numpy as np
import xarray as xr
# Requires the patched igor python module from Lanzara group
# pip install https://github.com/chstan/igorpy/tarball/712a4c4#egg=igor-0.3.1
import igor.igorpy as igor


def load_merlin_pxt_single(filename):
    conv = {'eV': 'energy', 'deg': 'slit'}

    #Read in using this byte-order, idk if this is going to work on everyone's machine
    main = igor.load(filename, initial_byte_order='>')
    igor_wave = main.children[1]

    dims = np.array(igor_wave.data).ndim
    axis_labels = []
    for axis in np.arange(dims):
        label_from_igor = igor_wave.axis_units[axis]
        try:
            axis_label = conv[label_from_igor]
            axis_labels.append(axis_label)
        except KeyError as e:
            print('Could not understand [{0}] as an arpys axis!'.format(label_from_igor))

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

# Input filenames must be a list of files that are to be loaded
# This method assumes that the map being loaded is a polar map
def load_merlin_pxt_map(filenames):
    singlescan_xarrays = []
    thetas = []
    for filename in filenames:
        one_slice = load_merlin_pxt_single(filename)
        one_slice_theta = float(one_slice.attrs['Polar'])
        singlescan_xarrays.append(one_slice)
        thetas.append(one_slice_theta)
    newcoord = {"perp":thetas}
    concat = xr.concat(singlescan_xarrays, 'perp')
    return concat.assign_coords(newcoord)

def load_merlin_pxt_photonescan_noalign(filenames):
    singlescan_xarrays = []
    photon_energies = []
    for filename in filenames:
        one_cut = load_merlin_pxt_single(filename)
        one_cut_hv = float(one_cut.attrs['BL Energy'])

        one_cut_be = -1*one_cut.arpes.energy
        one_cut = one_cut.assign_coords({'energy':one_cut_be})

        singlescan_xarrays.append(one_cut)
        photon_energies.append(one_cut_hv)

    zipped_object = zip(singlescan_xarrays, photon_energies)
    sorted_by_photon_energy = sorted(zipped_object, key=lambda x:x[1])

    scans_interped = []
    sorted_photon_energies = []

    scans_interped.append(sorted_by_photon_energy[0][0])
    sorted_photon_energies.append(sorted_by_photon_energy[0][1])

    for scan_no in np.arange(1,len(sorted_by_photon_energy)):
        scans_interped.append(sorted_by_photon_energy[scan_no][0].interp_like(sorted_by_photon_energy[0][0]))
        sorted_photon_energies.append(sorted_by_photon_energy[scan_no][1])

    total_scan = xr.concat(scans_interped, 'photon_energy')
    total_scan = total_scan.assign_coords({'photon_energy': sorted_photon_energies})
    return total_scan


# Input filenames must be a list of files that are to be loaded
# This method assumes that the map being loaded is a photon energy scan
# Each slice must be in binding energy prior to loading (this should be the default at merlin)
def load_merlin_pxt_photonescan(filenames):
    singlescan_xarrays = []
    photon_energies = []
    for filename in filenames:
        one_cut = load_merlin_pxt_single(filename)
        one_cut_hv = float(one_cut.attrs['BL Energy'])
        one_cut_be = one_cut.arpes.energy

        # Merlin uses positive binding energy to mean deeper in valence
        one_cut_ke = -1*one_cut_be + one_cut_hv - 4.2 #Fixed workfunction offset, won't introduce much error
        one_cut = one_cut.assign_coords({'energy':one_cut_ke})

        rough_ef = one_cut_hv - 4.2
        # guess ef using built in guess ef function. This is not perfect, need to think of something else
        ef_fit = one_cut.sel({'energy': slice(rough_ef-0.3, rough_ef+0.2)}).arpes.guess_ef()
        aligned_binding = one_cut_ke - ef_fit
        one_cut = one_cut.assign_coords({'energy':aligned_binding})
        singlescan_xarrays.append(one_cut)
        photon_energies.append(one_cut_hv)

    # We need to sort by photon energy so that we have a monotonically increasing photon energy axis at all times
    zipped_object = zip(singlescan_xarrays, photon_energies)
    sorted_by_photon_energy = sorted(zipped_object, key=lambda x: x[1])
    # Now need to align each xarray object onto a common binding energy grid (will use interpolation to be more precise)
    scans_interped = []
    scans_interped.append(sorted_by_photon_energy[0][0])
    sorted_photon_energies = []
    sorted_photon_energies.append(sorted_by_photon_energy[0][1])
    for scan_no in np.arange(1, len(sorted_by_photon_energy)):
        scans_interped.append(sorted_by_photon_energy[scan_no][0].interp_like(sorted_by_photon_energy[0][0]))
        sorted_photon_energies.append(sorted_by_photon_energy[scan_no][1])
    # Now concatenate them together and assign photon energy coordinate
    total_scan = xr.concat(scans_interped, 'photon_energy')
    total_scan = total_scan.assign_coords({'photon_energy':sorted_photon_energies})

    return total_scan


