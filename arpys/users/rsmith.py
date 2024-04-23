# Robert's functions
import xarray as xr

# used to normalize each perp. cut of a polar map
# not to be used for quantitative analysis, but is
# useful to more easily visualise FS shape and symmetry
def normalize_mech_map(polar_map):
    # make a list of each cut after normalization
    perp_cuts = []
    for theta in polar_map['perp']:
        perp_cut = polar_map.sel({'perp': theta}, method='nearest')
        sum = perp_cut.sum('slit').sum('energy')
        area = perp_cut.energy.size * perp_cut.slit.size
        average = sum/area
        perp_cut_normed = perp_cut/average
        # append the normalized cut into the list
        perp_cuts.append(perp_cut_normed)
    # concatenate the entire list along 'perp' into an xarray
    perp_normed = xr.concat(perp_cuts, 'perp')
    perp_normed = perp_normed.assign_coords({'perp':polar_map['perp']})
    return perp_normed