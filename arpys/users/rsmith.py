# Robert's functions
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# used to normalize each perp. cut of a polar map not to be used for quantitative analysis, 
# but is useful to more easily visualise FS shape and symmetry. Note: this uses an average
# normalization method, which handles hot pixels better than np.max does
def normalize_3D(map):
    coords_list = list(map.coords.keys())
    cuts = []
    if np.shape(map)[0] < 3:
        print("Dataset is less than 3 dimensions! You shall not pass!")
        return
    #slightly different procedure for hv vs mechanical map scans
    elif 'photon_energy' in coords_list:
        for hv in map['photon_energy'].values:
            perp_cut = map.sel({'photon_energy': hv}, method='nearest')
            area = perp_cut.binding.size * perp_cut.slit.size
            sum = perp_cut.sum(['binding','slit'],skipna=True)
            average = sum/area
            perp_cut_normed = perp_cut/average
            # append the normalized cut into the list
            cuts.append(perp_cut_normed)
        perp_normed = xr.concat(cuts, 'photon_energy')
        perp_normed = perp_normed.assign_coords({'photon_energy':map['photon_energy']})
    
    elif 'perp' in coords_list:
        for theta in map['perp'].values:
            perp_cut = map.sel(perp=theta, method='nearest')
            area = perp_cut.energy.size * perp_cut.slit.size
            #for some reason, summing over energy also brings back the original problem
            sum = perp_cut.sum(dim=['energy','slit'],skipna=True)
            average = sum/area
            perp_cut_normed = perp_cut/average
            # append the normalized cut into the list
            cuts.append(perp_cut_normed)
        # concatenate the entire list along 'perp' into an xarray
        perp_normed = xr.concat(cuts, 'perp')
        perp_normed = perp_normed.assign_coords({'perp':map['perp']})
    return perp_normed

def dewarp_spectrum(spectrum,cutoff_distance=0.1,return_fit=False):
    """
    Attempts to dewarp a spectrum collected with a straight slit
    by finding eF at each slit value, then fitting to a parabola
    and ignoring values larger than cutoff_distance (eV) away from initial guess.

    NOTES: 
        - coords are interpolated to match with each other after shifting
        - dewarped spectra is shifted back to kinetic based on the center of fitted parabola

    Args:
        spectrum (arpes object): a 2D data spectrum with coords {slit, energy}
        cutoff_distance (float): the distance in eV from the spectrum's eF_guess
            beyond which you'd like to ignore points
        return_fit (boolean)   : whether or not to return the fitted parabola, extracted
            eF values, and the slit_values used for the fit (to see which were ignored)

    Returns:
        dewarped spectra (arpes_object): a spectra with shifted
        ----- if return_fit == True -------
            parabola_func (function): function that returns the fitted parabola for a given angle
            slit_values_fit (list of floats): list of slit values used for the fitting
            new_efs (list of floats): list of extracted fermi energies for each slit EDC
    """
    temp_edcs = []
    new_efs = []
    slit_values_fit = []
    slit_values = spectrum.slit.values

    if spectrum.arpes.ef == None:
        spectrum.arpes.ef = spectrum.arpes.guess_ef()
        print(f"Dewarper: you haven't set Ef, I'm just guessing... perchance is Ef = {spectrum.arpes.ef}?")

    for angle in slit_values:
        edc = spectrum.sel(slit=slice(angle-0.1,angle+0.1)).sum('slit')
        edc = edc.sel(energy=slice(spectrum.arpes.ef - cutoff_distance,spectrum.arpes.ef + cutoff_distance))
        ef_guess = edc.arpes.guess_ef()
        if np.abs(spectrum.arpes.ef - ef_guess) > cutoff_distance/2:
            #print(f"skipping angle {angle}, Ef = {ef_guess} is too far from spectrum_ef = {spectrum.arpes.ef}")
            continue
        #print(f"Found Ef = {ef_guess} for slit value {angle}")

        new_efs.append(edc.arpes.guess_ef())
        slit_values_fit.append(angle)
        
    coeffs = np.polyfit(slit_values_fit, new_efs, 2)

    # Create a parabolic function from the coefficients
    parabola_func = np.poly1d(coeffs)
    print(f"I kept {len(slit_values_fit)}/{len(slit_values)} points.")
    #print(f"made parabola func with {coeffs}")
    for i,angle in enumerate(slit_values):
        fitted_ef = parabola_func(angle)
        edc = spectrum.sel(slit=angle,method='nearest')
        new_energies = edc.energy.values - fitted_ef + parabola_func(0)
        new_edc = edc.assign_coords(energy=new_energies)
        #print(f"finished angle {angle} with ef shift: {fitted_ef}")
        if i == 0:
            temp_edcs.append(new_edc)
        else:
            temp_edcs.append(new_edc.interp_like(temp_edcs[0]))
    if return_fit:
        return xr.concat(temp_edcs,'slit'), parabola_func, slit_values_fit,new_efs
    else:
        return xr.concat(temp_edcs,'slit')

# pass a k-space spectra converted using a regular meshgrid. then, tweak peak_scale and nrg_spacing
# so that the stacked lines dont peak too high and have an appropriate number of lines
def stack_lines(spectra,nrg_spacing = 0.02,peak_scale=0.2,fig=None,ax=None,**kwargs):
    if fig == None and ax == None:
        fig, ax = plt.subplots(1,1)
    max_nrg = np.max(spectra.coords['binding'].values)
    min_nrg = np.min(spectra.coords['binding'].values)
    nrg_range = np.linspace(min_nrg,max_nrg,int((max_nrg-min_nrg)/nrg_spacing),endpoint=True)
    for nrg in nrg_range:
        line = spectra.sel(binding=slice(nrg-nrg_spacing,nrg+nrg_spacing)).sum('binding')
        line = ((line-np.min(line))/(np.max(line))*peak_scale) + nrg
        ax.plot(spectra.coords['kx'].values,line,**kwargs)
    return fig,ax

# 2nd derivative functions: BOTH require exactly 2D datasets 3D seems unnecessary and
# computationally expensive.
def laplacian(data, bwx=5,bwy=5, w=1):
    from astropy.convolution import convolve, Gaussian2DKernel
    import numpy as np
    nums = data.values
    coords = list(data.coords)
    x = data.coords[coords[0]].values
    y = data.coords[coords[1]].values
    #astropy convolves to smooth data out before taking derivatives
    data_smth = convolve(nums, Gaussian2DKernel(x_stddev=bwx,y_stddev=bwy))
    # Laplacian simply sums the 2nd derivatives in x and y
    diff2 = np.abs(np.gradient(np.gradient(data_smth, x, axis=0), x, axis=0)) + \
        np.abs(w * w * np.gradient(np.gradient(data_smth, y, axis=1), y, axis=1))
    
    curvature = xr.DataArray(diff2, dims=data.dims,
                            coords=data.coords, attrs=data.attrs)
    return curvature

# Curvature function for ARPES data from paper below.
# Parameters are tricky and require some fidgeting still...
def cv2d(data, bwx=5, bwy=5, c1=0.001, c2=0.001, w=1):
    from astropy.convolution import convolve, Gaussian2DKernel
    import numpy as np
    coords = list(data.coords)
    x = data.coords[coords[0]].values
    y = data.coords[coords[1]].values
    data_smth = convolve(data, Gaussian2DKernel(x_stddev=bwx,y_stddev=bwy))
    dx = np.gradient(data_smth, axis=0)
    dy = np.gradient(data_smth, axis=1) * w
    d2x = np.gradient(np.gradient(data_smth, x, axis=0), x, axis=0)
    d2y = np.gradient(np.gradient(data_smth, y, axis=1), y, axis=1) * w * w
    dxdy = np.gradient(np.gradient(data_smth, x, axis=0), y, axis=1) * w

    # 2D curvature - https://doi.org/10.1063/1.3585113
    cv2d = (np.abs((1 + (c1*dx)**2)*c2*d2y - 2*(c1*c2*dx*dy)*dxdy) +
            np.abs((1 + (c2*dy)**2)*c1*d2x)) / np.abs(1 + (c1*dx)**2 + (c2*dy)**2)**1.5
    curvature = xr.DataArray(cv2d, dims=data.dims,
                            coords=data.coords, attrs=data.attrs)
    return curvature

#aligns a photon_energy scan so that Ef = 0 binding, to correct
#monochromator drift or similar effects
def align_hvscan(hvscan,inclusion_zone=0.02):
    hv_cuts = []
    last_ef = hvscan.rename({'binding':'energy'}).sel(photon_energy=0,method='nearest').sel(energy=slice(-1,0.2)).sum('slit').arpes.guess_ef()
    # fit each cut and align ef to 0 binding
    for hv in hvscan['photon_energy']:
        hv_cut = hvscan.sel({'photon_energy': hv}, method='nearest')
        print(np.min(hv_cut.coords['binding'].values),np.max(hv_cut.coords['binding']).values)
        nrg_max = np.min([np.max(hv_cut.coords['binding'].values)*inclusion_zone+last_ef,np.max(hv_cut.coords['binding'].values)])
        nrg_min = np.max([np.min(hv_cut.coords['binding'].values)*inclusion_zone+last_ef,np.min(hv_cut.coords['binding'].values)])

        edc = hv_cut.sel({'binding':slice(nrg_min,nrg_max)}).sel({'slit':slice(-15,15)}).sum('slit')
        #edc.plot(ax=ax[1])
        print("looking for eF in the range ",np.min(hv_cut.coords['binding'].values)*inclusion_zone+last_ef,np.max(hv_cut.coords['binding'].values)*inclusion_zone+last_ef)
        last_ef = edc.rename({'binding':'energy'}).arpes.guess_ef()
        print("ef found at ", last_ef)
        new_be = hv_cut['binding'] - last_ef
        hv_cuts.append(hv_cut.assign_coords({'binding': new_be}))

    hv_cuts_interped = []
    hv_cuts_interped.append(hv_cuts[0])
    for scan_no in np.arange(1,len(hv_cuts)):
        hv_cuts_interped.append(hv_cuts[scan_no].interp_like(hv_cuts[0]))

    hv_cuts_aligned = xr.concat(hv_cuts_interped, 'photon_energy')
    hv_cuts_aligned = hv_cuts_aligned.assign_coords({'photon_energy': hvscan['photon_energy']})
    return hv_cuts_aligned

def symmetrize_spectra(
    spectra: xr.DataArray,
    axis: str = 'slit',
    direction: str = 'positive'
) -> xr.DataArray:
    """
    Symmetrizes ARPES data along a specified axis by copying one side to the other.

    Parameters
    ----------
    spectra : xr.DataArray
        Input DataArray (2D or 3D) with 'slit' or 'perp' coordinate.
    axis : str
        Axis to symmetrize across. Must be 'slit' or 'perp'.
    direction : str
        Which side to preserve and copy from: 'positive' (default) or 'negative'.

    Returns
    -------
    xr.DataArray
        Symmetrized DataArray.
    """
    if axis not in spectra.coords:
        raise ValueError(f"Axis '{axis}' not found in coordinates.")
    if direction not in ['positive', 'negative']:
        raise ValueError("direction must be 'positive' or 'negative'")

    coords = spectra.coords[axis].values
    new_coords = []

    for coord in coords:
        if direction == 'positive' and coord > 0:
            new_coords.append(-1*coord)
            new_coords.append(coord)
        elif direction == 'negative' and coord < 0:
            new_coords.append(-1*coord)
            new_coords.append(coord)
        elif coord == 0:
            new_coords.append(coord)
    
    new_coords.sort()
    result = spectra.reindex({axis:new_coords})


    for i, val in enumerate(new_coords):
        if (direction == 'positive' and val > 0) or (direction == 'negative' and val < 0):
            mirror_val = -val
            j = (np.abs(new_coords - mirror_val)).argmin()
            #print(f"copying slit from index {i} to {j}")
            # use isel-based safe assignment
            src = result.isel({axis: i})
            result[{axis:j}] = src
            result[{axis:i}] = src

    return result

def symmetrize_quadrant_3d(
    spectra: xr.DataArray,
    keep_quadrant: tuple[str, str] = ('positive', 'positive'),
) -> xr.DataArray:
    """
    Symmetrizes a 3D ARPES map by keeping one quadrant of (slit, perp) and mirroring
    into the other three quadrants.

    Parameters
    ----------
    spectra : xr.DataArray
        Must be 3D and have both 'slit' and 'perp' coordinates.
    keep_quadrant : tuple of str
        Which quadrant to keep, e.g. ('positive', 'positive') means
        keep (slit > 0, perp > 0). Other options are combinations of
        'positive' and 'negative'.
    skip_zero : bool
        If True, prevents overwriting slit=0 or perp=0 values.

    Returns
    -------
    xr.DataArray
        Fully symmetrized 3D spectra.
    """
    if spectra.ndim != 3:
        raise ValueError("Quadrant symmetrization only applies to 3D data.")
    if 'slit' not in spectra.coords or 'perp' not in spectra.coords:
        raise ValueError("3D spectra must include 'slit' and 'perp' coordinates.")

    slit_dir, perp_dir = keep_quadrant
    if slit_dir not in ['positive', 'negative'] or perp_dir not in ['positive', 'negative']:
        raise ValueError("Each entry in keep_quadrant must be 'positive' or 'negative'")


    slit_vals = spectra.coords['slit'].values
    perp_vals = spectra.coords['perp'].values

    new_slit=[]
    new_perp=[]
    for coord in slit_vals:
        if slit_dir == 'positive' and coord > 0:
            new_slit.append(-1*coord)
            new_slit.append(coord)
        elif slit_dir == 'negative' and coord < 0:
            new_slit.append(-1*coord)
            new_slit.append(coord)
        elif coord == 0:
            new_slit.append(coord)

    for coord in perp_vals:
        if perp_dir == 'positive' and coord > 0:
            new_perp.append(-1*coord)
            new_perp.append(coord)
        elif perp_dir == 'negative' and coord < 0:
            new_perp.append(-1*coord)
            new_perp.append(coord)
        elif coord == 0:
            new_perp.append(coord)
    
    new_perp.sort()
    new_slit.sort()

    slit_keep = np.array(new_slit) >= 0 if slit_dir == 'positive' else np.array(new_slit) <= 0
    perp_keep = np.array(new_perp) >= 0 if perp_dir == 'positive' else np.array(new_perp) <= 0

    result = spectra.reindex({'slit':new_slit,'perp':new_perp})

    for i, s_val in enumerate(new_slit):
        for j, p_val in enumerate(new_perp):
            if slit_keep[i] and perp_keep[j]:
                src = result.isel({'slit': i, 'perp': j})
                mirrors = [(-s_val, p_val), (s_val, -p_val), (-s_val, -p_val)]
                for s_mir, p_mir in mirrors:
                    i_mir = (np.abs(new_slit - s_mir)).argmin()
                    j_mir = (np.abs(new_perp - p_mir)).argmin()
                    result[{'slit':i_mir,'perp':j_mir}] = src
    return result

def map_k_reg_fast(arpes_obj, phi0=0, theta0=0, azimuth=0, slit_orientation=0,
                                           num_threads=None, background_threshold=None):
    """Parallelized map_k_reg conversion tool. Uses same paramenters, but breaks map into "chunks"
    which is interpolated in parallel via a faster (than scipy) trillinear interpolation. Trillinear 
    also appears to be more friendly to multithreading. By default, uses total available cores - 1."""
    from concurrent.futures import ThreadPoolExecutor
    import time

    def parallel_reverse_k_conversion(energy_grid, kx_grid, ky_grid, reverse_func, num_threads=8, **kwargs):
        slices = np.array_split(np.arange(energy_grid.shape[0]), num_threads)

        def process_chunk(indices):
            e = energy_grid[indices, :, :]
            kx = kx_grid[indices, :, :]
            ky = ky_grid[indices, :, :]
            return reverse_func(e, kx, ky, **kwargs)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(process_chunk, slices))

        alpha = np.concatenate([r[0] for r in results], axis=0)
        beta  = np.concatenate([r[1] for r in results], axis=0)
        energy = np.concatenate([r[2] for r in results], axis=0)
        return alpha, beta, energy

    def trilinear_interp(data, grid_coords, points):
        E, S, P = grid_coords
        e_idx = np.searchsorted(E, points[:, 0]) - 1
        s_idx = np.searchsorted(S, points[:, 1]) - 1
        p_idx = np.searchsorted(P, points[:, 2]) - 1

        e_idx = np.clip(e_idx, 0, len(E) - 2)
        s_idx = np.clip(s_idx, 0, len(S) - 2)
        p_idx = np.clip(p_idx, 0, len(P) - 2)

        de = (points[:, 0] - E[e_idx]) / (E[e_idx + 1] - E[e_idx])
        ds = (points[:, 1] - S[s_idx]) / (S[s_idx + 1] - S[s_idx])
        dp = (points[:, 2] - P[p_idx]) / (P[p_idx + 1] - P[p_idx])

        result = np.zeros_like(de)
        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    w = ((1 - de) if dx == 0 else de) * \
                        ((1 - ds) if dy == 0 else ds) * \
                        ((1 - dp) if dz == 0 else dp)
                    result += w * data[
                        e_idx + dx,
                        s_idx + dy,
                        p_idx + dz
                    ]
        return result

    def parallel_trilinear_interp(data, grid_coords, points, num_threads):
        chunks = np.array_split(points, num_threads)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(lambda chunk: trilinear_interp(data, grid_coords, chunk), chunks))
        return np.concatenate(results)

    if num_threads is None:
        import os
        num_threads = os.cpu_count() -1 

    assert hasattr(arpes_obj, 'arpes')
    assert arpes_obj.arpes.ef is not None, "Define Ef or risk being sent to the underworld."

    ef = arpes_obj.arpes.ef
    copy = arpes_obj.copy().transpose('energy', 'slit', 'perp')

    # Grid bounds
    kxmin, _ = copy.arpes.forward_k_conversion(np.nanmax(copy.energy.values), np.nanmin(copy.slit.values), 0,
                                               phi0, theta0, azimuth, slit_orientation)
    kxmax, _ = copy.arpes.forward_k_conversion(np.nanmax(copy.energy.values), np.nanmax(copy.slit.values), 0,
                                               phi0, theta0, azimuth, slit_orientation)
    _, kymin = copy.arpes.forward_k_conversion(np.nanmax(copy.energy.values), 0, np.nanmin(copy.perp.values),
                                               phi0, theta0, azimuth, slit_orientation)
    _, kymax = copy.arpes.forward_k_conversion(np.nanmax(copy.energy.values), 0, np.nanmax(copy.perp.values),
                                               phi0, theta0, azimuth, slit_orientation)

    kx_new = np.sort(np.linspace(kxmin, kxmax, num=copy.slit.size))
    ky_new = np.sort(np.linspace(kymin, kymax, num=copy.perp.size))
    energy_new = np.linspace(np.nanmin(copy.energy.values), np.nanmax(copy.energy.values), num=copy.energy.size)

    # Create output meshgrid
    energy_grid, kx_grid, ky_grid = np.meshgrid(energy_new, kx_new, ky_new, indexing='ij')

    # Reverse convert to experimental space
    t0 = time.time()
    alpha, beta, energy = parallel_reverse_k_conversion(
        energy_grid, kx_grid, ky_grid,
        reverse_func=copy.arpes.reverse_k_conversion,
        num_threads=num_threads,
        phi0=phi0, theta0=theta0, azimuth=azimuth, slit_orientation=slit_orientation
    )
    t1 = time.time()
    print(f"Point generation time: {t1 - t0:.3f}s")

    points = np.stack((energy.ravel(), alpha.ravel(), beta.ravel()), axis=-1)
    grid_coords = (copy.energy.values, copy.slit.values, copy.perp.values)

    # Interpolation (multithreaded)
    t2 = time.time()
    interpolated = parallel_trilinear_interp(copy.values, grid_coords, points, num_threads)
    t3 = time.time()
    print(f"Interpolation time: {t3 - t2:.3f}s")

    result = interpolated.reshape((len(energy_new), len(kx_new), len(ky_new)))

    # Optional background thresholding
    if background_threshold is None:
        flat = result.ravel()
        flat_nonzero = flat[flat > 0]
        background_threshold = np.quantile(flat_nonzero, 0.01) if flat_nonzero.size > 0 else 0
    print(f"Applying background threshold at {background_threshold:.3g}")
    result[result < background_threshold] = 0

    return xr.DataArray(result, dims=['binding', 'kx', 'ky'],
                        coords={'binding': energy_new - ef, 'kx': kx_new, 'ky': ky_new},
                        attrs=copy.attrs)