# Robert's functions
import xarray as xr
import numpy as np

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
