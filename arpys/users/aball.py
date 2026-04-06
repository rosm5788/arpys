import arpys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
plt.rcParams['figure.dpi'] = 150
plt.rcParams['image.cmap'] = 'inferno'
import os

def fit_gold_edge(gold_edge:xr.DataArray,p0=None):
    """
    :param gold_edge: xarray where one of the dimensions is energy, this function will sum over the rest
    :param p0: Initial guess for scipy curve fit: [ef,T,m,b,y0] for (m*(e-ef)+b)/(e^(e-ef/kbT) + 1) + y0
    """
    def fermi_func(e,ef,T,m,b,y0):
        kb = 8.617e-5
        return (m*(e-ef)+b)/(np.exp((e - ef) / (kb*T)) + 1) + y0
    gold_edge = gold_edge.sum(dim=(set(gold_edge.dims) - {'energy'}))
    gold_edge.plot(color='k')
    if p0 is None:
        p0 = [gold_edge.arpes.guess_ef(),50,(gold_edge[1]-gold_edge[0])/(gold_edge.energy[1]-gold_edge.energy[0]),gold_edge.max(),gold_edge.min()]
    params,covmatrix = curve_fit(fermi_func,gold_edge.energy,gold_edge.values,p0=p0)
    plt.plot(gold_edge.energy,fermi_func(gold_edge.energy,*params))
    plt.xlabel("Kinetic Energy (eV)")
    print(f"Ef = {params[0]:.3f} +- {covmatrix[0,0]**0.5:.3f} eV\nT = {params[1]:.1f} +- {covmatrix[1,1]**0.5:.1f} K")
    return params,np.sqrt(np.diag(covmatrix))

def parabola_arpes_fit(cut:xr.DataArray,kmin,kmax,emin,emax,p0,inner_exclude=0.0,energy_res=None,plot_results=False,exclude_side=None,do_edcs=False,edc_emin=-0.6,edc_kmin=None,edc_kmax=None,return_edcs=False,return_measurables=False):
    """
    :param p0: Parabola fit intitial guess: [x0,y0,a]
    """
    def parabola(x,x0,y0,a):
        return a*(x-x0)**2 + y0
    def parabola_plotter(x0,y0,a,xmin,xmax,ax=None,**plot_kwargs):
        points = np.linspace(xmin,xmax,100,True)
        if ax is not None:
            ax.plot(points,parabola(points,x0,y0,a),**plot_kwargs)
        else: plt.plot(points,parabola(points,x0,y0,a),**plot_kwargs)
    def lorentzian(x,x0,gamma,a,y0):
        return y0+a/((1+((x-x0)/gamma)**2))
    def exponential(x,n,beta,x0):
        return n*np.exp(-(x-x0)*beta)
    def fermi(x,T,ef):
        kb=8.617e-5
        return 1+np.exp((x-ef)/(kb*T))
    def VEC_function(x,C,T,ef,mu,sigma,gamma,a,n,beta,x0):
        return (C+exponential(x,n,beta,x0)+a*voigt_profile(x-mu,sigma,gamma))/(fermi(x,T,ef))
    def fit_lorentzian(data,center,window,return_all_params=True):
        fit_data = data.sel({data.dims[0]:slice(center-window,center+window)})
        y0_guess = fit_data.min()
        a_guess = fit_data.max() - fit_data.min()
        params,covmatrix = curve_fit(lorentzian,fit_data.coords[fit_data.dims[0]].values,fit_data.values,[center,window/2,a_guess,y0_guess])
        if not return_all_params:
            return params[0]
        else:
            return params, np.sqrt(np.diag(covmatrix))
    # This'll only work on k-converted slices
    data = cut.sel(binding=slice(emin,emax),kx=slice(kmin,kmax))
    parabola_points = []
    windows = [0.05,0.075,0.1]
    for i,energy in enumerate(data.coords['binding'].values): # This is the loop that goes through each MDC
        if exclude_side != 'left':
            left_slice = data[i].sel(kx=slice(-1e6,-inner_exclude+(kmax+kmin)/2))
            left_max = float(left_slice.coords['kx'][left_slice.argmax()].values)
            left_peaks = np.zeros(len(windows))
            for j in range(len(windows)): # Try a bunch of different window sizes around the max value
                try:
                    left_peaks[j] = fit_lorentzian(left_slice,left_max,windows[j],False)
                except RuntimeError: # If the fit fails, throw np.inf in as the fit peak center
                    left_peaks[j] = np.inf
            if np.sum(left_peaks == np.inf) < len(windows): # If all three didn't work, skip this mdc
                chosen_left_point = left_peaks[np.abs(left_peaks-left_max).argmin()] # Find which lorentzian had a peak closest to the max and use that
                if chosen_left_point < -inner_exclude+(kmax+kmin)/2: # Excludes weird points where the lorentzian is peaked outside of the chosen range
                    parabola_points.append([chosen_left_point,energy]) 
                    #print(energy,left_max,left_peaks)
        if exclude_side != 'right':
            right_slice = data[i].sel(kx=slice(inner_exclude+(kmax+kmin)/2,1e6))
            right_max = float(right_slice.coords['kx'][right_slice.argmax()].values)
            right_peaks = np.zeros(len(windows))
            for j in range(len(windows)):
                try:
                    right_peaks[j] = fit_lorentzian(right_slice,right_max,windows[j],False)
                except RuntimeError:
                    right_peaks[j] = np.inf
            if np.sum(right_peaks == np.inf) < len(windows): 
                chosen_right_point = right_peaks[np.abs(right_peaks-right_max).argmin()]
                if chosen_right_point > inner_exclude+(kmax+kmin)/2: # Excludes weird points where the lorentzian is peaked outside of the chosen range
                    parabola_points.append([chosen_right_point,energy])
                    #print(energy,right_max,right_peaks)
    mdc_parabola_points = np.array(parabola_points)
    if energy_res is None: # If you don't know the energy resolution, try getting it from the xarray, or just make something up
        try:
            energy_res = cut.attrs['Total Energy Res']
        except KeyError:
            print("Couldn't get energy resolution, assuming pixel width (not a good assumption)")
            energy_res = (data.coords['binding'][1] - data.coords['binding'][0])
    if do_edcs:
        if edc_kmin is None:
            edc_kmin = np.max(mdc_parabola_points[:,0][mdc_parabola_points[:,0] <= 0])
        if edc_kmax is None:
            edc_kmax = np.min(mdc_parabola_points[:,0][mdc_parabola_points[:,0] >= 0])
        edc_data = cut.sel(binding=slice(edc_emin,0.1),kx=slice(edc_kmin,edc_kmax))
        edc_parabola_points = []
        edc_fits = []
        for i,k_val in enumerate(edc_data.coords['kx'].values):
            edc_slice = edc_data[:,i]
            param_names = ("C","T","ef","mu","sigma","gamma","a","n","beta","x0")
            init_guess = (edc_slice.max()/2,100,-0.01,-0.1,energy_res+0.01,0.05,edc_slice.max()/5,edc_slice.max()/2,20,-0.8)
            fit_bounds = ((0,0,-0.1,-0.2,energy_res,0,0,0,0,-np.inf),
                          (np.inf,200,0.1,0,1,1,np.inf,np.inf,np.inf,0))
            edc_params,edc_covmatrix=curve_fit(VEC_function,edc_slice.coords['binding'].values,edc_slice.values,init_guess,bounds=fit_bounds,maxfev=20000)
            #print(k_val,edc_params[3])
            if edc_params[3] < emin +0.05 and edc_params[3] < -0.01:
                edc_parabola_points.append([k_val,edc_params[3]])
                parabola_points.append([k_val,edc_params[3]])
                edc_fits.append(np.concatenate(([k_val],edc_params)))
        edc_fits = np.array(edc_fits)
        edc_parabola_points = np.array(edc_parabola_points)
    parabola_points = np.array(parabola_points)
    try:
        parabola_fit_bounds = ((kmin,-np.inf,0),(kmax,0,np.inf))
        params,covmatrix = curve_fit(parabola,parabola_points[:,0],parabola_points[:,1],p0,bounds=parabola_fit_bounds,sigma=energy_res,absolute_sigma=True)
    except RuntimeError:
        print("Scipy parabola fit failed, take a look at the points it was working with:")
        fig, ax = plt.subplots()
        plot_data = cut.sel(binding=slice(min(parabola_points[:,1])-0.01,emax),kx=slice(kmin,kmax))
        plot_data.plot(robust=True)
        if do_edcs:
            ax.scatter(edc_parabola_points[:,0],edc_parabola_points[:,1],s=10,color='blue')
        ax.scatter(mdc_parabola_points[:,0],mdc_parabola_points[:,1],s=10,color='red')
        plt.show()
        return parabola_points
    uncerts = np.sqrt(np.diag(covmatrix))
    if plot_results:
        fig, ax = plt.subplots()
        plot_data = cut.sel(binding=slice(min(parabola_points[:,1])-0.01,emax),kx=slice(kmin,kmax))
        plot_data.plot(robust=True)
        if do_edcs:
            ax.scatter(edc_parabola_points[:,0],edc_parabola_points[:,1],s=10,color='blue',label='EDC Fits')
        ax.scatter(mdc_parabola_points[:,0],mdc_parabola_points[:,1],s=10,color='red',label='MDC Fits')
        parabola_plotter(*params,xmin=kmin,xmax=kmax,ax=ax,color='magenta')
        plt.xlim(kmin,kmax)
        #plt.ylim(emin-.05,emax)
        plt.xlabel("$\\rm k_x~(\\AA^{-1})$")
        plt.ylabel("$\\rm E - E_F$ (eV)")
        plt.legend()
        plt.show()

    hbar = 6.582e-16 # ev*s
    c = 2.998e8 # m/s
    m_eff = (hbar**2/(2*params[2]))*(1e20) * c**2/511e3 # In electron masses
    m_eff_uncert = uncerts[2] * (hbar**2/(2*(params[2])**2))*(1e20) * c**2/511e3

    k_fermi = np.sqrt(-params[1]/params[2])
    dkfermi_dy0 = -1/(2*params[2]*k_fermi)
    dkdfermi_da = params[1]/(2*params[2]**2*k_fermi)
    kf_uncert = np.sqrt(dkfermi_dy0**2*uncerts[1]**2 + dkdfermi_da**2*uncerts[2]**2 + 2*dkfermi_dy0*dkdfermi_da*covmatrix[1,2])
    print(f"Fit Band Bottom: {params[1]:.4f} ± {uncerts[1]:.4f} eV\nk Fermi: {k_fermi:.4f} ± {kf_uncert:.4f} A^-1\nBand Mass: {m_eff:.4f} ± {m_eff_uncert:.4f} m_e")
    if return_edcs:
        return params, uncerts, edc_fits
    elif return_measurables: # Returns BB, kf, m_eff rather than parabola fit parameters
        return (params[1],k_fermi,m_eff), (uncerts[1],kf_uncert,m_eff_uncert)
    else: return params, uncerts

def plot_nickelate_BZ(kx0=0,ky0=0,angle=0,a=0.8,b=None,ax=None,text_color='k'):
    if b is None: # If the height of the brillouin zone isn't given, make it a square
        b=a
    theta = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(theta),-np.sin(theta)],
                                [np.sin(theta),np.cos(theta)]])
    square_points = np.array([[a,b],[a,-b],[-a,-b],[-a,b],[a,b]])
    diamond_points = np.array([[0,b],[a,0],[0,-b],[-a,0],[0,b]])
    for i in range(5):
        square_points[i] = rotation_matrix @ square_points[i]
        diamond_points[i] = rotation_matrix @ diamond_points[i]
    square_points = square_points + np.array([kx0,ky0])
    diamond_points = diamond_points + np.array([kx0,ky0])
    brillouin_points = np.array([[kx0,ky0],diamond_points[2],(diamond_points[2]+diamond_points[1])/2,square_points[1]])
    brillouin_labels = ("$\\rm \\Gamma$","M","X","$\\rm \\Gamma^{\\prime}$")
    if ax is not None:
        ax.plot(square_points[:,0],square_points[:,1],color='darkblue')
        ax.plot(diamond_points[:,0],diamond_points[:,1],color='black')
        ax.scatter(brillouin_points[:,0],brillouin_points[:,1],color='forestgreen')
        for i in range(4):
            ax.text(brillouin_points[i,0]+0.05,brillouin_points[i,1]-0.1,brillouin_labels[i],color=text_color)
        ax.set_aspect(1)
        return ax
    else:
        plt.plot(square_points[:,0],square_points[:,1],color='darkblue')
        plt.plot(diamond_points[:,0],diamond_points[:,1],color='black')
        plt.scatter(brillouin_points[:,0],brillouin_points[:,1],color='forestgreen')
        for i in range(4):
            plt.text(brillouin_points[i,0]+0.05,brillouin_points[i,1]-0.1,brillouin_labels[i],color=text_color)
        plt.gca().set_aspect(1)
        return plt.gca()

def symmetrize_spectrum(spectrum:xr.DataArray,half_to_use='positive',dim='kx'):
    original_dims = list(spectrum.dims)
    if len(spectrum.shape) == 3:
        is_3D = True
    else:
        is_3D = False
    original_attrs = spectrum.attrs
    spectrum = spectrum.transpose(dim,...)
    first_nonzero = np.where(spectrum.coords[dim] > 0)[0][0]
    if half_to_use in ['right','positive','pos']:
        axis = np.append(-spectrum.coords[dim].values[-1:first_nonzero-1:-1],spectrum.coords[dim].values[first_nonzero:])
        if is_3D:
            spectrum_sym = np.zeros((len(axis),spectrum.shape[1],spectrum.shape[2]))
        else:
            spectrum_sym = np.zeros((len(axis),spectrum.shape[1]))
        spectrum_sym[len(axis)//2:] = spectrum[first_nonzero:]
        spectrum_sym[0:len(axis)//2] = spectrum[len(spectrum)-1:first_nonzero-1:-1]
    elif half_to_use in ['left','neg','negative']:
        axis = np.append(spectrum.coords[dim].values[:first_nonzero],-spectrum.coords[dim].values[first_nonzero-1::-1])
        if is_3D:
            spectrum_sym = np.zeros((len(axis),spectrum.shape[1],spectrum.shape[2]))
        else:
            spectrum_sym = np.zeros((len(axis),spectrum.shape[1]))
        spectrum_sym[0:len(axis)//2] = spectrum[:first_nonzero]
        spectrum_sym[len(axis)//2:] = spectrum[first_nonzero-1::-1]
    if is_3D:
        spectrum_sym = xr.DataArray(spectrum_sym,{dim:axis,list(spectrum.dims)[1]:spectrum.coords[list(spectrum.dims)[1]],list(spectrum.dims)[2]:spectrum.coords[list(spectrum.dims)[2]]},attrs=original_attrs)
    else:
        spectrum_sym = xr.DataArray(spectrum_sym,{dim:axis,list(spectrum.dims)[1]:spectrum.coords[list(spectrum.dims)[1]]},attrs=original_attrs)
    return spectrum_sym.transpose(original_dims[0],original_dims[1],...)

def background_subtraction(spectrum:xr.DataArray,sample_range,dim='kx'):
    old_coords = list(spectrum.coords.keys())
    spectrum = spectrum.transpose(dim,...)
    background = spectrum.sel({dim:slice(sample_range[0],sample_range[1])}).mean(dim).values
    for i in range(len(spectrum)):
        spectrum[i] = spectrum[i] - background
    return spectrum.transpose(old_coords[0],old_coords[1]).clip(min=0)

def interpolate_spectrum(spectrum:xr.DataArray,N_interp:int=5,dims_to_interp=None): 
    """
    This function interpolates an n-dimensional xarray that's along two coordinates, ensuring the original points are in it
    
    :param N_interp: The number of points to be added between each point
    :param dims_to_interp: List of dimensions to interpolate over if you don't want them all
    """
    if dims_to_interp is None:
        dims_to_interp = [dim for dim in spectrum.coords if spectrum.coords[dim].size > 1]
    elif isinstance(dims_to_interp,str):
        dims_to_interp = [dims_to_interp]
    new_coords = {}
    for dim in dims_to_interp:
        original_points = spectrum.coords[dim].values
        new_points = np.zeros(len(original_points)+N_interp*(len(original_points)-1))
        for i in range(len(original_points)-1):
            new_points[i*(N_interp+1):(i+1)*(N_interp+1)+1] = np.linspace(original_points[i],original_points[i+1],N_interp+2,endpoint=True)
        new_coords[dim] = new_points
    new_spectrum = spectrum.interp(new_coords,method='cubic')
    return new_spectrum