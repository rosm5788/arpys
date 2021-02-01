from .core import Core
from typing import List, Dict, Set, Tuple, NewType
import xarray as xr
import numpy as np
from scipy.special import wofz
from lmfit import minimize, Parameters
import re
import copy

xarray = NewType('xarray', xr)
minnimizer_result = NewType('minnimizer_result', object)


class MDC_Fit_Parameter_Mislabel(Exception):
    pass


def Voigt(x, center, alpha, gamma, amplitude):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))

    return amplitude*np.real(wofz(((x-center) + 1j*gamma)/sigma/np.sqrt(2)))/sigma/np.sqrt(2*np.pi)


def Gaussian(x, center, std, amplitude):
    """
    Return Gaussian, centered at center, with standard deviation—std, and an Amplitude.
    """
    return (amplitude/(std*np.sqrt(2*np.pi)))*np.exp((-1/2)*(((x-center)/std)**2))


def Lorentzian(x, center, gamma, amplitude):
    return (amplitude/(2*np.pi))*gamma/(np.square(x-center) + np.square(gamma)/4)


@xr.register_dataarray_accessor("MDC")
class MDC(Core):

    def __init__(self, xarray_obj: xarray):
        super().__init__(xarray_obj=xarray_obj)  # init CORE
        self.params = Parameters()  # parameters of the fit. Instance of lmfit parameters object.
        self._fit_ROI_results = []  # List to be populated with minimizer_results objects from fits using lmfit minimize

        # Dict to store crop regions and corresponding fit results, if image is being fit piecewise.
        self._fit_results_stored = {'cr_bounds': [], 'fit_ROI_results': []}

        # Controls whether to use default bounds or not. Make True if you do not want me to change bounds ever.
        self._no_default_bounds = False
        # This parameter if true, makes the fitting process use the best fit params of the previous MDC fit as the
        # initial guess of the parameters to fit the next MDC. True is usually a good idea.
        self._previous_fit_as_guess_next_fit = True
        self._constant_key = None  # Key used by the user for the constant, since my code allows some flexibility.


        self._param_keys = []  # A list of parameter names populated after parameter names are confirmed correct.
        self._constants = []  # A list of fitted constants populated after fit.

        """
        A list of the function names + label (str_key synonomous). For each, there should also be all associated params 
        to complete parameterization of function. i.e. Suppose user defined 'Voigt_1_center', then my code attempts to 
        ensure there is also a 'Voigt_1_alpha', 'Voigt_1_gamma', and a 'Voigt_1_amplitude'. And this list will be 
        populated with a string 'Voigt_1_'. This population is done when self.prepare_fit_params() is run, which is 
        automatically done at the start of the self.fit_ROI() method. This is useful, because these keys fully describe 
        the model in less strings than all of the param names; so quicker loops.
        """
        self._function_key_labels = []

        # Automatically set crop region to exclude area above Ef, since MDCs there should not be fit.
        self.cr_bounds = [np.amin(self._obj.kx.values), np.amax(self._obj.kx.values),
                          np.amin(self._obj.binding.values), 0]

        """
        The following dict objects all have keys that are functions + str_key assigned by user, similar to 
        self._function_key_labels; however, the following dicts do not have keys for constants and also neglect 
        functions with str_key = 'bg', 'back_ground' ... (see self.extract_peaks method for if statement that explicitly
        shows which str_key labels are neglected). These are neglected because we do not want vf, or poly fits to 
        background peaks.
        """
        self._MDC_peaks = {}  # returns peak locations for each fit MDC populated by self.extract_peaks method

        # returns dict[key] = numpy.poly1d that represents dispersion. These objects can be fed a k value, and will
        # spit out E(k).
        self._dispersion = {}

        # kf and vf will be populated when self.extract_dispersion is run, IF the crop region contains Ef or has an
        # upper bound that is within 100 meV of Ef.
        # returns dict[key] = nd.array of vf in units of [binding]/[kx]
        self._vf = {}
        # returns dict[key] = nd.array of kf in units of [kx]
        self._kf = {}
        return

    @staticmethod
    def help(params_only: bool = False):
        if not params_only:
            print("Welcome! Hints for using this MDC extension to the xarray:")
            print("This extention is used to fit all MDCs in crop_region (self.cr) of an ARPES E-k image. At present, "
                  "the model used to fit MDCs is the same for all MDCs within the crop_region, i.e. not dynamic.")
            print("You define a model implicitly by adding parameters to xarray.MDC.params, which is an instance of an"
                  " lmfit Params object (see their docs on how to add parameters etc.). Your model will be a "
                  "linear combination of a constant and any number of Voigts, Gaussians, and Lorentzians that you want."
                  " You add one of these functions to your model by adding a parameter named 'constant' or "
                  "'Voigt'/'Gaussian'/'Lorentzian' + (arbitrary string that acts as a label) + 'center'. For each "
                  "parameter name of this type, the model will contain the appropriate function. For example, lets say "
                  "you add parameters with the following names: 'constant', 'Voigt_1_center', and 'Gaussian_center', "
                  "then your model is a linear combination of 1 constant, 1 Voigt function with the label '_1_' and "
                  "1 Gaussian with the label '_'. Clearly Gaussians, Lorentzians, and Voigts have parameters other than"
                  "the location of their peaks == 'center', and each function needs all of the appropriate parameters;"
                  "so for every [function name (e.g. 'Voigt') + label + 'center'], there must also be (function name)"
                  " + label (SAME Label) + parameter for all parameter strings associated with a function. Voigts have "
                  "parameters 'center', 'alpha', 'gamma', and 'amplitude'. Gaussians have a 'center', 'std', and "
                  "'amplitude'. Lorentzians have a 'center', 'gamma', and 'amplitude'. If you define a 'center' param "
                  "for any function, all other params will be created automatically when you try to fit if you have not"
                  " created them manually. Functions are added based on the presence of 'center' param, not any other "
                  "param; so, for example, do not create a 'Voigt_alpha' param unless there is a 'Voigt_center', but "
                  "you can create a 'Voigt_center' without the other params, e.g. 'Voigt_alpha'. Lastly, you may use "
                  "the special label 'bg', 'back_ground', or most other reasonable references to a back ground, which "
                  "excludes the function from being used to extract dispersion (E[k]) or an MDC peak, which is useful"
                  "for plotting reasons. Constants are automatically considered background, don't add bg, etc. to them")
            print("With a model specified, run xarray.MDC.fit_ROI, which fits the xarray.MDC.cr, which you can change "
                  "by running xarray.MDC.cr_bounds = [kmin, kmax, Emin, Emax] to crop to that domain for fitting. You "
                  "should also do any smoothing, down-sampling, etc. before running the fit. Then visualize your "
                  "results with plotting using .waterfall and .plot_arpes_image methods. At this point, you can print"
                  ".vf and .kf to find fermi velocities and fermi wave-vectors. Good Luck! Adjust initial guesses and "
                  "bounds appropriately to get good fits (see lmfit.Params())")
            print("Advanced: Two control parameters you might want to change, maybe: ._no_default_bounds is default "
                  "False, but if the algorithm is setting bounds you really do not want set, then set this to True. "
                  "Second, _previous_fit_as_guess_next_fit is default True, and as the name suggests, this feeds the "
                  "fit params of the model from previously fit MDC as initial guess into fitting the next MDC. Set this"
                  " to False if you do not want to do this, but I am not exactly what the guesses will be in that case")
        else:
            print("All parameters must be either 'constant' or a string with the following components: Function name "
                  "('Voigt', 'Gaussian', or 'Lorentzian') + label + parameter. label and parameter are stings. label "
                  "can be arbitrary string, this is your naming convention for each unique function. If label contains "
                  "'bg' then it will be treated as a part of fitting the background. Nothing changes during fitting "
                  "process (you probably need to select bounds and guesses to coerce this to fit the background during "
                  "fitting), but some analysis will be different, e.g. 'Gaussian_bg_center' params won't be treated as "
                  "peaks of MDCs, no polynomial fit of them will be found, and they won't be plotted as peaks on your"
                  "ARPES image. params are unique parameters of each function: Voigts have parameters 'center', 'alpha'"
                  ", 'gamma', and 'amplitude'. Gaussians have a 'center', 'std', and 'amplitude'. Lorentzians have a "
                  "'center', 'gamma', and 'amplitude'. For every unique Function name ('Voigt', 'Gaussian', or "
                  "'Lorentzian') + label combo you have, you need to have a 'center' param, at least, but you can also."
                  "add the other params manually if you like, which allows you to manually control bounds etc., but you"
                  "need to be consistent with the label for each function parameter.")
        print('I tried to program a bit of flexibility in parameter names, but there is not much. For example, function'
              'names can start with lower-case or capital, but not so for parameter names. constant can be Constant,'
              'DC_Offset, or many other reasonable names for a constant, and the special treatment of labels containing'
              'bg also happens if label contains back_ground or BackGround or other reasonable variants of that, but '
              'again no promises; so try to follow the spelling/conventions I explained above, and if you don"t, I will'
              'try to catch you and warn you. Good Luck!')
        return

    def fit_MDC(self, MDC: List[float], eps: List[float] = None, function_key_labels: List[str] = None):
        return minimize(self.residual, self.params, args=(self.cr.kx.values, MDC, eps, function_key_labels))

    @staticmethod
    def residual(params, kx, MDC: List[float] = None, eps: List[float] = None, function_key_labels: List[str] = None):
        """
        Return the residuals of the model (MDC-fit)*eps, if eps are supplied. If no MDC data is supplied, then
        just retun the fit. eps is the uncertainty on data, assumed 1 if not supplied.
        The model used is defined implicitly based on the params being used. Basically, your fit must be a linear
        combination of Gaussians, Voigts, Lorentzians, and a constant. For every Gaussian/Voigt/Lorentzian + str +
        'center' there will be a Gaussian/Voigt/Lorentzian function added to the model for fitting.
        The remaining parameters of each function type may be
        defined explicitly, otherwise, default initial guesses and bounds will be supplied.

        inputs: params is an instance of lmfit's Parameters() object, kx is the domain of the fit List or np.array of
        floats, MDC is the data to be fit (list or np.array of floats of the same length of kx, eps is the uncertainty,
        which should have the same length as MDC or constnat, haven't tested this. function_key_labels is a list of str
        that represent all of the functions comprising the fit. So, if you have a param 'Gaussian_center', then function
        key labels contains 'Gaussian_'.
        """
        parvals = params.valuesdict()
        fit = np.zeros(len(kx))
        if isinstance(function_key_labels, list) and function_key_labels:  # If I know these, build the model.
            # Only advantage is that key_labels has fewer elements. key_label is a str(function name)+str(some string
            # that acts as a label)
            for keys in function_key_labels:
                if keys == 'DC_offset' or keys == 'Constant' or keys == 'constant' or keys == 'Offset' or \
                        keys == 'offset' or keys == 'DC_Offset':
                    fit += np.ones(len(kx)) * parvals[keys]
                if 'Voigt' in keys or 'voigt' in keys:
                    try:
                        fit += Voigt(kx, parvals[keys+'center'],
                                     parvals[keys+'alpha'],
                                     parvals[keys+'gamma'], parvals[keys+'amplitude'])
                    except KeyError:
                        raise MDC_Fit_Parameter_Mislabel("Found a 'Voigt' + str_key, but you probably defined a "
                                                         "parameter of the Voigt function that did not follow the same "
                                                         "pattern of 'Voigt' + str_key + (lower-case parameter: "
                                                         "alpha, gamma, or amplitude).")
                if 'Gaussian' in keys or 'gaussian' in keys:
                    try:
                        fit += Gaussian(kx, parvals[keys+'center'],
                                     parvals[keys+'std'],
                                     parvals[keys+'amplitude'])
                    except KeyError:
                        raise MDC_Fit_Parameter_Mislabel("Found a 'Gaussian' + str_key, but you probably defined a "
                                                         "parameter of the Gaussian function that did not follow the "
                                                         "same pattern of 'Gaussian' + str_key + (lower-case parameter:"
                                                         " std or amplitude).")
                if 'Lorentzian' in keys or 'lorentzian' in keys:
                    try:
                        fit += Lorentzian(kx, parvals[keys + 'center'],
                                          parvals[keys + 'gamma'],
                                          parvals[keys + 'amplitude'])
                    except KeyError:
                        raise MDC_Fit_Parameter_Mislabel("Found a 'Lorentzian' + str_key, but you probably defined a "
                                                         "parameter of the Lorentzian function that did not follow the "
                                                         "same pattern of 'Lorentzian' + str_key + (lower-case "
                                                         "parameter: gamma or amplitude).")
        # Go through parameter names and find constants, Gaussians, Voigts, and Lorentzians by the definition of their
        # center parameter. Use all necesesary, similarly labeled parameters for the function type to add to the
        # function to the overall fit model.
        else:
            for keys in parvals:
                if keys == 'DC_offset' or keys == 'Constant' or keys == 'constant' or keys == 'Offset' or \
                        keys == 'offset' or keys == 'DC_Offset':
                    fit += np.ones(len(kx)) * parvals[keys]
                if ('Voigt' in keys or 'voigt' in keys) and 'center' in keys:
                    try:
                        label = re.search('Voigt(.+?)center', keys, re.IGNORECASE).group(1)
                    except AttributeError:
                        # did not find something between 'Voigt' and 'center'
                        raise MDC_Fit_Parameter_Mislabel("You defined a parameter containing voigt and center, but there "
                                                         "must be some character, any character, between voigt and center")
                    try:
                        fit += Voigt(kx, parvals[keys[0:5]+label+'center'],
                                     parvals[keys[0:5]+label+'alpha'],
                                     parvals[keys[0:5]+label+'gamma'], parvals[keys[0:5]+label+'amplitude'])
                    except KeyError:
                        raise MDC_Fit_Parameter_Mislabel("Found a 'Voigt' + str + 'center', but you probably defined a "
                                                         "parameter of the Voigt function that did not follow the same "
                                                         "pattern of 'Voigt' + str + (lower-case parameter: alpha, gamma, "
                                                         " or amplitude).")
                if ('Gaussian' in keys or 'gaussian' in keys) and 'center' in keys:
                    try:
                        label = re.search('Gaussian(.+?)center', keys, re.IGNORECASE).group(1)
                    except AttributeError:
                        # did not find something between 'Gaussian' and 'center'
                        raise MDC_Fit_Parameter_Mislabel("You defined a parameter containing gaussian and center, but there"
                                                         " must be some character, any character, between gaussian and "
                                                         "center")
                    try:
                        fit += Gaussian(kx, parvals[keys[0:8]+label+'center'],
                                     parvals[keys[0:8]+label+'std'],
                                     parvals[keys[0:8]+label+'amplitude'])
                    except KeyError:
                        raise MDC_Fit_Parameter_Mislabel("Found a 'Gaussian' + str + 'center', but you probably defined a "
                                                         "parameter of the Gaussian function that did not follow the same "
                                                         "pattern of 'Gaussian' + str + (lower-case parameter: std or "
                                                         "amplitude).")
                if ('Lorentzian' in keys or 'lorentzian' in keys) and 'center' in keys:
                    try:
                        label = re.search('Lorentzian(.+?)center', keys, re.IGNORECASE).group(1)
                    except AttributeError:
                        # did not find something between 'Gaussian' and 'center'
                        raise MDC_Fit_Parameter_Mislabel(
                            "You defined a parameter containing lorentzian and center, but there"
                            " must be some character, any character, between lorentzian and "
                            "center")
                    try:
                        fit += Lorentzian(kx, parvals[keys[0:10] + label + 'center'],
                                          parvals[keys[0:10] + label + 'gamma'],
                                          parvals[keys[0:10] + label + 'amplitude'])
                    except KeyError:
                        raise MDC_Fit_Parameter_Mislabel(
                            "Found a 'Lorentzian' + str + 'center', but you probably defined a "
                            "parameter of the Lorentzian function that did not follow the same "
                            "pattern of 'Lorentzian' + str + (lower-case parameter: std or "
                            "amplitude).")

        if MDC is None:
            return fit
        if eps is None:
            return fit - MDC
        return (fit - MDC)/eps

    def fit_ROI(self, eps: List[float] = None):
        """
        Main use of class: fit the preselected crop region's MDCs with a fit function.
        Take every MDC in the ROI and fit it with a specified fit function.
        Workflow: prior to calling this method, user should select the ROI, pre-smooth/down-sample the ARPES image.
        Then, call this function, which will fit every MDC with the same fit function; so if no downsampling nor
         cropping, then this function tries to fit something like 1000 MDCs, depending on image resolution, probably
        slow...
        inputs: eps needs to be an nd.array of the same shape as the ROI/...MDC.cr haven't tested.
        """

        # Make sure parameters are specified correctly and apply lower bound of 0 on all widths and amplitudes, unless
        # User specified the bounds of parameters to be something other than -np.inf already.

        # Prepare/check the self.params to be an acceptable set of parameters for my fitting procedure.
        if not self._param_keys:  # If the param_keys is empty, then have never run self.prepare_fit_params()
            self.prepare_fit_params()  # So, run self.prepare_fit_params()
        else:  # We have already run self.prepare_fit_params().
            try:
                _ = self.check_params_labels(self._param_keys)  # check to see if the parameters have changed.
            except MDC_Fit_Parameter_Mislabel:  # If they have, this error will be thrown.
                # So, empty and _param_keys and re-prepare them for the fit.
                self._param_keys = []
                self.prepare_fit_params()

        # New fit: new results, dispersions, etc...
        # Empty self._fit_ROI_results:
        if self._fit_ROI_results:
            self._fit_ROI_results = []
        # Empty self._MDC_peaks:
        if self._MDC_peaks:
            self._MDC_peaks = {}
        # Empty self._dispersion
        if self._dispersion:
            self._dispersion = {}
        # Empty self._constants:
        if self._constants:
            self._constants = []
        # Empty self._vf
        if self._vf:
            self._vf = {}
        # Empty self._kf
        if self._kf:
            self._kf = {}

        if eps is None:
            for E in self.cr.binding.values:
                # Grab every MDC in the image.
                data = self.cr.sel(binding=E).values
                # Fit every MDC:
                self.fit_ROI_results = self.fit_MDC(data, function_key_labels=self._function_key_labels)
                if self._previous_fit_as_guess_next_fit:
                    self.previous_fit_as_guess_next_fit()
        else:
            if eps.shape != self.cr.values.shape:
                raise Exception('You need to specify an eps with a shape equal to the shape of the ROI/cr of your ARPES'
                                ' image being fit, i.e. ...MDC.cr.shape')
            count = 0
            for E in self.cr.binding.values:
                # Grab every MDC in the image.
                data = self.cr.sel(binding=E).values
                # Fit every MDC with the appropriate eps.
                self.fit_ROI_results = self.fit_MDC(data, eps=eps[count, :],
                                                    function_key_labels=self._function_key_labels)
                count += 1
                if self._previous_fit_as_guess_next_fit:
                    self.previous_fit_as_guess_next_fit()
        return

    def previous_fit_as_guess_next_fit(self):
        # Set the starting value for fitting the next MDC as the results of the previous fitted MDC.
        parvals = self.params.valuesdict()
        for keys in parvals:
            self.params[keys].value = self.fit_ROI_results[-1].params[keys].value
        return

    @property
    def fit_ROI_results(self):
        """
        This property is a list of length len(self.cr.binding.values) and contains all minimizer results (see lmfit
         minimize docs) from fit.
        """
        if not self._fit_ROI_results:
            print('Need to run fit_ROI method on your MDC extension of xarray. If you think you have, for some reason'
                  'this was not populated with your results of fits...?')
            return
        return self._fit_ROI_results

    @fit_ROI_results.setter
    def fit_ROI_results(self, lmfit_minnimizer_result: minnimizer_result):
        self._fit_ROI_results.append(lmfit_minnimizer_result)
        return

    @property
    def fit_results_stored(self):
        """
        Not tested yet.
        """
        return self._fit_results_stored

    @fit_results_stored.setter
    def fit_results_stored(self, cr_bounds: List[float], fit_ROI_results: List[minnimizer_result]):
        # Probably does not work.
        self._fit_results_stored['cr_bounds'].append(cr_bounds)
        self._fit_results_stored['fit_ROI_results'].append(fit_ROI_results)
        return

    def assign_default_bounds_Gaussian(self, label: str, Gaussian: str):
        """
        Assign default bounds to the Gaussian parameters, unless bounds have already been explicitly defined.
        inputs: label is a string that contains the unique label of this Gaussian function. Gaussian is a string that is
        either 'gaussian' or 'Gaussian'—had no way to know if user used capital letter or not.
        """
        # Center of peak should be within the domain.
        if self.params[Gaussian+label+'center'].min == -np.inf:
            self.params[Gaussian + label + 'center'].min = np.amin(self.cr.kx.values)  # The minnimum of the domain.
        if self.params[Gaussian+label+'center'].max == np.inf:
            self.params[Gaussian + label + 'center'].max = np.amax(self.cr.kx.values)  # The maximum of the domain.
        # standard deviation must be >0
        if self.params[Gaussian + label + 'std'].min == -np.inf:
            self.params[Gaussian + label + 'std'].min = 0
        # Amplitude must be > 0
        if self.params[Gaussian + label + 'amplitude'].min == -np.inf:
            self.params[Gaussian + label + 'amplitude'].min = 0
        return

    def check_Gaussian_names(self, label: str, Gaussian: str):
        """
        This function makes sure that for any label defining a Gaussian function, there are all of the
        associated parameters required to fully parameterize a Gaussian function.
        inputs: label is a string that contains the unique label of this Gaussian function. Gaussian is a string that is
        either 'gaussian' or 'Gaussian'—had no way to know if user used capital letter or not.
        """
        parvals = self.params.valuesdict()
        # Treat a background specially? Not yet.
        try:
            # Check if std for this labeled Gaussian exists.
            _ = parvals[Gaussian + label + 'std']
        except KeyError:
            self.params.add(Gaussian + label + 'std', value=0.2)
        try:
            # Check if amplitude for this labeled Gaussian exists.
            _ = parvals[Gaussian + label + 'amplitude']
        except KeyError:
            self.params.add(Gaussian + label + 'amplitude', value=1.0)
        return

    def assign_default_bounds_Voigt(self, label: str, Voigt: str):
        """
        Assign default bounds to the Voigt parameters, unless bounds have already been explicitly defined.
        inputs: label is a string that contains the unique label of this Voigt function. Voigt is a string that is
        either 'voigt' or 'Voigt'—had no way to know if user used capital letter or not.
        """
        # Bound the center to the  crop region of the fit. No peaks outside the domain.
        if self.params[Voigt + label + 'center'].min == -np.inf:
            self.params[Voigt + label + 'center'].min = np.amin(self.cr.kx.values)  # The minnimum of the domain.
        if self.params[Voigt + label + 'center'].max == np.inf:
            self.params[Voigt + label + 'center'].max = np.amax(self.cr.kx.values)  # The maximum of the domain.
        # alpha is a width, which >0
        if self.params[Voigt + label + 'alpha'].min == -np.inf:
            self.params[Voigt + label + 'alpha'].min = 0
        # Gamma is FWHM of Gaussian >0
        if self.params[Voigt + label + 'gamma'].min == -np.inf:
            self.params[Voigt + label + 'gamma'].min = 0
        # Amplitude > 0
        if self.params[Voigt + label + 'amplitude'].min == -np.inf:
            self.params[Voigt + label + 'amplitude'].min = 0
        return

    def check_Voigt_names(self, label: str, Voigt: str):
        """
        This function makes sure that for any label defining a voigt function, there are all of the parameters required
        to fully parameterize a Voigt function.
        inputs: label is a string that contains the unique label of this Voigt function. Voigt is a string that is
        either 'voigt' or 'Voigt'—had no way to know if user used capital letter or not.
        """

        parvals = self.params.valuesdict()
        # Treat a background specially? Not yet.
        try:
            # Check if alpha for this labeled voigt exists.
            _ = parvals[Voigt + label + 'alpha']
        except KeyError:
            self.params.add(Voigt + label + 'alpha', value=0.2)

        try:
            # Check if gamma for this labeled voigt exists.
            _ = parvals[Voigt + label + 'gamma']
        except KeyError:
            self.params.add(Voigt + label + 'gamma', value=0.2)

        try:
            # Check if amplitude for this labeled voigt exists.
            _ = parvals[Voigt + label + 'amplitude']
        except KeyError:
            self.params.add(Voigt + label + 'amplitude', value=1.0)
        return

    def check_Lorentzian_names(self, label: str, Lorentzian:str):
        """
        This function makes sure that for any label defining a Lorentzian function, there are all of the parameters
        required to fully parameterize a Lorentzian function.
        inputs: label is a string that contains the unique label of this Lorentzian function. Lorentzian is a string
        that is either 'lorentzian' or 'Lorentzian'—had no way to know if user used capital letter or not.
        """

        parvals = self.params.valuesdict()
        try:
            # Check if gamma for this labeled lorentzian exists.
            _ = parvals[Lorentzian + label + 'gamma']
        except KeyError:
            self.params.add(Lorentzian + label + 'gamma', value=0.2)

        try:
            # Check if amplitude for this labeled lorentzian exists.
            _ = parvals[Lorentzian + label + 'amplitude']
        except KeyError:
            self.params.add(Lorentzian + label + 'amplitude', value=1.0)
        return

    def assign_default_bounds_Lorentzian(self, label: str, Lorentzian: str):
        """
        Assign default bounds to the Lorentzian parameters, unless bounds have already been explicitly defined.
        inputs: label is a string that contains the unique label of this Lorentzian function. Lorentzian is a string
        that is either 'lorentzian' or 'Lorentzian'—had no way to know if user used capital letter or not.
        """
        # Bound the center to the  crop region of the fit. No peaks outside the domain.
        if self.params[Lorentzian + label + 'center'].min == -np.inf:
            self.params[Lorentzian + label + 'center'].min = np.amin(self.cr.kx.values)  # The minnimum of the domain.
        if self.params[Lorentzian + label + 'center'].max == np.inf:
            self.params[Lorentzian + label + 'center'].max = np.amax(self.cr.kx.values)  # The maximum of the domain.
        # Gamma is FWHM of lorentzian >0
        if self.params[Lorentzian + label + 'gamma'].min == -np.inf:
            self.params[Lorentzian + label + 'gamma'].min = 0
        # Amplitude > 0
        if self.params[Lorentzian + label + 'amplitude'].min == -np.inf:
            self.params[Lorentzian + label + 'amplitude'].min = 0
        return

    def prepare_fit_params(self):
        """
        Make sure the self.params object has some parameters, that any function with a center parameter also has
        the other required parameters (amplitude, etc.), that the naming scheme follows the required naming scheme of
        params, and assign some default bounds (unless the user set self._no_default_bounds = True).
        """
        parvals = self.params.valuesdict()
        expected_parameter_names = []
        self._function_key_labels = []
        # check if there are any defined parameters.
        if not parvals.keys():
            # There are no parameters. Assign some default params.
            print("Be advised, you specified no parameters, that is no model for the fitting of your MDCs. Thus, I am"
                  "assigning the default model: (One Voigt function) + constant")
            self.params.add('Voigt_center', value=0)
            self.params.add('constant', value=0)
            parvals = self.params.valuesdict()
            self._no_default_bounds = False

        # Go through parameter names and find constants, Gaussians, Voigts, and Lorentzians by the definition of their
        # center parameter. Use their labels and make sure the remaining parameters for the functions exist with the
        # same labels and have reasonable bounds.
        for keys in parvals:
            if (keys == 'DC_offset' or keys == 'DC_Offset' or keys == 'Constant' or keys == 'constant' or
                    keys == 'Offset' or keys == 'offset'):
                self._constant_key = keys
                self._function_key_labels.append(keys)  # Add constant key
                if not self._no_default_bounds:
                    # If there is a constant defined in the model with -inf lower bound, replace lower bound with 0
                    if self.params[keys].min == -np.inf:
                        self.params[keys].min = 0
                # Create a list of expected parameter names:
                expected_parameter_names.append(keys)
            elif ('Voigt' in keys or 'voigt' in keys) and 'center' in keys:
                try:
                    label = re.search('Voigt(.+?)center', keys, re.IGNORECASE).group(1)
                    self.check_Voigt_names(label, keys[0:5])
                    if not self._no_default_bounds:
                        self.assign_default_bounds_Voigt(label, keys[0:5])
                    self._function_key_labels.append(keys[0:5] + label)  # Add function name + label
                    # Create a list of expected parameter names:
                    expected_parameter_names.append(keys[0:5] + label + 'center')
                    expected_parameter_names.append(keys[0:5] + label + 'alpha')
                    expected_parameter_names.append(keys[0:5] + label + 'gamma')
                    expected_parameter_names.append(keys[0:5] + label + 'amplitude')
                except AttributeError:
                    # did not find something between 'Voigt' and 'center'
                    raise MDC_Fit_Parameter_Mislabel("You defined a parameter containing voigt and center, but there "
                                                     "must be some character, any character, between voigt and center")
            elif ('Gaussian' in keys or 'gaussian' in keys) and 'center' in keys:
                try:
                    label = re.search('Gaussian(.+?)center', keys, re.IGNORECASE).group(1)
                    self.check_Gaussian_names(label, keys[0:8])
                    self._function_key_labels.append(keys[0:8] + label)  # Add function name + label
                    if not self._no_default_bounds:
                        self.assign_default_bounds_Gaussian(label, keys[0:8])
                    # Create a list of expected parameter names:
                    expected_parameter_names.append(keys[0:8] + label + 'center')
                    expected_parameter_names.append(keys[0:8] + label + 'std')
                    expected_parameter_names.append(keys[0:8] + label + 'amplitude')
                except AttributeError:
                    # did not find something between 'Gaussian' and 'center'
                    raise MDC_Fit_Parameter_Mislabel("You defined a parameter containing gaussian and center, but there"
                                                     " must be some character, any character, between gaussian and "
                                                     "center")
            elif ('Lorentzian' in keys or 'lorentzian' in keys) and 'center' in keys:
                try:
                    label = re.search('Lorentzian(.+?)center', keys, re.IGNORECASE).group(1)
                    self.check_Lorentzian_names(label, keys[0:10])
                    self._function_key_labels.append(keys[0:10] + label)  # Add function name + label
                    if not self._no_default_bounds:
                        self.assign_default_bounds_Lorentzian(label, keys[0:10])
                    # Create a list of expected parameter names:
                    expected_parameter_names.append(keys[0:10] + label + 'center')
                    expected_parameter_names.append(keys[0:10] + label + 'gamma')
                    expected_parameter_names.append(keys[0:10] + label + 'amplitude')
                except AttributeError:
                    # did not find something between 'Lorentzian' and 'center'
                    raise MDC_Fit_Parameter_Mislabel("You defined a parameter containing lorentzian and center, but "
                                                     "there must be some character, any character, between lorentzian "
                                                     "and center")

        # Check to see if the parameters have all of the names that I expect them to.
        self._param_keys = self.check_params_labels(expected_parameter_names)
        return

    def check_params_labels(self, expected_params: List[str]):
        """
        Compare the names of parameters to a list of parameter names that I expect to find, expected_params. If the
        parameter names are not the expected_params exactly, throw reasonable error to help user name them correctly.
        """
        parvals = self.params.valuesdict()
        param_names = []
        for keys in parvals:
            param_names.append(keys)
        param_names_clean = copy.deepcopy(param_names)
        if len(param_names) > len(expected_params):
            for key in expected_params:
                if key in param_names:
                    param_names.remove(key)
            raise MDC_Fit_Parameter_Mislabel('You have more parameters than I expected. And here are the names of '
                                             'parameters that did not match the correct labeling scheme or are extra:',
                                             param_names, 'Here is what I expected:', expected_params,
                                             'And what you have:', param_names_clean)
        elif len(param_names) < len(expected_params):
            for key in expected_params:
                if key in param_names:
                    param_names.remove(key)
            if param_names:
                raise MDC_Fit_Parameter_Mislabel('You do not have as many parameters as expected. And you also have '
                                                 'some parameters with mislabeled names:', param_names,
                                                 'Here is what I expected:', expected_params, 'And what you have:',
                                                 param_names_clean)
            else:
                raise MDC_Fit_Parameter_Mislabel('You are missing some parameters. Everything you do have is labeled '
                                                 'correctly.', 'Here is what I expected:', expected_params,
                                                 'And what you have:', param_names_clean)
        else:
            for key in expected_params:
                if key in param_names:
                    param_names.remove(key)
            if param_names:
                raise MDC_Fit_Parameter_Mislabel('You have some mislabeled parameters. You do have the same number of '
                                                 'expected parameters, but not the same labels as I expect. Here are '
                                                 'the labels that are wrong:', param_names, 'Here is what I expected:',
                                                 expected_params, 'And what you have:', param_names_clean)
        return param_names_clean

    def reset_params(self):
        """
        Reset the ...MDC.params to a blank Paramaters() object of lmfit.
        """
        self.params = Parameters()
        return

    def waterfall(self, ax, plot_MDC: bool = True, tot_fits: bool = True, individual_fits: bool = False,
                  remove_constant: bool = True, only_one: int = None, offset: float = 1, dE: float = 0.05,
                  plot_single_fit_function: str = False, MDC_args: tuple = ('b--'),
                  MDC_kwargs: dict = {}, tot_fit_args: tuple = ('r'),  tot_fit_kwargs: dict = {},
                  individual_fits_args: tuple or List[tuple] = ('k'),
                  individual_fits_kwargs: dict or List[dict] = {}, single_fit_function_args: tuple = ('g'),
                  single_fit_function_kwargs: dict = {}):
        """
        plot a waterfall of MDCs, total fit, and individual functions of the fit on axis, ax.
        inputs:
            ax = axis to be plotted on from matplotlib
            plot_MDC: if true, then plot the MDC data.
            tot_fits: if true, then plot the total fit for each MDC plotted on the axis, overlaying MDCs.
            individual_fits: if true, then plot individually all of the functions which add up to the total fit function
            remove_constant: if true, subtract constant from everything plotted, e.g. MDC + offset - constant.
            only_one: if an int, then acts as the index of the MDC,fit, etc. being plotted. As in, no waterfall,
            only_one MDC, etc.
            offset = spacing between each MDC, total fits and individual fits of MDC.
            dE = energy spacing between each MDC/fit grabbed for plotting, if dE=0 plot every MDC, fit, etc.
            plot_single_fit_function if you provide a string, it should be the function name + label of the individual
                fit you want to plot, which is in constrast to making infividual_fits true, which plots ALL individual
                functions of the fit instead of just the one you specify with this.
            MDC, single_fit_function, and tot_fit _args are tuples that pass args into ax.plot() for MDC,
                single_fit_function, and tot_fit, respectively. args are from matplotlib.plot(), follow their
                documentation.
            MDC, single_fit_function, and tot_fit _kwargs are dicts that pass kwargs into ax.plot() for MDC,
                single_fit_function, and tot_fit, respectively. kwargs are from matplotlib.plot(), follow their
                documentation.
            individual_fits_args can either be a single tuple of args to be used for every individual
                function of the total fit function, or can be a list of tuples of the same length as the number of
                individual fitting functions, including constant, comprising the total fit model. In the latter case,
                the list will be unpacked supplying args tuple for each fit function, in the order in which the
                constant or _center parameter of the labeled functions occurs in your xarray.MDC.params.valuesdict().
                That is, print(xarray.MDC.params.valuesdict()) and the first constant or center parameter found will
                specify the labeled function that will receive the args of the tuple located at
                individual_fits_lineparams[0], and so on...
            individual_fits_kwargs can either be a single dict of kwargs to be used for every individual
                function of the total fit function, or can be a list of dict of the same length as the number of
                individual fitting functions, including constant, comprising the total fit model. In the latter case,
                the list will be unpacked supplying kwargs dict for each fit function, in the order in which the
                constant or _center parameter of the labeled functions occurs in your xarray.MDC.params.valuesdict().
                That is, print(xarray.MDC.params.valuesdict()) and the first constant or center parameter found will
                specify the labeled function that will receive the kwargs of the dict located at
                individual_fits_lineparams[0], and so on...
        """
        if remove_constant:
            self.acquire_all_constants()

        if isinstance(only_one, int):
            index = only_one
            if plot_MDC:
                if remove_constant:
                    ax.plot(self.cr.kx.values,
                            (self.cr.isel(binding=index).values + index * offset - self.constants[index]),
                            *MDC_args, **MDC_kwargs)
                else:
                    ax.plot(self.cr.kx.values, (self.cr.isel(binding=index).values + index * offset), *MDC_args,
                            **MDC_kwargs)
            if tot_fits:
                if remove_constant:
                    ax.plot(self.cr.kx.values,
                            (self.residual(self.fit_ROI_results[index].params, self.cr.kx.values) + index * offset
                             - self.constants[index]),
                            *tot_fit_args, **tot_fit_kwargs)
                else:
                    ax.plot(self.cr.kx.values,
                            (self.residual(self.fit_ROI_results[index].params, self.cr.kx.values) + index * offset),
                            *tot_fit_args, **tot_fit_kwargs)
            if isinstance(plot_single_fit_function, str):
                fits = self.individual_fits(self.fit_ROI_results[index].params, self.cr.kx.values)
                try:
                    ax.plot(self.cr.kx.values, (fits[plot_single_fit_function] + index * offset), *single_fit_function_args,
                            **single_fit_function_kwargs)
                except KeyError:
                    raise KeyError('plot_single_fit_function needs to be one of the following:', fits.keys())
            if individual_fits:
                fits = self.individual_fits(self.fit_ROI_results[index].params, self.cr.kx.values)
                if isinstance(individual_fits_args, list):
                    if isinstance(individual_fits_kwargs, list):
                        try:
                            j = 0
                            for key in fits.keys():
                                ax.plot(self.cr.kx.values, (fits[key] + index * offset), *individual_fits_args[j],
                                        **individual_fits_kwargs[j])
                                j += 1
                        except IndexError:
                            raise IndexError('Note that the length of individual_fits_args and '
                                             'individual_fits_kwargs list should be equal to number of functions in'
                                             ' your overall fit, including constant.')
                    else:
                        try:
                            j = 0
                            for key in fits.keys():
                                ax.plot(self.cr.kx.values, (fits[key] + index * offset), *individual_fits_args[j],
                                        **individual_fits_kwargs)
                                j += 1
                        except IndexError:
                            raise IndexError('Note that the length of individual_fits_args list should be equal to '
                                             'number of functions in your overall fit, including constant.')
                else:
                    if isinstance(individual_fits_kwargs, list):
                        try:
                            j = 0
                            for key in fits.keys():
                                ax.plot(self.cr.kx.values, (fits[key] + index * offset), *individual_fits_args,
                                        **individual_fits_kwargs[j])
                                j += 1
                        except IndexError:
                            raise IndexError('Note that the length of individual_fits_kwargs list should be equal '
                                             'to number of functions in your overall fit, including constant.')
                    else:
                        for key in fits.keys():
                            ax.plot(self.cr.kx.values, (fits[key] + index * offset), *individual_fits_args,
                                    **individual_fits_kwargs)

            # Fix some bug where y-axis ticks are not visible without the below.
            ylim = ax.get_ylim()
            ax.set_ylim(0, ylim[1])
            ylim = ax.get_ylim()
            y_ticks = np.linspace(ylim[0], ylim[1], 4)
            ax.set_yticks(y_ticks)
            y_labels = [str(round(num, 1)) for num in y_ticks]
            ax.set_yticklabels(y_labels)
            return

        if dE != 0:
            energies = np.arange(self.cr_bounds[2], self.cr_bounds[3], step=dE)
            count = 0
            for E in energies:
                # Find closest energy
                index = np.argmin(np.abs(self.cr.binding.values-E))
                if plot_MDC:
                    if remove_constant:
                        ax.plot(self.cr.kx.values,
                                (self.cr.isel(binding=index).values+count*offset - self.constants[index]),
                                *MDC_args, **MDC_kwargs)
                    else:
                        ax.plot(self.cr.kx.values,
                                (self.cr.isel(binding=index).values + count * offset),
                                *MDC_args, **MDC_kwargs)
                if tot_fits:
                    if remove_constant:
                        ax.plot(self.cr.kx.values,
                                (self.residual(self.fit_ROI_results[index].params, self.cr.kx.values)
                                 + count * offset - self.constants[index]),
                                *tot_fit_args, **tot_fit_kwargs)
                    else:
                        ax.plot(self.cr.kx.values,
                                (self.residual(self.fit_ROI_results[index].params, self.cr.kx.values)+count*offset),
                                *tot_fit_args, **tot_fit_kwargs)
                if isinstance(plot_single_fit_function, str):
                    fits = self.individual_fits(self.fit_ROI_results[index].params, self.cr.kx.values)
                    try:
                        ax.plot(self.cr.kx.values, (fits[plot_single_fit_function] + index * offset),
                                *single_fit_function_args,
                                **single_fit_function_kwargs)
                    except KeyError:
                        raise KeyError('plot_single_fit_function needs to be one of the following:', fits.keys())
                if individual_fits:
                    fits = self.individual_fits(self.fit_ROI_results[index].params, self.cr.kx.values,
                                                function_key_labels=self._function_key_labels)
                    if isinstance(individual_fits_args, list):
                        if isinstance(individual_fits_kwargs, list):
                            try:
                                j = 0
                                for key in fits.keys():
                                    ax.plot(self.cr.kx.values, (fits[key]+count*offset), *individual_fits_args[j],
                                            **individual_fits_kwargs[j])
                                    j += 1
                            except IndexError:
                                raise IndexError('Note that the length of individual_fits_args and '
                                                 'individual_fits_kwargs list should be equal to number of functions in'
                                                 ' your overall fit, including constant.')
                        else:
                            try:
                                j = 0
                                for key in fits.keys():
                                    ax.plot(self.cr.kx.values, (fits[key]+count*offset), *individual_fits_args[j],
                                            **individual_fits_kwargs)
                                    j += 1
                            except IndexError:
                                raise IndexError('Note that the length of individual_fits_args list should be equal to '
                                                 'number of functions in your overall fit, including constant.')
                    else:
                        if isinstance(individual_fits_kwargs, list):
                            try:
                                j = 0
                                for key in fits.keys():
                                    ax.plot(self.cr.kx.values, (fits[key] + count * offset), *individual_fits_args,
                                            **individual_fits_kwargs[j])
                                    j += 1
                            except IndexError:
                                raise IndexError('Note that the length of individual_fits_kwargs list should be equal '
                                                 'to number of functions in your overall fit, including constant.')
                        else:
                            for key in fits.keys():
                                ax.plot(self.cr.kx.values, (fits[key] + count * offset), *individual_fits_args,
                                        **individual_fits_kwargs)
                count += 1
        else:
            for index in range(len(self.cr.binding.values)):
                if plot_MDC:
                    if remove_constant:
                        ax.plot(self.cr.kx.values,
                                (self.cr.isel(binding=index).values + index * offset - self.constants[index]),
                                *MDC_args, **MDC_kwargs)
                    else:
                        ax.plot(self.cr.kx.values, (self.cr.isel(binding=index).values+index*offset), *MDC_args,
                                **MDC_kwargs)
                if tot_fits:
                    if remove_constant:
                        ax.plot(self.cr.kx.values,
                                (self.residual(self.fit_ROI_results[index].params, self.cr.kx.values) + index * offset
                                 - self.constants[index]),
                                *tot_fit_args, **tot_fit_kwargs)
                    else:
                        ax.plot(self.cr.kx.values,
                                (self.residual(self.fit_ROI_results[index].params, self.cr.kx.values)+index*offset),
                                *tot_fit_args, **tot_fit_kwargs)
                if isinstance(plot_single_fit_function, str):
                    fits = self.individual_fits(self.fit_ROI_results[index].params, self.cr.kx.values)
                    try:
                        ax.plot(self.cr.kx.values, (fits[plot_single_fit_function] + index * offset),
                                *single_fit_function_args,
                                **single_fit_function_kwargs)
                    except KeyError:
                        raise KeyError('plot_single_fit_function needs to be one of the following:', fits.keys())
                if individual_fits:
                    fits = self.individual_fits(self.fit_ROI_results[index].params, self.cr.kx.values)
                    if isinstance(individual_fits_args, list):
                        if isinstance(individual_fits_kwargs, list):
                            try:
                                j = 0
                                for key in fits.keys():
                                    ax.plot(self.cr.kx.values, (fits[key]+index*offset), *individual_fits_args[j],
                                            **individual_fits_kwargs[j])
                                    j += 1
                            except IndexError:
                                raise IndexError('Note that the length of individual_fits_args and '
                                                 'individual_fits_kwargs list should be equal to number of functions in'
                                                 ' your overall fit, including constant.')
                        else:
                            try:
                                j = 0
                                for key in fits.keys():
                                    ax.plot(self.cr.kx.values, (fits[key]+index*offset), *individual_fits_args[j],
                                            **individual_fits_kwargs)
                                    j += 1
                            except IndexError:
                                raise IndexError('Note that the length of individual_fits_args list should be equal to '
                                                 'number of functions in your overall fit, including constant.')
                    else:
                        if isinstance(individual_fits_kwargs, list):
                            try:
                                j = 0
                                for key in fits.keys():
                                    ax.plot(self.cr.kx.values, (fits[key] + index * offset), *individual_fits_args,
                                            **individual_fits_kwargs[j])
                                    j += 1
                            except IndexError:
                                raise IndexError('Note that the length of individual_fits_kwargs list should be equal '
                                                 'to number of functions in your overall fit, including constant.')
                        else:
                            for key in fits.keys():
                                ax.plot(self.cr.kx.values, (fits[key] + index * offset), *individual_fits_args,
                                        **individual_fits_kwargs)

        # Fix some bug where y-axis ticks are not visible without the below.
        ylim = ax.get_ylim()
        ax.set_ylim(0, ylim[1])
        ylim = ax.get_ylim()
        y_ticks = np.linspace(ylim[0], ylim[1], 4)
        ax.set_yticks(y_ticks)
        y_labels = [str(round(num, 1)) for num in y_ticks]
        ax.set_yticklabels(y_labels)
        return

    def plot_arpes_image(self, ax, plot_data: bool = True, plot_peaks: bool = True, plot_dispersion: bool = True,
                         plot_crop_region_bounds: bool = True, pad_k_dispersion: float = 0,
                         plot_peaks_args: tuple = ('x'),
                         plot_dispersion_args: tuple = ('r'), plot_crop_region_bounds_args: tuple = ('w'),
                         plot_peaks_kwargs: dict = {}, plot_dispersion_kwargs: dict = {},
                         plot_crop_region_bounds_kwargs: dict = {}):
        """
        plot E-k ARPES image, peak locations, and fitted dispersion.
        inputs:
        ax: an axis object from matplotlib, on which things will be plotted.
        plot_data: If true, plot E-k ARPES image
        plot_peaks: If true, plot the peak locations from MDC fittings as (E,k) points.
        plot_dispersion: If true, plot polynomial fits to MDC peak locations.
        plot_crop_region_bounds: If true, plot a box representing the crop region of the ARPES image used for MDC
            fitting.
        pad_k_dispersion: Float that extends the domain of the plotted polynomial dispersion beyond the k/E-range of the
             MDC peak locations used for the polyfit by +/-pad_k_dispersion
        plot_peaks, plot_dispersion, and plot_crop_region_bounds + _args is a tuple of arguments passed to ax.plot()
            function from matplotlib. See their document for acceptable args.
        plot_peaks, plot_dispersion, and plot_crop_region_bounds + _kwargs is a dict of key word arguments passed to
            ax.plot() function from matplotlib. See their document for acceptable kwargs.
        """
        if plot_data:  # Then plot the image on ax.
            self._obj.plot(x='kx', y='binding', ax=ax)
        if plot_peaks:
            if not self.MDC_peaks:
                self.extract_peaks()
            for key in self.MDC_peaks.keys():
                ax.plot(self.MDC_peaks[key], self.cr.binding.values, *plot_peaks_args, **plot_peaks_kwargs)
        if plot_dispersion:
            if not self.dispersion:
                self.extract_dispersion()
            for key in self.dispersion.keys():
                if self.dispersion[key].variable == 'kx':
                    k_min = np.amin(self.MDC_peaks[key]) - pad_k_dispersion
                    k_max = np.amax(self.MDC_peaks[key]) + pad_k_dispersion
                    k = np.arange(k_min, k_max,
                                  step=(self.cr.kx.values[1]-self.cr.kx.values[0]))
                    ax.plot(k, self.dispersion[key](k), *plot_dispersion_args, **plot_dispersion_kwargs)
                else:
                    E_min = np.amin(self.cr.binding.values) - pad_k_dispersion
                    E_max = np.amax(self.cr.binding.values) + pad_k_dispersion
                    E = np.arange(E_min, E_max, self.cr.binding.values[1] - self.cr.binding.values[0])
                    ax.plot(self.dispersion[key](E), E, *plot_dispersion_args, **plot_dispersion_kwargs)
        if plot_crop_region_bounds:
            ax.plot([self.cr_bounds[0], self.cr_bounds[1], self.cr_bounds[1], self.cr_bounds[0], self.cr_bounds[0]],
                    [self.cr_bounds[2], self.cr_bounds[2], self.cr_bounds[3], self.cr_bounds[3], self.cr_bounds[2]],
                    *plot_crop_region_bounds_args, **plot_crop_region_bounds_kwargs)
        return

    @property
    def MDC_peaks(self):
        """
        List of (list of peaks for each energy) where the number of peaks corresponds to how many peaked functions
        defined in your model that do not include 'bg' or some variant of 'background' in the label.
        returned as nd.array
        """
        return self._MDC_peaks

    def extract_peaks(self):
        """
        extract the centers of Gaussians, Voigts, and Lorentzians that are not labeled with  bg, background
        """
        # Want to use self._function_key_labels; so better not be empty, and should be populated during fit anyway.
        if self._function_key_labels:
            for key in self._function_key_labels:
                if not key == self._constant_key:
                    if 'bg' not in key and 'background' not in key and 'back_ground' not in key and \
                            'BG' not in key and 'BackGround' not in key:
                        peaks = []
                        for j in range(len(self.fit_ROI_results)):
                            peaks.append(self.fit_ROI_results[j].params[key+'center'].value)
                        self._MDC_peaks[key] = np.asarray(peaks)
        else:
            raise Exception("_function_key_labels empty. Did not think this was possible, unless you have not run"
                            "xarray.MDC.fit_ROI()")
        return

    def extract_dispersion(self, force_E_dependent = False, **kwargs):
        """
        Get the dispersion, E(k), where E(k) is a fited polynomial. pass deg=int to control order of poly
        inputs:
            kwargs are key word args for np.polyfit, see their documentation.
        """
        # If you are extracting dispersion, reset all populated dicts to empty
        if self._dispersion:
            self._dispersion = {}
        if self.kf:
            self._kf = {}
        if self.vf:
            self._kf = {}

        # Make sure a deg for the polynomial is specified.
        try:
            _ = kwargs['deg']
        except KeyError:
            kwargs = {'deg': 1}

        # Must have MDC_peaks to extract dispersion, get them.
        if not self._MDC_peaks:
            self.extract_peaks()

        # Loop over the keys of MDC_peaks, which should be all function labels that contain center parameters, no bg
        Energies = self.cr.binding.values  # Same energies for all fits.
        for key in self.MDC_peaks.keys():
            k = self.MDC_peaks[key]
            # Turns out the polyfits are better, if domain is variable with larger variance...
            if Energies.std() > k.std() and not force_E_dependent:
                # In this case, E will be the domain, and we will find k(E)
                x = Energies
                y = k
                E_independent = True
            else:
                # In this case, k will be the domain, and we will find E(k)
                x = k
                y = Energies
                E_independent = False
            coefs = np.polyfit(x, y, **kwargs)  # returns coefficients of poly fit

            # Dict: keys function labels, returns 1dpoly interpolation object. get E_{key}(k) = self.dispersion[key](k)
            # If variable is 'kx', but if it is binding get k_{key}(binding) = self.dispersion[key](binding)
            # Check what the independent variable is with self.dispersion[key].variable
            if E_independent:
                self._dispersion[key] = np.poly1d(coefs, variable='binding')
            else:
                self._dispersion[key] = np.poly1d(coefs, variable='kx')

            # Get vf if appropriate; so I need kf if the band crosses Ef.
            # If Ef is not within fitting window or within 100 meV of upper bound of fitting window, don't get vf or kf.
            if self.cr_bounds[2] < 0.0 < self.cr_bounds[3] + 0.1:
                # New set of peaks, reset parameters to extract:
                d_coefs = []
                all_kf = []
                all_vf = []
                if not E_independent:
                    roots = np.roots(coefs)  # Get roots of E(k), since Ef=0
                    realroots = [n for n in roots if isinstance(n, float)]  # keep only real roots.
                    count = 0
                    for root in realroots:
                        if np.amax(x) > root > np.amin(x):
                            kf = root  # E is in binding; sp of root is real and within domain of the fit, it is kf.
                            count += 1  # Number of viable kfs found, expect to be only 1.
                            if not d_coefs:
                                N = kwargs['deg']
                                for j in range(N):
                                    d_coefs.append((N - j) * coefs[j])
                            vf = np.poly1d(d_coefs)(kf)  # vf is, of course, the derivative of E(k) at kf.
                            all_kf.append(kf)
                            all_vf.append(vf)
                        elif self.cr_bounds[0] < root < self.cr_bounds[1]:
                            print('Be advised for function_key: ', key, ' found kf outside of range min(MDC_peaks)'
                                                                        'to max(MDC_peaks), which is domain of polyfit; '
                                                                        'however, I did find a kf within your cr_bounds'
                                                                        ' over which MDCs were fit; so I will find vf and '
                                                                        'store vf and kf for this point, but this may be'
                                                                        'misleading, use discretion.')
                            kf = root  # E is in binding; sp of root is real and within domain of the fit, it is kf.
                            count += 1  # Number of viable kfs found, expect to be only 1.
                            if not d_coefs:
                                N = kwargs['deg']
                                for j in range(N):
                                    d_coefs.append((N-j)*coefs[j])
                            vf = np.poly1d(d_coefs)(kf)  # vf is, of course, the derivative of E(k) at kf.
                            all_kf.append(kf)
                            all_vf.append(vf)
                    # Store the kf and vf pairs we found.
                    self._vf[key] = np.asarray(all_vf)
                    self._kf[key] = np.asarray(all_kf)
                    if count > 1:
                        print("Be advised: found more than 1 kf, meaning polynomial fit crosses Ef more than once within"
                              "the domain of the fit of the polynomial to (centers, binding), for this function_key_label",
                              key)
                else:
                    if np.amax(y) > np.poly1d(coefs)(0) > np.amin(y):  # is k(E=0) within k's of fit?
                        kf = np.poly1d(coefs)(0)  # k(E=0) is kf and there can only be one, because k(E) is a function.
                        if not d_coefs:
                            N = kwargs['deg']
                            for j in range(N):
                                d_coefs.append((N - j) * coefs[j])
                        vf = 1/np.poly1d(d_coefs)(kf)  # vf is, of course, the derivative of E(k) at kf. or (dk/dE)^-1
                        all_kf.append(kf)
                        all_vf.append(vf)
                    elif self.cr_bounds[0] < np.poly1d(coefs)(0) < self.cr_bounds[1]:
                        print('Be advised for function_key: ', key, ' found kf outside of range min(MDC_peaks)'
                                                                    'to max(MDC_peaks), which is domain of polyfit; '
                                                                    'however, I did find a kf within your cr_bounds'
                                                                    ' over which MDCs were fit; so I will find vf and '
                                                                    'store vf and kf for this point, but this may be'
                                                                    'misleading, use discretion.')
                        kf = np.poly1d(coefs)(0)  # k(E=0) is kf and there can only be one, because k(E) is a function.
                        if not d_coefs:
                            N = kwargs['deg']
                            for j in range(N):
                                d_coefs.append((N-j)*coefs[j])
                        vf = 1/np.poly1d(d_coefs)(kf)  # vf is, of course, the derivative of E(k) at kf. or (dk/dE)^-1
                        all_kf.append(kf)
                        all_vf.append(vf)
                    # Store the kf and vf pairs we found.
                    self._vf[key] = np.asarray(all_vf)
                    self._kf[key] = np.asarray(all_kf)
        return

    @property
    def dispersion(self):
        """
        dict with keys that are functions + str_key assigned by user, similar to
        self._function_key_labels; however, does not have keys for constants and also neglect
        functions with str_key = 'bg', 'back_ground' ... (see self.extract_peaks method for if statement that explicitly
        shows which str_key labels are neglected). These are neglected because we do not want dispersion of
        background peaks.
        self.dispersion[key].variable tells you whether E or k are independent
                            variable.

        return: dict[key] = numpy.poly1d that represents dispersion. These objects can be fed a k [E] value, and will
                            spit out E(k) [k(E)].
        """
        if not self._dispersion:  # If still empty, extract it.
            self.extract_dispersion()
        return self._dispersion

    @staticmethod
    def individual_fits(params: minnimizer_result, kx, function_key_labels: List[str] = None):
        """
        Return the individual functions of the total fit, e.g. each Voigt, Gaussian, Offset, and Lorentzian individually
        Follows similar form as residual method.
        Return: list of (numpy arrays with length of kx), where each element of the list is one of the functions and
        thus the list has length equal to the number of Gaussians, Voigts, and Lorentzians in model. Ignore the constant
        """
        parvals = params.valuesdict()
        fit = {}
        # Go through parameter names and find constants, Gaussians, Voigts, and Lorentzians by the definition of their
        # center parameter. Use all necesesary, similarly labeled parameters for the function type to add to the
        # function to the overall fit model.
        if isinstance(function_key_labels, list) and function_key_labels:
            for keys in function_key_labels:
                if 'Voigt' in keys or 'voigt' in keys:
                    try:
                        fit[keys] = Voigt(kx, parvals[keys + 'center'], parvals[keys + 'alpha'], parvals[keys + 'gamma'],
                                         parvals[keys + 'amplitude'])
                    except KeyError:
                        raise MDC_Fit_Parameter_Mislabel("Found a 'Voigt' + str + 'center', but you probably defined a "
                                                         "parameter of the Voigt function that did not follow the same "
                                                         "pattern of 'Voigt' + str + (lower-case parameter: alpha, "
                                                         "gamma,  or amplitude).")
                if 'Gaussian' in keys or 'gaussian' in keys:
                    try:
                        fit[keys] = Gaussian(kx, parvals[keys + 'center'], parvals[keys + 'std'],
                                            parvals[keys + 'amplitude'])
                    except KeyError:
                        raise MDC_Fit_Parameter_Mislabel("Found a 'Gaussian' + str + 'center', but you probably defined"
                                                         " a parameter of the Gaussian function that did not follow the"
                                                         " same pattern of 'Gaussian' + str + (lower-case parameter: "
                                                         "std or amplitude).")
                if 'Lorentzian' in keys or 'lorentzian' in keys:
                    try:
                        fit[keys] = Lorentzian(kx, parvals[keys + 'center'], parvals[keys + 'gamma'],
                                              parvals[keys + 'amplitude'])
                    except KeyError:
                        raise MDC_Fit_Parameter_Mislabel("Found a 'Lorentzian' + str + 'center', but you probably "
                                                         "defined a parameter of the Lorentzian function that did not "
                                                         "follow the same pattern of 'Lorentzian' + str + "
                                                         "(lower-case parameter: gamma or amplitude).")
        else:
            for keys in parvals:
                if ('Voigt' in keys or 'voigt' in keys) and 'center' in keys:
                    try:
                        label = re.search('Voigt(.+?)center', keys, re.IGNORECASE).group(1)
                    except AttributeError:
                        # did not find something between 'Voigt' and 'center'
                        raise MDC_Fit_Parameter_Mislabel("You defined a parameter containing voigt and center, but there "
                                                         "must be some character, any character, between voigt and center")
                    try:
                        fit[keys[0:5] + label] = Voigt(kx, parvals[keys[0:5] + label + 'center'],
                                     parvals[keys[0:5] + label + 'alpha'],
                                     parvals[keys[0:5] + label + 'gamma'], parvals[keys[0:5] + label + 'amplitude'])
                    except KeyError:
                        raise MDC_Fit_Parameter_Mislabel("Found a 'Voigt' + str + 'center', but you probably defined a "
                                                         "parameter of the Voigt function that did not follow the same "
                                                         "pattern of 'Voigt' + str + (lower-case parameter: alpha, gamma, "
                                                         " or amplitude).")
                if ('Gaussian' in keys or 'gaussian' in keys) and 'center' in keys:
                    try:
                        label = re.search('Gaussian(.+?)center', keys, re.IGNORECASE).group(1)
                    except AttributeError:
                        # did not find something between 'Gaussian' and 'center'
                        raise MDC_Fit_Parameter_Mislabel("You defined a parameter containing gaussian and center, but there"
                                                         " must be some character, any character, between gaussian and "
                                                         "center")
                    try:
                        fit[keys[0:8] + label] = Gaussian(kx, parvals[keys[0:8] + label + 'center'],
                                        parvals[keys[0:8] + label + 'std'],
                                        parvals[keys[0:8] + label + 'amplitude'])
                    except KeyError:
                        raise MDC_Fit_Parameter_Mislabel("Found a 'Gaussian' + str + 'center', but you probably defined a "
                                                         "parameter of the Gaussian function that did not follow the same "
                                                         "pattern of 'Gaussian' + str + (lower-case parameter: std or "
                                                         "amplitude).")
                if ('Lorentzian' in keys or 'lorentzian' in keys) and 'center' in keys:
                    try:
                        label = re.search('Lorentzian(.+?)center', keys, re.IGNORECASE).group(1)
                    except AttributeError:
                        # did not find something between 'Lorentzian' and 'center'
                        raise MDC_Fit_Parameter_Mislabel("You defined a parameter containing lorentzian and center, "
                                                         "but there must be some character, any character, between "
                                                         "lorentzian and center")
                    try:
                        fit[keys[0:10] + label] = Lorentzian(kx, parvals[keys[0:10] + label + 'center'],
                                              parvals[keys[0:10] + label + 'gamma'],
                                              parvals[keys[0:10] + label + 'amplitude'])
                    except KeyError:
                        raise MDC_Fit_Parameter_Mislabel("Found a 'Lorentzian' + str + 'center', but you probably "
                                                         "defined a parameter of the Lorentzian function that did not "
                                                         "follow the same pattern of 'Lorentzian' + str + "
                                                         "(lower-case parameter: gamma or amplitude).")

        return fit

    @property
    def vf(self):
        """
        dict with keys that are functions + str_key assigned by user, similar to
        self._function_key_labels; however, does not have keys for constants and also neglect
        functions with str_key = 'bg', 'back_ground' ... (see self.extract_peaks method for if statement that explicitly
        shows which str_key labels are neglected). These are neglected because we do not want dispersion of
        background peaks.

        return: dict[key] = nd.array of vf in units of [binding]/[kx]
        """
        if not self._vf:
            print('Be advised: vf is empty, which could mean you need to extract_dispersion, or maybe your crop region'
                  'does not contain Ef within 100 meV of upper bound, if so I will not automatically extract vf.')
        return self._vf

    @property
    def kf(self):
        """
        dict with keys that are functions + str_key assigned by user, similar to
        self._function_key_labels; however, does not have keys for constants and also neglect
        functions with str_key = 'bg', 'back_ground' ... (see self.extract_peaks method for if statement that explicitly
        shows which str_key labels are neglected). These are neglected because we do not want dispersion of
        background peaks.

        return: dict[key] = nd.array of kf in units of [kx]
        """
        if not self._kf:
            print('Be advised: kf is empty, which could mean you need to extract_dispersion, or maybe your crop region'
                  'does not contain Ef within 100 meV of upper bound, if so I will not automatically extract kf.')
        return self._kf

    @property
    def constants(self):
        """
        List of floats that are the constants of the fits to MDC. Therefore, list has length of self.cr.binding.values.
        """
        return self._constants

    @constants.setter
    def constants(self, value: float):
        self._constants.append(value)
        return

    def acquire_all_constants(self):
        """
        Get all of the constants from fits. 
        """
        # Make sure fits exist; otherwise, no constants to extract from fits.
        if not self.fit_ROI_results:
            raise Exception("Cannot get constants of the fits if fits haven't been found")

        # Check that the model contains one constant:
        check = 0
        for keys in self.params.valuesdict():
            if keys == 'DC_offset' or keys == 'Constant' or keys == 'constant' or keys == 'Offset' or \
                    keys == 'offset' or keys == 'DC_Offset':
                check += 1
        if check == 0:
            raise Exception("Your model does not contain a constant, yet you are trying to extract the constants of "
                            "the fits?")
        elif check > 1:
            raise Exception("Your model has more than 1 constant. Why? I assume you either use 1 or no constants. "
                            "Afterall, constant1+constant2+... is still a constant.")

        # Now go find the constants and add them to self.constants
        for index in range(len(self.fit_ROI_results)):
            self.constants = self.fit_ROI_results[index].params.valuesdict()[self._constant_key]
        return
