"""
This core class will define methods and properties that are generically useful to all analysis packages, e.g. MDC class,
which will all subclass this core class; so that we do not have to rewrite these methods, etc. For example, selecting a
cropped region to use for fitting will be generically useful to MDC and EDC fitting. However, do note that we should not
repeat properties or methods that are defined in arpes class, which will have an extention for MDC and other analysis
classes all future analysis packages that subclass core.
"""

from typing import List, Dict, Set, Tuple, NewType
import xarray as xr
from ..Arpes import *
import numpy as np

xarray = NewType('xarray', xr)


class ConventionError(Exception):
    pass


class Core:

    def __init__(self, xarray_obj: xarray):
        """
        Must instantiate this object with an instance of an ARPES object.
        All analysis will be done, assuming that the xarray object is on regular k-space and energy griding with Ef
        defined. This is accomplished by successfully running xarray.arpes.spectra_k_reg().
        and k-space coordinates.
        """
        self._obj = xarray_obj
        self._cr = self._obj  # ROI will be entire ARPES image if not specified, using ROI setter.
        self._cr_bounds = None

    @property
    def cr(self):
        return self._cr

    @cr.setter
    def cr(self, crop_region: List[float]):
        """
        crop_region: list of 4 floats [k_min, k_max, E_min, E_max]
        """
        if crop_region[2] < 0:
            self._cr = self._obj.sel(kx=slice(crop_region[0], crop_region[1])).sel(binding=slice(crop_region[2],
                                                                                                 crop_region[3]))
        elif crop_region[2] > 0:
            raise ConventionError("Seems like your xarray may be in kinetic energy; however, "
                                  "it must be in binding energy")
        else:
            raise ConventionError("If E_min is negative, you are using binding energy."
                                  "If E_min is positive you are in kinetic energy. Somehow, E_min = 0, "
                                  "which seems impossible.")
        return

    @property
    def cr_bounds(self):
        return self._cr_bounds

    @cr_bounds.setter
    def cr_bounds(self, crop_region: List[float]):
        """
        crop_region: list of 4 floats [k_min, k_max, E_min, E_max]
        """
        self._cr_bounds = crop_region
        self.cr = self._cr_bounds
        return



