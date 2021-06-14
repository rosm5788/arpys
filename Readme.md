# ARPYS

Angle resolved Python spectroscopy - Dessau Group

Dependencies: 
numpy,scipy, xarray, matplotlib, pandas, astropy (FITS), nexusformat (Diamond NEXUS files),
PyImageTool (Kyle Gordon - https://github.com/kgord831/PyImageTool), and a patched version
of igorpy from Conrad Stansbury (Lanzara Group): https://github.com/chstan/igorpy

First, create a virtual environment with at least python 3.0, and then install the aforementioned
dependencies. (Try to install from conda forge where possible: conda install -c conda-forge)

For PyImageTool, the installation instructions can be found at the github link above and should be
installed in the same virtual environment as ARPYS.

Same goes for the patched version of igorpy from Conrad Stansbury. (Necessary for loading MERLIN .pxt files)

"Garrison" branch includes Garrison's MDC fitting packages and subroutines. 
