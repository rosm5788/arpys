# ARPYS

Angle resolved Python spectroscopy - Dessau Group

Current, primary brand which is up to date and maintained is the "experimental" branch. External
users, please use "experimental" for your installation.

Dependencies: 
numpy, scipy, xarray, matplotlib, pandas, astropy (FITS), nexusformat (Diamond NEXUS files),
PyImageTool, igorpy (forked/updated versions available at https://github.com/rosm5788)

Installation:
- First, create a virtual environment with at least python 3.0, and then install the aforementioned
dependencies. (Try to install from conda forge where possible: conda install -c conda-forge)

- For PyImageTool and igorpy, the installation instructions can be found at the github link above and should be
installed in the same virtual environment as ARPYS. 

- For beginners, you may download the repo as a .zip and extract it somewhere easy to find locally. 
Then, navigate to the subfolder containing the setup.py file, and run "pip install ./" in your python terminal.
You will repeat this process for the arpys repo, the igorpy repo, and the pyimagetool repos identically.

"Garrison" branch includes Garrison's MDC fitting packages and subroutines, although has not been maintained since his departure.
