[![Anaconda-Server Badge](https://anaconda.org/opcon/plutokore/badges/version.svg)](https://anaconda.org/opcon/plutokore)

**PlutoKore:** A python module that helps with analysing PLUTO simulation output files.

# Features:

* Can load PLUTO output files
* Can calculate cell volume and area
* Can calculate energy components per cell
* Can calculate feedback efficiency
* Can calculate radio luminosity, flux density and surface brightness
* Classes for Makino and King environments, which calculate density profile
* AstroJet class for calculating length scales in different environments
* Configuration helpers for verifying simulation setup and enabling easy simulation cataloguing
* Has methods for doing some specific sorts of plots
* Can create movies

# Installing

## Anaconda

PlutoKore is uploaded as an Anaconda package at [https://anaconda.org/opcon/plutokore](https://anaconda.org/opcon/plutokore), and can be installed with:

```
conda install -c opcon plutokore
```

## Setup.py

PlutoKore can be installed through `setup.py` by downloading/cloning the git repository, and then running

```
python setup.py install
```

This installs PlutoKore through the pip package manager.

PlutoKore can also be used without installing, as long as the outer `plutokore` folder is in the Python package path.

## Requirements

The requirements should be handled automatically if installing from Anaconda or through `setup.py`.

For installing manually, the requirements are as follows:

* numpy
* astropy
* matplotlib
* tabulate
* contextlib2
* h5py
* pyyaml
* scipy
* future (for Python 2 compatibility)
* jupyter (for some interactive things)
* pytest (for tests)
* pytest-runner (for tests)
