#!/bin/env python3

import os
import sys
if os.path.exists(os.path.expanduser('~/plutokore')):
    sys.path.append(os.path.expanduser('~/plutokore'))
else:
    sys.path.append(os.path.expanduser('~/uni/plutokore'))
import plutokore as pk
import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plot
import numpy as np
import argparse
from plutokore import radio
from numba import jit
from astropy.convolution import convolve, Gaussian2DKernel
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
import pathlib

def calculate_total_luminosity(*, sim_dir, out_dir, output_number):
    redshift=0.1
    beamsize=5 * u.arcsec
    pixel_size = 1.8 * u.arcsec
    vmin = -3.0
    vmax = 2.0

    # load sim config
    uv, env, jet = pk.configuration.load_simulation_info(sim_dir + 'config.yaml')

    # create our figure
    fig, ax = plot.subplots(figsize=(2, 2))

    # calculate beam radius
    sigma_beam = (beamsize / 2.355)

    # calculate kpc per arcsec
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(redshift).to(u.kpc / u.arcsec)

    # load timestep data file
    d = pk.simulations.load_timestep_data(output_number, sim_dir)

    # calculate luminosity and unraytraced flux
    l = radio.get_luminosity(d, uv, redshift, beamsize)

    return l.sum()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('simulation_directory', help='Simulation directory', type=str)
    parser.add_argument('output_directory', help='Output directory', type=str)
    parser.add_argument('output', help='Output number', type=int, nargs = '+')
    args = parser.parse_args()

    # create output directory if needed
    pathlib.Path(args.output_directory).mkdir(parents = True, exist_ok = True)

    for i in args.output:
        tot = calculate_total_luminosity(sim_dir = args.simulation_directory,
                out_dir = args.output_directory,
                output_number = i)
        print(tot)

if __name__ == '__main__':
    main()
