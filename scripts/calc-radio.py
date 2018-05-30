#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from astropy.table import QTable
import pathlib
import h5py

@jit(nopython=True, cache=True)
def raytrace_surface_brightness(r, theta, x, y, z, raytraced_values, original_values):
    phi = 0
    rmax = np.max(r)
    thetamax = np.max(theta)
    x_half_step = (x[1] - x[0]) * 0.5
    pi2_recip = (1 / (2 * np.pi))

    visited = np.zeros(original_values.shape)
    for x_index in range(len(x)):
        for z_index in range(len(z)):
            visited[:,:] = 0
            for y_index in range(len(y)):
                # Calculate the coordinates of this point
                ri = np.sqrt(x[x_index] **2 + y[y_index] ** 2 + z[z_index] ** 2)
                if ri == 0:
                    continue
                if ri > rmax:
                    continue
                thetai = np.arccos(z[z_index] / ri)
                if thetai > thetamax:
                    continue
                phii = 0 # Don't care about phii!!

                chord_length = np.abs(np.arctan2(y[y_index], x[x_index] + x_half_step) - np.arctan2(y[y_index], x[x_index] - x_half_step))

                # Now find index in r and theta arrays corresponding to this point
                r_index = np.argmax(r>ri)
                theta_index = np.argmax(theta>thetai)
                # Only add this if we have not already visited this cell (twice)
                if visited[r_index, theta_index] <= 1:
                    raytraced_values[x_index, z_index] += original_values[r_index, theta_index] * chord_length * pi2_recip
                    visited[r_index, theta_index] += 1
    #return raytraced_values
    return

def calculate_surface_brightness(*, sim_dir, output_number, xmax, ymax, redshift, beamsize, pixel_size):
    xlim = [-xmax, xmax]
    ylim = [-ymax, ymax]

    # load sim config
    uv, env, jet = pk.configuration.load_simulation_info(os.path.join(sim_dir, 'config.yaml'))

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
    f = radio.get_flux_density(l, redshift).to(u.Jy).value

    # calculate raytracing grid
    xmax = (((xlim[1] * u.arcsec + pixel_size) * kpc_per_arcsec) / uv.length).si
    xstep = (pixel_size * kpc_per_arcsec / uv.length).si
    zmax = (((ylim[1] * u.arcsec + pixel_size) * kpc_per_arcsec) / uv.length).si
    zstep = (pixel_size * kpc_per_arcsec / uv.length).si
    ymax = max(xmax, zmax)
    ystep = min(xstep, zstep)
    ystep = 0.5

    x = np.arange(0, xmax, xstep)
    z = np.arange(0, zmax, zstep)
    y = np.arange(-ymax, ymax, ystep)
    raytraced_flux = np.zeros((x.shape[0], z.shape[0]))

    # raytrace surface brightness
    raytrace_surface_brightness(
        r=d.x1,
        theta=d.x2,
        x=x,
        y=y,
        z=z,
        original_values=f,
        raytraced_values=raytraced_flux
    )

    raytraced_flux = raytraced_flux * u.Jy

    # beam information
    area_beam_kpc2 = (np.pi * (sigma_beam * kpc_per_arcsec)
                      **2).to(u.kpc**2)
    beams_per_cell = (((pixel_size * kpc_per_arcsec) ** 2) / area_beam_kpc2).si

    raytraced_flux /= beams_per_cell

    beam_kernel = Gaussian2DKernel(sigma_beam.value)
    flux = convolve(raytraced_flux.to(u.Jy), beam_kernel, boundary='extend') * u.Jy

    X1 = x * (uv.length / kpc_per_arcsec).to(u.arcsec).value
    X2 = z * (uv.length / kpc_per_arcsec).to(u.arcsec).value

    return (X1, X2, flux.to(u.mJy), l.sum())

def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('simulation_directory', help='Simulation directory', type=str)
    parser.add_argument('output_directory', help='Output directory', type=str)
    parser.add_argument('outputs', help='Output numbers', type=int, nargs='+')
    parser.add_argument('--redshift', help='Redshift', type=float, default=0.05)
    parser.add_argument('--beamsize', help='Observing beam size (arcsec)', type=float, default=5)
    parser.add_argument('--pixel_size', help='Observing pixel size (arcsec)', type=float, default=1.8)
    parser.add_argument('--xmax', help='Maximum x coordinate (arcsec)', type=float, default=90)
    parser.add_argument('--ymax', help='Maximum y coordinate (arcsec)', type=float, default=300)
    args = parser.parse_args()

    # create output directory if needed
    pathlib.Path(args.output_directory).mkdir(parents = True, exist_ok = True)

    for i in args.outputs:
        (x1_sb, x2_sb, sb, lum_sum) = calculate_surface_brightness(sim_dir = args.simulation_directory,
                output_number = i,
                xmax = args.xmax,
                ymax = args.ymax,
                redshift = args.redshift,
                beamsize = args.beamsize * u.arcsec,
                pixel_size = args.pixel_size * u.arcsec)
        with h5py.File(f'{args.output_directory}/radio.{i:04}.hdf5', 'w') as f:
            grp = f.create_group(f'{i:04}')

            grp.create_dataset('X1', data=x1_sb)
            grp.create_dataset('X2', data=x2_sb)
            grp.create_dataset('sb', data=sb)

            grp.attrs['total_luminosity'] = lum_sum
            grp.attrs['redshift'] = args.redshift
            grp.attrs['beamsize'] = args.beamsize * u.arcsec
            grp.attrs['pixel_size'] = args.pixel_size * u.arcsec
            grp.attrs['output'] = i
            grp.attrs['xmax'] = args.xmax
            grp.attrs['ymax'] = args.ymax

if __name__ == '__main__':
    main()
