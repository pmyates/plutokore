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

@jit(nopython=True)
def raytrace_surface_brightness(r, θ, x, y, z, raytraced_values, original_values):
    φ = 0
    rmax = np.max(r)
    θmax = np.max(θ)
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
                θi = np.arccos(z[z_index] / ri)
                if θi > θmax:
                    continue
                φi = 0 # Don't care about φi!!

                chord_length = np.abs(np.arctan2(y[y_index], x[x_index] + x_half_step) - np.arctan2(y[y_index], x[x_index] - x_half_step))

                # Now find index in r and θ arrays corresponding to this point
                r_index = np.argmax(r>ri)
                θ_index = np.argmax(θ>θi)
                # Only add this if we have not already visited this cell (twice)
                if visited[r_index, θ_index] <= 1:
                    raytraced_values[x_index, z_index] += original_values[r_index, θ_index] * chord_length * pi2_recip
                    visited[r_index, θ_index] += 1
    #return raytraced_values
    return

def plot_sb(*, sim_dir, out_dir, output_number, xmax, ymax):
    xlim = [-xmax, xmax]
    ylim = [-ymax, ymax]
    redshift=0.05
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
        θ=d.x2,
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

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    contour_color = 'cyan'
    contour_linewidth = 0.33
    contour_levels = [-2, -1, 0, 1, 2] # Contours start at 10 μJy
    contour_linestyles = ['dashed', 'dashed', 'solid', 'solid', 'solid']

    im = ax.pcolormesh(
        X1,
        X2,
        np.log10(flux.to(u.mJy).value).T,
        shading='flat',
        vmin=vmin,
        vmax=vmax,
        cmap='afmhot')
    im.set_rasterized(True)
    ax.contour(X1, X2, np.log10(flux.to(u.mJy).value).T, contour_levels, linewidths=contour_linewidth, colors=contour_color, linestyles=contour_linestyles)

    im = ax.pcolormesh(
        -X1,
        X2,
        np.log10(flux.to(u.mJy).value).T,
        shading='flat',
        vmin=vmin,
        vmax=vmax,
        cmap='afmhot')
    im.set_rasterized(True)
    ax.contour(-X1, X2, np.log10(flux.to(u.mJy).value).T, contour_levels, linewidths=contour_linewidth, colors=contour_color, linestyles=contour_linestyles)

    im = ax.pcolormesh(
        X1,
        -X2,
        np.log10(flux.to(u.mJy).value).T,
        shading='flat',
        vmin=vmin,
        vmax=vmax,
        cmap='afmhot')
    im.set_rasterized(True)
    ax.contour(X1, -X2, np.log10(flux.to(u.mJy).value).T, contour_levels, linewidths=contour_linewidth, colors=contour_color, linestyles=contour_linestyles)

    im = ax.pcolormesh(
        -X1,
        -X2,
        np.log10(flux.to(u.mJy).value).T,
        shading='flat',
        vmin=vmin,
        vmax=vmax,
        cmap='afmhot')
    im.set_rasterized(True)
    ax.contour(-X1, -X2, np.log10(flux.to(u.mJy).value).T, contour_levels, linewidths=contour_linewidth, colors=contour_color, linestyles=contour_linestyles)

    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')

    pk.plot.savefig(f'{out_dir}/{output_number:04}', fig, png=True, dpi=500, kwargs={
        'bbox_inches': 'tight',
        'pad_inches': 0})
    plot.close();

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('simulation_directory', help='Simulation directory', type=str)
    parser.add_argument('output_directory', help='Output directory', type=str)
    parser.add_argument('output', help='Output number', type=int)
    parser.add_argument('xmax', help='Max x limit (in arcsec)', type=float)
    parser.add_argument('ymax', help='Max y limit (in arcsec)', type=float)
    args = parser.parse_args()

    # create output directory if needed
    pathlib.Path(args.output_directory).mkdir(parents = True, exist_ok = True)

    plot_sb(sim_dir = args.simulation_directory,
            out_dir = args.output_directory,
            output_number = args.output,
            xmax = args.xmax,
            ymax = args.ymax)


if __name__ == '__main__':
    main()
