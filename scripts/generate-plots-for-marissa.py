#!/usr/bin/env python3

""" Generate some simple plots from simulations

This script generates a few simple plots from the given simulation.
The goal is to highlight both the 1st and 2nd outburst in a 4-outburst
simulation.

The plots generated are:

* Density (full-plane reflected)
* Tracers (full-plane reflected)
* Surface brightness (full-plane reflected)

Changes:

* Inital version (Patrick, 27.10.2018)
"""

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
from matplotlib.colors import LinearSegmentedColormap
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
import code

def create_plots(*, sim_dir, plot_dir, output_number, sim_info, observing_properties, plot_properties):
    """
    This function creates the three plots we want
    """
    # load the simulation information
    uv = sim_info['uv']
    env = sim_info['env']
    jet = sim_info['jet']

    # load simulation
    sim_data = pk.simulations.load_timestep_data(output_number, sim_dir, mmap=True)

    sim_data.x2r[-1] = np.pi
    rr, tt = np.meshgrid(sim_data.x1r, sim_data.x2r)
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)

    rmesh, tmesh = np.meshgrid(sim_data.x1, sim_data.x2)

    # x, y = pk.simulations.sphericaltocartesian(sim_data, rotation=plot_properties['rotation'])
    x = x * uv.length
    y = y * uv.length
    if plot_properties['plot_in_arcsec']:
        x = (x * observing_properties['kpc2arcsec']).to(u.arcsec)
        y = (y * observing_properties['kpc2arcsec']).to(u.arcsec)

    # let's check if this simulation is quarter-plane or half-plane (2D)
    if (sim_data.geometry == 'SPHERICAL') and (len(sim_data.nshp) == 2):
        pass
    else:
        quit('Unsupported geometry and/or dimensions')
    is_quarter_plane = (sim_data.x2[-1] - sim_data.x2[0]) < (3.0*np.pi / 4.0)

    # plot density
    f,a = setup_figure(sim_time = (sim_data.SimTime * uv.time).to(u.Myr), plot_properties = plot_properties, observing_properties = observing_properties)
    rho = sim_data.rho * uv.density.to(u.kg / u.m ** 3).value

    im = a.pcolormesh(x, y, np.log10(rho.T), vmin=-27, vmax=-23, rasterized = True, edgecolors = 'none', shading = 'flat')
    im = a.pcolormesh(x, -y, np.log10(rho.T), vmin=-27, vmax=-23, rasterized = True, edgecolors = 'none', shading = 'flat')
    if is_quarter_plane:
        im = a.pcolormesh(-x, y, np.log10(rho.T), vmin=-27, vmax=-23, rasterized = True, edgecolors = 'none', shading = 'flat')
        im = a.pcolormesh(-x, -y, np.log10(rho.T), vmin=-27, vmax=-23, rasterized = True, edgecolors = 'none', shading = 'flat')
    cb = f.colorbar(im)
    cb.set_label('Density [log10 kg cm^-3]')

    save_figure(
        fig=f,
        ax=a,
        cbx=cb,
        plot_properties=plot_properties,
        fig_path=os.path.join(plot_dir, f'density_{output_number:02d}.png'),
    )
    plot.close(f)

    # plot pressure
    f,a = setup_figure(sim_time = (sim_data.SimTime * uv.time).to(u.Myr), plot_properties = plot_properties, observing_properties = observing_properties)
    prs = sim_data.prs * uv.pressure.to(u.Pa).value
    im = a.pcolormesh(x, y, np.log10(prs.T), vmin=-16, vmax=-11.5, rasterized = True, edgecolors = 'none', shading = 'flat')
    im = a.pcolormesh(x, -y, np.log10(prs.T), vmin=-16, vmax=-11.5, rasterized = True, edgecolors = 'none', shading = 'flat')
    if is_quarter_plane:
        im = a.pcolormesh(-x, -y, np.log10(prs.T), vmin=-16, vmax=-11.5, rasterized = True, edgecolors = 'none', shading = 'flat')
        im = a.pcolormesh(-x, y, np.log10(prs.T), vmin=-16, vmax=-11.5, rasterized = True, edgecolors = 'none', shading = 'flat')
    cb = f.colorbar(im)
    cb.set_label('Pressure [log10 Pa]')
    save_figure(
        fig=f,
        ax=a,
        cbx=cb,
        plot_properties=plot_properties,
        fig_path=os.path.join(plot_dir, f'pressure_{output_number:02d}.png'),
    )
    plot.close(f)

    # plot jet velocity
    f,a = setup_figure(sim_time = (sim_data.SimTime * uv.time).to(u.Myr), plot_properties = plot_properties, observing_properties = observing_properties)
    vx = (sim_data.vx1 * (np.sin(tmesh.T)) + rmesh.T * sim_data.vx2 * (np.cos(tmesh.T))) * uv.speed.to(u.km / u.s).value
    vx = sim_data.vx1 * uv.speed.to(u.km / u.s).value
    # import ipdb; ipdb.set_trace()
    im = a.pcolormesh(x, y, vx.T, vmin=None, vmax=None, rasterized = True, edgecolors = 'none', shading = 'flat')
    im = a.pcolormesh(x, -y, vx.T, vmin=None, vmax=None, rasterized = True, edgecolors = 'none', shading = 'flat')
    if is_quarter_plane:
        im = a.pcolormesh(-x, -y, vx.T, vmin=-16, vmax=-11.5, rasterized = True, edgecolors = 'none', shading = 'flat')
        im = a.pcolormesh(-x, y, vx.T, vmin=-16, vmax=-11.5, rasterized = True, edgecolors = 'none', shading = 'flat')
    cb = f.colorbar(im)
    cb.set_label('Velocity [km s^-1]')

    save_figure(
        fig=f,
        ax=a,
        cbx=cb,
        plot_properties=plot_properties,
        fig_path=os.path.join(plot_dir, f'velocity_{output_number:02d}.png'),
    )
    plot.close(f)

    # plot tracer
    f,a = setup_figure(sim_time = (sim_data.SimTime * uv.time).to(u.Myr), plot_properties = plot_properties, observing_properties = observing_properties)

    tracer_count = pk.simulations.get_tracer_count_data(sim_data)
    tr1 = sim_data.tr1

    im1 = a.pcolormesh(x, y, tr1.T, vmin=0, vmax=1, cmap='Blues_alpha', rasterized = True, edgecolors = 'none', shading = 'flat')
    im1 = a.pcolormesh(x, -y, tr1.T, vmin=0, vmax=1, cmap='Blues_alpha', rasterized = True, edgecolors = 'none', shading = 'flat')
    if is_quarter_plane:
        im1 = a.pcolormesh(-x, -y, tr1.T, vmin=0, vmax=1, cmap='Blues_alpha', rasterized = True, edgecolors = 'none', shading = 'flat')
        im1 = a.pcolormesh(-x, y, tr1.T, vmin=0, vmax=1, cmap='Blues_alpha', rasterized = True, edgecolors = 'none', shading = 'flat')

    # only plot second tracer if we have more than one!
    if tracer_count > 1:
        tr2 = sim_data.tr2
        im1 = a.pcolormesh(x, y, tr2.T, vmin=0, vmax=1, cmap='Reds_alpha', rasterized = True, edgecolors = 'none', shading = 'flat')
        im1 = a.pcolormesh(x, -y, tr2.T, vmin=0, vmax=1, cmap='Reds_alpha', rasterized = True, edgecolors = 'none', shading = 'flat')
        if is_quarter_plane:
            im1 = a.pcolormesh(-x, -y, tr2.T, vmin=0, vmax=1, cmap='Reds_alpha', rasterized = True, edgecolors = 'none', shading = 'flat')
            im1 = a.pcolormesh(-x, y, tr2.T, vmin=0, vmax=1, cmap='Reds_alpha', rasterized = True, edgecolors = 'none', shading = 'flat')

    save_figure(
        fig=f,
        ax=a,
        cbx=cb,
        plot_properties=plot_properties,
        fig_path=os.path.join(plot_dir, f'tracers_{output_number:02d}.png'),
    )
    plot.close(f)

    f,a = setup_figure(sim_time = (sim_data.SimTime * uv.time).to(u.Myr), plot_properties = {**plot_properties, 'plot_in_arcsec': True}, observing_properties = observing_properties)
    (X, Y, sb) = calculate_surface_brightness(
        sim_data = sim_data,
        uv = uv,
        observing_properties = observing_properties,
        is_quarter_plane = is_quarter_plane,
        do_convolve = True,
    )

    im = a.pcolormesh(Y, X, np.log10(sb.value), vmin=-3, vmax=2, rasterized = True, edgecolors = 'face', shading = 'flat')
    im = a.pcolormesh(Y, -X, np.log10(sb.value), vmin=-3, vmax=2, rasterized = True, edgecolors = 'face', shading = 'flat')
    if is_quarter_plane:
        im = a.pcolormesh(-Y, X, np.log10(sb.value), vmin=-3, vmax=2, rasterized = True, edgecolors = 'face', shading = 'flat')
        im = a.pcolormesh(-Y, -X, np.log10(sb.value), vmin=-3, vmax=2, rasterized = True, edgecolors = 'face', shading = 'flat')
    cb = f.colorbar(im)
    cb.set_label('Surface Brightness [log10 mJy beam^-1]')
    save_figure(
        fig=f,
        ax=a,
        cbx=cb,
        plot_properties=plot_properties,
        fig_path=os.path.join(plot_dir, f'sb_{output_number:02d}.png'),
    )
    plot.close(f)

def setup_figure(*, sim_time, plot_properties, observing_properties):
    fig,ax = plot.subplots(figsize=(10,5))

    ax.set_xlim(observing_properties['xlim'].value)
    ax.set_ylim(observing_properties['ylim'].value)

    if plot_properties['plot_in_arcsec']:
        ax.set_xlabel('X ["]')
        ax.set_ylabel('Y ["]')
    else:
        ax.set_xlabel('X [kpc]')
        ax.set_ylabel('Y [kpc]')
    ax.set_title(f'{sim_time:0.02f}')
    ax.set_aspect('equal')

    return fig,ax

def save_figure(*, fig, ax, cbx, plot_properties, fig_path):
    if plot_properties['fluff'] is False:
        if cbx.ax in fig.axes:
            fig.delaxes(cbx.ax)
        ax.set_title('')
        ax.set_position([0, 0, 1, 1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
    fig.savefig(fig_path, dpi=plot_properties['dpi'], bbox_inches='tight')

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

def calculate_surface_brightness(*, sim_data, uv, observing_properties, do_convolve, is_quarter_plane):
    xlim = observing_properties['ylim']
    ylim = observing_properties['xlim']

    # calculate beam radius
    sigma_beam = (observing_properties['beamwidth'] / 2.355)

    # calculate kpc per arcsec
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(observing_properties['redshift']).to(u.kpc / u.arcsec)

    # load timestep data file
    d = sim_data

    # calculate luminosity and unraytraced flux
    l = radio.get_luminosity(d, uv, observing_properties['redshift'], observing_properties['beamwidth'])
    f = radio.get_flux_density(l, observing_properties['redshift']).to(u.Jy).value

    # calculate raytracing grid
    xmax = ((xlim[1] + observing_properties['pixelsize'] * kpc_per_arcsec) / uv.length).si
    zmax = ((ylim[1] + observing_properties['pixelsize'] * kpc_per_arcsec) / uv.length).si
    if not is_quarter_plane:
        xmin = ((xlim[0] - observing_properties['pixelsize'] * kpc_per_arcsec) / uv.length).si
        zmin = ((ylim[0] - observing_properties['pixelsize'] * kpc_per_arcsec) / uv.length).si
    xstep = (observing_properties['pixelsize'] * kpc_per_arcsec / uv.length).si
    zstep = (observing_properties['pixelsize'] * kpc_per_arcsec / uv.length).si
    ymax = max(xmax, zmax)
    ystep = min(xstep, zstep)
    # ystep = ((0.25 * u.kpc) / uv.length).si

    if is_quarter_plane:
        x = np.arange(0, xmax, xstep)
        z = np.arange(0, zmax, zstep)
    else:
        x = np.arange(0, xmax, xstep)
        z = np.arange(zmin, zmax, zstep)
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
    beams_per_cell = (((observing_properties['pixelsize'] * kpc_per_arcsec) ** 2) / area_beam_kpc2).si

    raytraced_flux /= beams_per_cell

    beam_kernel = Gaussian2DKernel(sigma_beam.value)
    if do_convolve:
        flux = convolve(raytraced_flux.to(u.Jy), beam_kernel, boundary='extend') * u.Jy
    else:
        flux = raytraced_flux

    X1 = x * (uv.length / kpc_per_arcsec).to(u.arcsec).value
    X2 = z * (uv.length / kpc_per_arcsec).to(u.arcsec).value

    return (X1, X2, flux.to(u.mJy))

def create_alpha_colormap(*, name):
    ncolors = 256
    color_array = plot.get_cmap(name)(range(ncolors))
    color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)
    map_object = LinearSegmentedColormap.from_list(name=f'{name}_alpha', colors=color_array)
    plot.register_cmap(cmap=map_object)

def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('simulation_directory', help='Simulation directory', type=str)
    parser.add_argument('output_directory', help='Output directory', type=str)
    parser.add_argument('outputs', help='Output numbers', type=int, nargs='+')
    parser.add_argument('--trc_cutoff', help='Tracer cutoff', type=float, default=1e-14)
    parser.add_argument('--redshift', help='Redshift value', type=float, default=0.05)
    parser.add_argument('--beamwidth', help='Observing beam width [arcsec]', type=float, default=5)
    parser.add_argument('--pixelsize', help='Observing pixel size [arcsec]', type=float, default=1.8)
    parser.add_argument('--xlim', help='X limits [kpc]', type=float, nargs=2, default=[-60,60])
    parser.add_argument('--ylim', help='Y limits [kpc]', type=float, nargs=2, default=[-60,60])
    parser.add_argument('--plot_in_arcsec', help='Plot axes in arsec')
    parser.add_argument('--rotation', help='Rotation of output', type=float, default=np.pi / 2)
    parser.add_argument('--dpi', help='DPI to save figure at', type=float, default=300)
    parser.add_argument('--no_fluff', help='Save the figure without any axes labels, ticks, or titles', action='store_true')
    args = parser.parse_args()

    # Update observing properties
    observing_properties = {
        'redshift': args.redshift,
        'beamwidth': args.beamwidth * u.arcsec,
        'pixelsize': args.pixelsize * u.arcsec,
        'xlim': args.xlim * u.kpc,
        'ylim': args.ylim * u.kpc,
        'kpc2arcsec': 1.0/cosmo.kpc_proper_per_arcmin(args.redshift).to(u.kpc / u.arcsec)
    }

    # update plot propterties
    plot_properties = {
        'plot_in_arcsec': args.plot_in_arcsec,
        'rotation': args.rotation,
        'dpi': args.dpi,
        'fluff': not args.no_fluff,
    }

    # load the simulation information
    uv, env, jet = pk.configuration.load_simulation_info(os.path.join(args.simulation_directory, 'config.yaml'))
    sim_info = {
        'uv': uv,
        'env': env,
        'jet': jet,
    }

    print('Generating plots for the following outputs:')
    print(args.outputs)
    print()

    print('Observing propreties are:')
    print(f'> r: {observing_properties["redshift"]}, beamwidth: {observing_properties["beamwidth"]}, pixelsize: {observing_properties["pixelsize"]}')
    print(f'> xlim: {observing_properties["xlim"]}, ylim: {observing_properties["ylim"]}')
    print()

    print('The environment and jet properties are:')
    print(f'> Environment: {type(env).__name__}, halo mass = {np.log10(env.halo_mass.value)}, central density = {env.central_density}')
    print(f'> Jet: power = {jet.Q}, density = {jet.rho_0}, mach number = {jet.M_x}, half-opening angle = {np.rad2deg(jet.theta)}')
    print()

    # create output directory if needed
    pathlib.Path(args.output_directory).mkdir(parents = True, exist_ok = True)

    # Let's generate our custom colormaps
    create_alpha_colormap(name='Blues')
    create_alpha_colormap(name='Reds')

    for i in args.outputs:
        create_plots(
            sim_dir = args.simulation_directory,
            plot_dir = args.output_directory,
            output_number = i,
            sim_info = sim_info,
            observing_properties = observing_properties,
            plot_properties = plot_properties,
        )

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
