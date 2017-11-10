#!/bin/env python3

import sys
import os
import matplotlib as mpl
#mpl.use('Qt5Agg')
mpl.use('Agg')
import matplotlib.pyplot as plot
import numpy as np
try:
    import plutokore as pk
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import plutokore as pk
import argparse
import yaml
from astropy import units as u

def plot_mach200():
    directory = '/u/pmyates/simulations/runs/kunanyi/honours/m14.5-M200/'
    c200 = pk.configuration.load_simulation_info(directory + 'config.yaml')
    dn = 472
    data = pk.simulations.load_timestep_data(dn, directory)
    xx, yy = pk.simulations.sphericaltocartesian(data)
    xx = xx * c200[0].length.value
    yy = yy * c200[0].length.value

    f,((a1,a2),(a3,a4)) = plot.subplots(2, 2, sharex='row', sharey='col', figsize=(10,10), subplot_kw = {'xlim': (0, 30), 'ylim': (0, 30), 'aspect': 'equal'})

    a1.pcolormesh(xx, yy, np.log10(data.rho.T), rasterized=True, shading = 'flat', edgecolors = 'face')
    a2.pcolormesh(xx, yy, np.log10(data.prs.T), rasterized=True, shading = 'flat', edgecolors = 'face')
    a3.pcolormesh(xx, yy, data.vx1.T, rasterized=True, shading = 'flat', edgecolors = 'face')
    a4.pcolormesh(xx, yy, data.tr1.T, rasterized=True, shading = 'flat', edgecolors = 'face')

    a1.set_title('density')
    a2.set_title('pressure')
    a3.set_title('radial velocity')
    a4.set_title('jet tracer')

    # plot.show()

    fname = directory + f'{dn}-rho.png'
    f.savefig(fname, dpi=400)

def get_times(sim_dir):
    with open(os.path.join(sim_dir, 'dbl.out'), "r") as f_var:
        tlist = []
        for line in f_var.readlines():
            tlist.append(float(line.split()[1]))
    return np.asarray(tlist)

def plot_morphology_comparison(d25, d200):
    def internal(f, a1, a2, a3, a4):
        c25 = pk.configuration.load_simulation_info(d25 + 'config.yaml') # Config for mach 25
        c200 = pk.configuration.load_simulation_info(d200 + 'config.yaml') # Config for mach 200

        times25 = get_times(d25) * c25[0].time
        times200 = get_times(d200) * c200[0].time

        comp_time = times25[-1] if times25[-1] < times200[-1] else times200[-1]

        on200 = np.where(times200 >= comp_time)[0][0]
        on25 = np.where(times25 >= comp_time)[0][0]
        data25 = pk.simulations.load_timestep_data(on25, d25, mmap=True)
        data200 = pk.simulations.load_timestep_data(on200, d200, mmap=True)

        xx25, yy25 = pk.simulations.sphericaltocartesian(data25)
        xx25 *= c25[0].length.value
        yy25 *= c25[0].length.value

        xx200, yy200 = pk.simulations.sphericaltocartesian(data200)
        xx200 *= c200[0].length.value
        yy200 *= c200[0].length.value

        im1 = a1.pcolormesh(-xx25, yy25, np.log10(data25.rho * c25[0].density.to(u.g / u.cm ** 3).value).T, rasterized=True, shading = 'flat', edgecolors = 'face')
        im2 = a1.pcolormesh(xx200, yy200, np.log10(data200.rho * c200[0].density.to(u.g / u.cm ** 3).value).T, rasterized=True, shading = 'flat', edgecolors = 'face')
        im2.set_clim(im1.get_clim())

        im1 = a2.pcolormesh(-xx25, yy25, np.log10(data25.prs * c25[0].pressure.to(u.Pa).value).T, rasterized=True, shading = 'flat', edgecolors = 'face')
        im2 = a2.pcolormesh(xx200, yy200, np.log10(data200.prs * c200[0].pressure.to(u.Pa).value).T, rasterized=True, shading = 'flat', edgecolors = 'face')
        im2.set_clim(im1.get_clim())

        im1 = a3.pcolormesh(-xx25, yy25, (data25.vx1 * c25[0].speed.to(u.km / u.s).value).T, rasterized=True, shading = 'flat', edgecolors = 'face')
        im2 = a3.pcolormesh(xx200, yy200, (data200.vx1 * c200[0].speed.to(u.km / u.s).value).T, rasterized=True, shading = 'flat', edgecolors = 'face')
        im2.set_clim(im1.get_clim())

        im1 = a4.pcolormesh(-xx25, yy25, data25.tr1.T, rasterized=True, shading = 'flat', edgecolors = 'face')
        im2 = a4.pcolormesh(xx200, yy200, data200.tr1.T, rasterized=True, shading = 'flat', edgecolors = 'face')
        im2.set_clim(im1.get_clim())

        f.suptitle('M25 at {0:02.2f} (left) and M200 at {1:02.2f} (right)'.format(data25.SimTime * c25[0].time, data200.SimTime * c200[0].time))
        return comp_time

    f,((a1,a2),(a3,a4)) = plot.subplots(2, 2, sharex='row', sharey='col', figsize=(10,10), subplot_kw = {'xlim': (-30, 30), 'ylim': (0, 60), 'aspect': 'equal', 'xlabel': 'X [kpc]', 'ylabel': 'Y [kpc]'})

    a1.set_title('density')
    a2.set_title('pressure')
    a3.set_title('radial velocity')
    a4.set_title('jet tracer')

    comp_time = internal(f, a1, a2, a3, a4)

    f.savefig('morphology-{0:02.2f}.png'.format(comp_time), bbox_inches='tight', dpi=400)
    plot.close(f)


def plot_lobe_lengths(d25, d200):
    def internal(f, a):
        trc_cutoff = 0.001
        c25 = pk.configuration.load_simulation_info(d25 + 'config.yaml') # Config for mach 25
        c200 = pk.configuration.load_simulation_info(d200 + 'config.yaml') # Config for mach 200

        o25 = np.arange(1, 149, 4)
        o200 = np.arange(1, 282, 4)

        calc25 = np.zeros((len(o25), 2))
        calc200 = np.zeros((len(o200), 2))

        for ind, i in enumerate(o25):
            d = pk.simulations.load_timestep_data(i, d25, mmap=True)
            m = d.x1[np.where(d.tr1[:,0] > trc_cutoff)[0][-1]]*c25[0].length.value
            calc25[ind, :] = [m, d.SimTime * c25[0].time.value]
        for ind, i in enumerate(o200):
            d = pk.simulations.load_timestep_data(i, d200, mmap=True)
            m = d.x1[np.where(d.tr1[:,0] > trc_cutoff)[0][-1]]*c200[0].length.value
            calc200[ind, :] = [m, d.SimTime * c200[0].time.value]

        a.plot(calc25[:,1], calc25[:,0], label='M25')
        a.plot(calc200[:,1], calc200[:,0], label='M200')

    f, a = plot.subplots(1,1)

    internal(f, a)

    a.set_xlabel('Time [Myr]')
    a.set_ylabel('Lobe length [kpc]')
    a.legend()
    f.suptitle('Lobe length for 2d simulations')
    f.savefig('length.png', bbox_inches='tight', dpi=200)
    plot.close(f)

def plot_lobe_volumes(d25, d200):
    def internal(f, a):
        trc_cutoff = 0.001
        c25 = pk.configuration.load_simulation_info(d25 + 'config.yaml') # Config for mach 25
        c200 = pk.configuration.load_simulation_info(d200 + 'config.yaml') # Config for mach 200

        o25 = np.arange(1, 149, 4)
        o200 = np.arange(1, 282, 4)

        calc25 = np.zeros((len(o25), 2))
        calc200 = np.zeros((len(o200), 2))

        for ind, i in enumerate(o25):
            d = pk.simulations.load_timestep_data(i, d25, mmap=True)
            v = pk.simulations.calculate_cell_volume(d)
            s = np.sum(v * (d.tr1 > trc_cutoff)) * (c25[0].length ** 3).value
            calc25[ind, :] = [s, d.SimTime * c25[0].time.value]
        for ind, i in enumerate(o200):
            d = pk.simulations.load_timestep_data(i, d200, mmap=True)
            v = pk.simulations.calculate_cell_volume(d)
            s = np.sum(v * (d.tr1 > trc_cutoff)) * (c200[0].length ** 3).value
            calc200[ind, :] = [s, d.SimTime * c200[0].time.value]

        a.plot(calc25[:,1], calc25[:,0], label='M25')
        a.plot(calc200[:,1], calc200[:,0], label='M200')

    f, a = plot.subplots(1,1)

    internal(f, a)

    a.set_xlabel('Time [Myr]')
    a.set_ylabel('Lobe volume [kpc^3]')
    a.legend()
    f.suptitle('Lobe volume for 2d simulations')
    f.savefig('volume.png', bbox_inches='tight', dpi=200)
    plot.close(f)

# Surface brightness
def plot_surface_brightness(sim_id,
                            timestep,
                            unit_values,
                            run_dirs,
                            filename,
                            redshift=0.1,
                            beamsize=5 * u.arcsec,
                            showbeam=True,
                            xlim=(-1, 1),
                            ylim=(-1.5, 1.5),
                            xticks=None,
                            pixel_size=1.8 * u.arcsec,
                            beam_x=0.8,
                            beam_y=0.8,
                            png=False,
                            contours=True,
                            convolve=True,
                            half_plane=False,
                            vmin=-3.0,
                            vmax=2.0,
                            style='flux-plot.mplstyle',
                            no_labels=False,
                            with_hist=True,
                            ):
    from plutokore import radio
    from numba import jit
    from astropy.convolution import convolve, Gaussian2DKernel

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

    fig, ax = newfig(1, 1.8)

    # calculate beam radius
    sigma_beam = (beamsize / 2.355)

    # calculate kpc per arcsec
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(redshift).to(u.kpc / u.arcsec)

    # load timestep data file
    d = pk.simulations.load_timestep_data(timestep, run_dirs[sim_id])

    X1, X2 = pk.simulations.sphericaltocartesian(d)
    X1 = X1 * (unit_values.length / kpc_per_arcsec).to(u.arcsec).value
    X2 = X2 * (unit_values.length / kpc_per_arcsec).to(u.arcsec).value

    l = radio.get_luminosity(d, unit_values, redshift, beamsize)
    f = radio.get_flux_density(l, redshift).to(u.Jy).value
    #sb = radio.get_surface_brightness(f, d, unit_values, redshift, beamsize).to(u.Jy)

    xmax = (((xlim[1] * u.arcsec + pixel_size) * kpc_per_arcsec) / unit_values.length).si
    xstep = (pixel_size * kpc_per_arcsec / unit_values.length).si
    zmax = (((ylim[1] * u.arcsec + pixel_size) * kpc_per_arcsec) / unit_values.length).si
    zstep = (pixel_size * kpc_per_arcsec / unit_values.length).si
    ymax = max(xmax, zmax)
    ystep = min(xstep, zstep)
    ystep = 0.5

    if half_plane:
        x = np.arange(-xmax, xmax, xstep)
        z = np.arange(-zmax, zmax, zstep)
    else:
        x = np.arange(0, xmax, xstep)
        z = np.arange(0, zmax, zstep)
    y = np.arange(-ymax, ymax, ystep)
    raytraced_flux = np.zeros((x.shape[0], z.shape[0]))

    # print(f'xlim in arcsec is {xlim[1]}, xlim in code units is {xlim[1] * u.arcsec * kpc_per_arcsec / unit_values.length}')
    # print(f'zlim in arcsec is {ylim[1]}, zlim in code units is {ylim[1] * u.arcsec * kpc_per_arcsec / unit_values.length}')
    # print(f'xmax is {xmax}, ymax is {ymax}, zmax is {zmax}')
    # print(f'x shape is {x.shape}; y shape is {y.shape}; z shape is {z.shape}')

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
    sigma_beam_arcsec = beamsize / 2.355
    area_beam_kpc2 = (np.pi * (sigma_beam_arcsec * kpc_per_arcsec)
                      **2).to(u.kpc**2)
    beams_per_cell = (((pixel_size * kpc_per_arcsec) ** 2) / area_beam_kpc2).si
    #beams_per_cell = (area_beam_kpc2 / ((pixel_size * kpc_per_arcsec) ** 2)).si

    # radio_cell_areas = np.full(raytraced_flux.shape, xstep * zstep) * (unit_values.length ** 2)

    # n beams per cell
    #n_beams_per_cell = (radio_cell_areas / area_beam_kpc2).si

    raytraced_flux /= beams_per_cell

    stddev = sigma_beam_arcsec / beamsize
    beam_kernel = Gaussian2DKernel(stddev)
    if convolve:
        flux = convolve(raytraced_flux.to(u.Jy), beam_kernel, boundary='extend') * u.Jy
    else:
        flux = raytraced_flux.to(u.Jy)
    #flux = radio.convolve_surface_brightness(raytraced_flux, unit_values, redshift, beamsize)
    #flux = raytraced_flux

    X1 = x * (unit_values.length / kpc_per_arcsec).to(u.arcsec).value
    X2 = z * (unit_values.length / kpc_per_arcsec).to(u.arcsec).value

    # plot data
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    contour_color = 'k'
    contour_linewidth = 0.33
    # contour_levels = [-3, -1, 1, 2]
    contour_levels = [-2, -1, 0, 1, 2] # Contours start at 10 μJy

    with plt.style.context('flux-plot.mplstyle'):
        im = ax.pcolormesh(
            X1,
            X2,
            np.log10(flux.to(u.mJy).value).T,
            shading='flat',
            vmin=vmin,
            vmax=vmax)
        im.set_rasterized(True)
        if contours:
            ax.contour(X1, X2, np.log10(flux.to(u.mJy).value).T, contour_levels, linewidths=contour_linewidth, colors=contour_color)

        im = ax.pcolormesh(
            -X1,
            X2,
            np.log10(flux.to(u.mJy).value).T,
            shading='flat',
            vmin=vmin,
            vmax=vmax)
        im.set_rasterized(True)
        if contours:
            ax.contour(-X1, X2, np.log10(flux.to(u.mJy).value).T, contour_levels, linewidths=contour_linewidth, colors=contour_color)

        if not half_plane:
            im = ax.pcolormesh(
                X1,
                -X2,
                np.log10(flux.to(u.mJy).value).T,
                shading='flat',
                vmin=vmin,
                vmax=vmax)
            im.set_rasterized(True)
            if contours:
                ax.contour(X1, -X2, np.log10(flux.to(u.mJy).value).T, contour_levels, linewidths=contour_linewidth, colors=contour_color)

            im = ax.pcolormesh(
                -X1,
                -X2,
                np.log10(flux.to(u.mJy).value).T,
                shading='flat',
                vmin=vmin,
                vmax=vmax)
            im.set_rasterized(True)
            if contours:
                ax.contour(-X1, -X2, np.log10(flux.to(u.mJy).value).T, contour_levels, linewidths=contour_linewidth, colors=contour_color)

        if with_hist:
            div = make_axes_locatable(ax)
            ax_hist = div.append_axes('right', '30%', pad=0.0)
            s = np.sum(flux.to(u.mJy).value, axis=0)
            ax_hist.plot(np.concatenate([s, s]), np.concatenate([X2, -X2]))
            ax_hist.set_yticklabels([])

        if not no_labels:
            (ca, div, cax) = create_colorbar(
                im, ax, fig, position='right', padding=0.5)
            ca.set_label(r'$\log_{10}\mathrm{mJy / beam}$')

    circ = plt.Circle(
        (xlim[1] * beam_x, ylim[0] * beam_y),
        color='w',
        fill=True,
        radius=sigma_beam.to(u.arcsec).value,
        alpha=0.7)
    #circ.set_rasterized(True)

    if showbeam:
        ax.add_artist(circ)

    # reset limits
    if not no_labels:
        ax.set_xlabel('X ($\'\'$)')
        ax.set_ylabel('Y ($\'\'$)')
    ax.set_aspect('equal')

    if xticks is not None:
        ax.set_xticks(xticks)

    if no_labels:
        ax.set_position([0, 0, 1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')

    ax.set_aspect('equal')

    if no_labels:
        savefig(filename, fig, png=png, kwargs={
            'bbox_inches': 'tight',
            'pad_inches': 0}
                )
    else:
        savefig(filename, fig, png=png)
    plt.close();

def plot_sb_comparison(d25, d200):
    def internal(fig, a, sim, times, unit_values):
        from plutokore import radio
        from plutokore.plot import create_colorbar
        from numba import jit
        from astropy.convolution import convolve, Gaussian2DKernel
        from astropy.cosmology import Planck15 as cosmo

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

        redshift=0.1
        beamsize=5 * u.arcsec
        showbeam=True
        xlim=(-15, 15)
        ylim=(-30, 30)
        xticks=None
        pixel_size=1.8 * u.arcsec
        beam_x=0.8
        beam_y=0.8
        png=False
        contours=True
        should_convolve=True
        half_plane=False
        vmin=-3.0
        vmax=2.0
        style='flux-plot.mplstyle'
        no_labels=False
        with_hist=True
        trc_cutoff = 0.001

        output = np.where(times >= comp_time)[0][0]

        # calculate beam radius
        sigma_beam = (beamsize / 2.355)

        # calculate kpc per arcsec
        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(redshift).to(u.kpc / u.arcsec)

        # load timestep data file
        d = pk.simulations.load_timestep_data(output, sim)

        X1, X2 = pk.simulations.sphericaltocartesian(d)
        X1 = X1 * (unit_values.length / kpc_per_arcsec).to(u.arcsec).value
        X2 = X2 * (unit_values.length / kpc_per_arcsec).to(u.arcsec).value

        l = radio.get_luminosity(d, unit_values, redshift, beamsize)
        f = radio.get_flux_density(l, redshift).to(u.Jy).value
        #sb = radio.get_surface_brightness(f, d, unit_values, redshift, beamsize).to(u.Jy)

        xmax = (((xlim[1] * u.arcsec + pixel_size) * kpc_per_arcsec) / unit_values.length).si
        xstep = (pixel_size * kpc_per_arcsec / unit_values.length).si
        zmax = (((ylim[1] * u.arcsec + pixel_size) * kpc_per_arcsec) / unit_values.length).si
        zstep = (pixel_size * kpc_per_arcsec / unit_values.length).si
        ymax = max(xmax, zmax)
        ystep = min(xstep, zstep)
        ystep = 0.5

        if half_plane:
            x = np.arange(-xmax, xmax, xstep)
            z = np.arange(-zmax, zmax, zstep)
        else:
            x = np.arange(0, xmax, xstep)
            z = np.arange(0, zmax, zstep)
        y = np.arange(-ymax, ymax, ystep)
        raytraced_flux = np.zeros((x.shape[0], z.shape[0]))

        # print(f'xlim in arcsec is {xlim[1]}, xlim in code units is {xlim[1] * u.arcsec * kpc_per_arcsec / unit_values.length}')
        # print(f'zlim in arcsec is {ylim[1]}, zlim in code units is {ylim[1] * u.arcsec * kpc_per_arcsec / unit_values.length}')
        # print(f'xmax is {xmax}, ymax is {ymax}, zmax is {zmax}')
        # print(f'x shape is {x.shape}; y shape is {y.shape}; z shape is {z.shape}')

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
        sigma_beam_arcsec = beamsize / 2.355
        area_beam_kpc2 = (np.pi * (sigma_beam_arcsec * kpc_per_arcsec)
                          **2).to(u.kpc**2)
        beams_per_cell = (((pixel_size * kpc_per_arcsec) ** 2) / area_beam_kpc2).si
        #beams_per_cell = (area_beam_kpc2 / ((pixel_size * kpc_per_arcsec) ** 2)).si

        # radio_cell_areas = np.full(raytraced_flux.shape, xstep * zstep) * (unit_values.length ** 2)

        # n beams per cell
        #n_beams_per_cell = (radio_cell_areas / area_beam_kpc2).si

        raytraced_flux /= beams_per_cell

        stddev = sigma_beam_arcsec / beamsize
        beam_kernel = Gaussian2DKernel(stddev)
        if should_convolve:
            flux = convolve(raytraced_flux.to(u.Jy), beam_kernel, boundary='extend') * u.Jy
        else:
            flux = raytraced_flux.to(u.Jy)
        #flux = radio.convolve_surface_brightness(raytraced_flux, unit_values, redshift, beamsize)
        #flux = raytraced_flux

        X1 = x * (unit_values.length / kpc_per_arcsec).to(u.arcsec).value
        X2 = z * (unit_values.length / kpc_per_arcsec).to(u.arcsec).value

        # plot data
        a.set_xlim(xlim)
        a.set_ylim(ylim)

        contour_color = 'k'
        contour_linewidth = 0.33
        # contour_levels = [-3, -1, 1, 2]
        contour_levels = [-2, -1, 0, 1, 2] # Contours start at 10 μJy

        im = a.pcolormesh(
            X1,
            X2,
            np.log10(flux.to(u.mJy).value).T,
            shading='flat',
            edgecolors = 'face',
            rasterized = True,
            vmin=vmin,
            vmax=vmax)
        if contours:
            a.contour(X1, X2, np.log10(flux.to(u.mJy).value).T, contour_levels, linewidths=contour_linewidth, colors=contour_color)

        im = a.pcolormesh(
            -X1,
            X2,
            np.log10(flux.to(u.mJy).value).T,
            shading='flat',
            vmin=vmin,
            vmax=vmax)
        im.set_rasterized(True)
        if contours:
            a.contour(-X1, X2, np.log10(flux.to(u.mJy).value).T, contour_levels, linewidths=contour_linewidth, colors=contour_color)

        if not half_plane:
            im = a.pcolormesh(
                X1,
                -X2,
                np.log10(flux.to(u.mJy).value).T,
                shading='flat',
                vmin=vmin,
                vmax=vmax)
            im.set_rasterized(True)
            if contours:
                a.contour(X1, -X2, np.log10(flux.to(u.mJy).value).T, contour_levels, linewidths=contour_linewidth, colors=contour_color)

            im = a.pcolormesh(
                -X1,
                -X2,
                np.log10(flux.to(u.mJy).value).T,
                shading='flat',
                vmin=vmin,
                vmax=vmax)
            im.set_rasterized(True)
            if contours:
                a.contour(-X1, -X2, np.log10(flux.to(u.mJy).value).T, contour_levels, linewidths=contour_linewidth, colors=contour_color)

        (ca, div, cax) = create_colorbar(
            im, a, fig, position='right', padding=0.5)
        ca.set_label(r'$\log_{10}\mathrm{mJy / beam}$')

        circ = plot.Circle(
            (xlim[1] * beam_x, ylim[0] * beam_y),
            color='w',
            fill=True,
            radius=sigma_beam.to(u.arcsec).value,
            alpha=0.7)
        a.add_artist(circ)

    f, (a1, a2) = plot.subplots(1,2, figsize=(10, 10))

    c25 = pk.configuration.load_simulation_info(d25 + 'config.yaml') # Config for mach 25
    c200 = pk.configuration.load_simulation_info(d200 + 'config.yaml') # Config for mach 200

    times25 = get_times(d25) * c25[0].time
    times200 = get_times(d200) * c200[0].time

    comp_time = times25[-1] if times25[-1] < times200[-1] else times200[-1]

    internal(f, a1, d25, times25, c25[0])
    internal(f, a2, d200, times200, c200[0])

    a1.set_title('Mach 25')
    a2.set_title('Mach 200')
    a1.set_xlabel('X ($\'\'$)')
    a1.set_ylabel('Y ($\'\'$)')
    a2.set_xlabel('X ($\'\'$)')
    a2.set_ylabel('Y ($\'\'$)')

    a1.set_aspect('equal')
    a2.set_aspect('equal')

    f.suptitle('Surface brightness at {0:02.2f}'.format(comp_time))
    f.savefig('surface-brightness.png', bbox_inches='tight', dpi=400)
    plot.close(f)

def plot_pd_tracks(
        simulation_ids,
        labels,
        times,
        unit_values,
        run_dirs,
        filename,
        trc_cutoff=0.005,
        xlim=(0, 200),
        ylim=(10**25.5, 10**26.5),
        xticks=[2, 10, 100, 200],
        xtick_labels=['2', '10', '100', '200'],
        yticks=[np.float(10**25.5), np.float(10**26), np.float(10**26.5)],
        ytick_labels=[r'$10^{25.5}$', r'$10^{26}$', r'$10^{26.5}$'],
        legend_loc='upper left'):
    def get_lobe_length(simulation_data, tracer_threshold, ntracers):
        pk.radio.combine_tracers(simulation_data, ntracers)
        radio_tracer_mask, clamped_tracers, radio_combined_tracers = pk.radio.clamp_tracers(
            simulation_data, ntracers, tracer_threshold=tracer_threshold)

        theta_index = 0
        r_indicies = np.where(radio_tracer_mask[:, theta_index] == 1)[-1]
        if len(r_indicies) == 0:
            final_r_index = 0
        else:
            final_r_index = r_indicies[-1]
        return simulation_data.x1[final_r_index], final_r_index

    def calculate_length_and_luminosity(sim_dir, tsteps, unit_values):
        lengths = []
        lumin = []

        for i in tsteps:
            d = pk.simulations.load_timestep_data(i, sim_dir)
            ntrc = pk.simulations.get_tracer_count_data(d)
            lobe_length, ind = get_lobe_length(d, trc_cutoff, ntrc)
            luminosity = load_luminosity(sim_dir, i)
            lengths.append(lobe_length)
            lumin.append(luminosity.sum().to(u.W / u.Hz).value)
        return (lengths, lumin)

    data = []
    fig, ax = newfig(1)

    for sim_index, sim_id in enumerate(simulation_ids):
        lengths, luminosities = calculate_length_and_luminosity(
            run_dirs[sim_id], times[sim_index], unit_values[sim_index])
        #data.append({'lengths' : np.asarray(lengths), 'luminosities' : np.asarray(luminosities)})
        ax.loglog(
            np.asarray(lengths) * unit_values[sim_index].length.value,
            luminosities,
            label=labels[sim_index])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel('Lobe length (kpc)')
    ax.set_ylabel(r'1.4-GHz luminosity (W Hz$^{-1}$)')

    ax.set_xticks(xticks, minor=False)
    ax.set_xticklabels(xtick_labels)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    ax.legend(loc=legend_loc)

    savefig(filename, fig, kwargs={'bbox_inches' : 'tight'})
    plt.close()

def plot_pd_comparison(d25, d200):
    def internal(f, a):
        trc_cutoff = 0.001
        r = 0.1
        bw = 5 * u.arcsec

        c25 = pk.configuration.load_simulation_info(d25 + 'config.yaml') # Config for mach 25
        c200 = pk.configuration.load_simulation_info(d200 + 'config.yaml') # Config for mach 200

        t25 = get_times(d25)
        t200 = get_times(d200)

        o25 = np.arange(1, len(t25) - 1, 25)
        o200 = np.arange(1, len(t200) - 1, 25)

        calc25 = np.zeros((len(o25), 2))
        calc200 = np.zeros((len(o200), 2))

        for ind, i in enumerate(o25):
            d = pk.simulations.load_timestep_data(i, d25, mmap=True)
            m = d.x1[np.where(d.tr1[:,0] > trc_cutoff)[0][-1]]*c25[0].length.value
            l = pk.radio.get_luminosity(simulation_data = d,
                                        unit_values = c25[0],
                                        redshift = r,
                                        beam_FWHM_arcsec = bw)
            calc25[ind, :] = [m, l.sum().to(u.W / u.Hz).value]
        for ind, i in enumerate(o200):
            d = pk.simulations.load_timestep_data(i, d200, mmap=True)
            m = d.x1[np.where(d.tr1[:,0] > trc_cutoff)[0][-1]]*c200[0].length.value
            l = pk.radio.get_luminosity(simulation_data = d,
                                        unit_values = c200[0],
                                        redshift = r,
                                        beam_FWHM_arcsec = bw)
            calc200[ind, :] = [m, l.sum().to(u.W / u.Hz).value]

        print(calc25)
        print(calc200)
        a.loglog(
            calc25[:,0],
            calc25[:,1],
            label='M25')
        a.loglog(
            calc200[:,0],
            calc200[:,1],
            label='M200')

    f, a = plot.subplots(1,1, figsize=(10, 10))
    internal(f, a)

    xticks=[2, 10, 100, 200],
    xtick_labels=['2', '10', '100', '200'],
    yticks=[np.float(10**25.5), np.float(10**26), np.float(10**26.5)],
    ytick_labels=[r'$10^{25.5}$', r'$10^{26}$', r'$10^{26.5}$'],

    # a.set_xlabel('Lobe length (kpc)')
    # a.set_ylabel(r'1.4-GHz luminosity (W Hz$^{-1}$)')
    # a.set_xticks(xticks, minor=False)
    # a.set_xticklabels(xtick_labels)
    # a.set_yticks(yticks)
    # a.set_yticklabels(ytick_labels)
    a.legend()

    f.suptitle('Luminosity vs Lobe Length')
    f.savefig('pd.png', bbox_inches='tight', dpi=200)
    plot.close(f)

def main():
    # plot_mach200()
    # print('Plotted morphology')

    # plot_morphology_comparison('/u/pmyates/simulations/runs/kunanyi/honours/m14.5-M25/',
    #      '/u/pmyates/simulations/runs/kunanyi/honours/m14.5-M200/')
    # print('Plotted morphology comparison')

    # plot_lobe_lengths('/u/pmyates/simulations/runs/kunanyi/honours/m14.5-M25/',
    #      '/u/pmyates/simulations/runs/kunanyi/honours/m14.5-M200/')
    # print('Plotted lobe lengths')

    # plot_lobe_volumes('/u/pmyates/simulations/runs/kunanyi/honours/m14.5-M25/',
    #     '/u/pmyates/simulations/runs/kunanyi/honours/m14.5-M200/')
    # print('Plotted lobe volumes')

    plot_sb_comparison('/u/pmyates/simulations/runs/kunanyi/honours/m14.5-M25/',
                       '/u/pmyates/simulations/runs/kunanyi/honours/m14.5-M200/')
    print('Plotted surface brightness comparison')

    plot_pd_comparison('/u/pmyates/simulations/runs/kunanyi/honours/m14.5-M25/',
                       '/u/pmyates/simulations/runs/kunanyi/honours/m14.5-M200/')
    print('Plotted pd track comparison')

if __name__ == "__main__":
    main()
