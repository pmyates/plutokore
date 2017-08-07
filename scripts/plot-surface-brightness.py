#!/bin/env python3

import sys
import os
sys.path.append(os.path.abspath('/home/patrick/uni/plutokore/'))
import plutokore as pk
from plutokore import jet
from plutokore import io
from plutokore import radio
from plutokore.environments.makino import MakinoProfile
from astropy import units as u
from astropy import cosmology
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from numba import jit

@jit(nopython=True)
def raytrace_surface_brightness(r, θ, x, y, z, final, sb):
    φ = 0
    rmax = np.max(r)
    θmax = np.max(θ)

    for x_index in range(len(x)):
        for z_index in range(len(z)):
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

                # Now find index in r and θ arrays corresponding to this point
                # nb: Go higher or lower??
                r_index = np.argmax(r>ri)
                θ_index = np.argmax(θ>θi)
                final[x_index, z_index] += sb[r_index, θ_index]
    return

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    mass = (10**12.5) * u.M_sun
    redshift = 0
    makino_env_12p5 = MakinoProfile(
        mass,
        redshift,
        delta_vir=200,
        cosmo=cosmology.Planck15,
        concentration_method='klypin-planck-relaxed')

    theta_deg = 15
    M_x = 25
    Q = 1e37 * u.W
    jet_12p5 = jet.AstroJet(theta_deg, M_x, makino_env_12p5.sound_speed,
                        makino_env_12p5.central_density, Q,
                        makino_env_12p5.gamma)

    data = pk.simulations.load_timestep_data(452, '../tests/data/pluto/')

    redshift = 0.1
    beam_width = 5 * u.arcsec

    uv = jet.get_unit_values(makino_env_12p5, jet_12p5)
    l = radio.get_luminosity(data, uv, redshift, beam_width, )
    f = radio.get_flux_density(l, redshift)
    #sb = radio.get_surface_brightness(f, data, uv, redshift, beam_width).to(u.Jy).value
    #sb = radio.convolve_surface_brightness(sb, uv, redshift, beam_width).to(u.Jy)
    sb = f.to(u.Jy).value

    xmax=30
    xstep=1
    zmax=60
    zstep=1
    ymax = max(xmax, zmax)
    ystep = min(xstep, zstep)
    
    x = np.arange(0, xmax, xstep)
    z = np.arange(0, zmax, zstep)
    y = np.arange(-ymax, ymax, ystep)
    final = np.zeros((x.shape[0], z.shape[0]))

    raytrace_surface_brightness(
        r=data.x1,
        θ=data.x2,
        x=x,
        y=y,
        z=z,
        final=final,
        sb=sb
    )

    final = final * u.Jy

    kpc_per_arcsec = cosmology.Planck15.kpc_proper_per_arcmin(redshift).to(u.kpc /
                                                               u.arcsec)
    # beam information
    sigma_beam_arcsec = beam_width / 2.355
    area_beam_kpc2 = (np.pi * (sigma_beam_arcsec * kpc_per_arcsec)
                      **2).to(u.kpc**2)

    radio_cell_areas = np.full(final.shape, xstep * zstep)

    # in physical units
    radio_cell_areas_physical = radio_cell_areas * uv.length**2

    # n beams per cell
    n_beams_per_cell = (radio_cell_areas_physical / area_beam_kpc2).si

    final = final / n_beams_per_cell
    final_convolved = radio.convolve_surface_brightness(final, uv, redshift, beam_width)

    # rr, θθ = np.meshgrid(r, θ)

    # x = r * np.sin(θθ) * np.cos(φ)
    # y = r * np.sin(θθ) * np.sin(φ)
    # z = r * np.cos(θθ)

    #im = ax.pcolormesh(x, z, np.log10(sb.to(u.mJy).T.value), vmin=-3, vmax=0, cmap='viridis')
    im = ax.pcolormesh(x, z, np.log10(final_convolved.to(u.mJy).T.value), vmin=-3, vmax=3, cmap='viridis')
    fig.colorbar(im)

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 60)
    ax.set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    main()
