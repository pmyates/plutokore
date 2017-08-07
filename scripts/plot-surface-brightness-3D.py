#!/bin/env python3

import sys
import os
try:
    import plutokore as pk
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
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
from mpl_toolkits.mplot3d import Axes3D

def main():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    mass = (10**12.5) * u.M_sun
    z = 0
    makino_env_12p5 = MakinoProfile(
        mass,
        z,
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

    z = 0.1
    beam_width = 5 * u.arcsec

    uv = jet.get_unit_values(makino_env_12p5, jet_12p5)
    l = radio.get_luminosity(data, uv, z, beam_width, )
    f = radio.get_flux_density(l, z)
    fc = radio.get_convolved_flux_density(f, z, beam_width)
    sb = radio.get_surface_brightness(f, data, uv, z, beam_width)

    cmap = plt.cm.plasma
    norm = mpl.colors.Normalize(vmin=-3, vmax=0)
    colors = cmap(norm(np.log10(sb.to(u.mJy).T.value)))

    k = 0

    for i in range(0, 360, 90):
        #r = np.arange(1, 11)
        #θ = np.arange(0, 100, 10)
        r = data.x1[:600]
        θ = data.x2
        φ = np.deg2rad(np.full_like(r, i))
        rr, θθ, φφ = np.meshgrid(r, θ, φ)

        xx = rr * np.sin(θθ) * np.cos(φφ)
        yy = rr * np.sin(θθ) * np.sin(φφ)
        zz = rr * np.cos(θθ)
        #ax.plot(xx[:,:,1], yy[:,:,1], zs=φφ[:,1,1], zdir='z')
        #ax.plot(xx[:,:,1], yy[:,:,1], zs=zz[:,1,1], color=f'C{k}')
        #ax.plot(xx[:,:,1].T, yy[:,:,1].T, zs=zz[1,:,1], color=f'C{k}')
        ax.plot_surface(xx[:,:,1], yy[:,:,1], zz[:,:,1], linewidth=0, antialiased=True, alpha=1, facecolors=colors)
        ax.set_aspect('equal')
        k += 1
    sc = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sc.set_array([])
    plt.colorbar(sc)
    plt.show()

if __name__ == '__main__':
    main()
