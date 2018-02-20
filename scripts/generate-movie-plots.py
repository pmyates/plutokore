#!/usr/bin/env python3

import os
import sys
import argparse
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
from astropy import units as u
import pathlib

def plot_movie(simulation_dir,
               output_dir,
               outputs,
               unit_values,
               environment,
               jet):
    x = None
    y = None
    pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
    for o in outputs:
        data = pk.simulations.load_timestep_data(o, simulation_dir)
        if x is None:
            x, y = pk.simulations.sphericaltocartesian(data, rotation = 0) * unit_values.length
        f, a = plot.subplots(figsize=(10, 10))
        a.set_aspect('equal')
        a.set_xlim(-1100, 1100)
        a.set_ylim(-500, 500)
        im1 = a.pcolormesh(x, -y, np.log10(data.rho * unit_values.density.value).T, rasterized = True, edgecolors = 'face', shading = 'flat', vmin = -30, vmax = -26)
        im2 = a.pcolormesh(x, y, np.log10(data.rho * unit_values.density.value).T, rasterized = True, edgecolors = 'face', shading = 'flat', vmin = -30, vmax = -26)
        a.set_position([0, 0, 1, 1])
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_xticks([])
        a.set_yticks([])
        a.set_axis_off()

        im1.set_cmap('viridis')
        im2.set_cmap('viridis')
        f.savefig(f'{output_dir}/rho-viridis-{o:03d}.png', dpi=200, bbox_inches = 'tight', pad_inches = 0)

        im1.set_cmap('magma')
        im2.set_cmap('magma')
        f.savefig(f'{output_dir}/rho-magma-{o:03d}.png', dpi=200, bbox_inches = 'tight', pad_inches = 0)

        plot.close(f)
        print(f'Plotted {o} for {simulation_dir}')

def get_times(sim_dir):
    with open(os.path.join(sim_dir, 'dbl.out'), "r") as f_var:
        tlist = []
        for line in f_var.readlines():
            tlist.append(float(line.split()[1]))
    return np.asarray(tlist)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--simulation', help='Simulation number (0, 1, 2)', type=int)
    #parser.add_argument(
    args = parser.parse_args()
    s = args.simulation

    simulation_directories = ['/u/pmyates/katie-sims/M25_15deg_OFF4R_Q37',
                              '/u/pmyates/katie-sims/M25_15deg_OFF4R_Q38',
                              '/u/pmyates/katie-sims/M25_30deg_OFF4R_Q38',
                              '/u/pmyates/katie-sims/M25_30deg_OFF1R_Q38']

    output_directories = ['/u/pmyates/katie-sims/movies/M25_15deg_OFF4R_Q37',
                          '/u/pmyates/katie-sims/movies/M25_15deg_OFF4R_Q38',
                          '/u/pmyates/katie-sims/movies/M25_30deg_OFF4R_Q38',
                          '/u/pmyates/katie-sims/movies/M25_30deg_OFF1R_Q38']

    halo_mass = (10 ** 14.5) * u.M_sun
    redshift = 0
    env = pk.KingProfile(halo_mass, redshift)

    theta1 = 15
    theta2 = 30
    M_x = 25
    Q1 = (10 ** 37) * u.W
    Q2 = (10 ** 38) * u.W

    jet_37 = pk.AstroJet(theta1,
                         M_x,
                         env.sound_speed,
                         env.central_density,
                         Q1,
                         env.gamma)
    jet_38_15 = pk.AstroJet(theta1,
                         M_x,
                         env.sound_speed,
                         env.central_density,
                         Q2,
                         env.gamma)
    jet_38_30 = pk.AstroJet(theta2,
                         M_x,
                         env.sound_speed,
                         env.central_density,
                         Q2,
                         env.gamma)

    uv_37 = pk.jet.get_unit_values(env, jet_37)
    uv_38_15 = pk.jet.get_unit_values(env, jet_38_15)
    uv_38_30 = pk.jet.get_unit_values(env, jet_38_30)

    step = 1

    outputs = [np.arange(0, pk.simulations.get_output_count(simulation_directories[0]), step),
               np.arange(0, pk.simulations.get_output_count(simulation_directories[1]), step),
               np.arange(0, pk.simulations.get_output_count(simulation_directories[2]), step),
               np.arange(0, pk.simulations.get_output_count(simulation_directories[3]), step)]

    if (s is None or s == 0):
        plot_movie(simulation_directories[0],
                  output_directories[0],
                  outputs[0],
                  uv_37,
                  env,
                  jet_37)
    if (s is None or s == 1):
        plot_movie(simulation_directories[1],
                   output_directories[1],
                   outputs[1],
                   uv_38_15,
                   env,
                   jet_38_15)
    if (s is None or s == 2):
        plot_movie(simulation_directories[2],
                   output_directories[2],
                   outputs[2],
                   uv_38_30,
                   env,
                   jet_38_30)
    if (s is None or s == 3):
        plot_movie(simulation_directories[3],
                   output_directories[3],
                   outputs[3],
                   uv_38_30,
                   env,
                   jet_38_30)

if __name__ == '__main__':
    main()
