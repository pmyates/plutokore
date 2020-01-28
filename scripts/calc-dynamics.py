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
import code

def calculate_dynamics(*, sim_dir, tracer_cutoff, two_jets = False):

    # load the simulation information
    uv, env, jet = pk.configuration.load_simulation_info(os.path.join(sim_dir, 'config.yaml'))

    # load the simulation times
    times = pk.simulations.get_times(sim_dir) * uv.time.to(u.Myr)
    times = times[0:5]
    total = len(times)

    # setup output arrays
    lengths = np.zeros(times.shape) * u.kpc
    volumes = np.zeros(times.shape) * (u.kpc ** 3)
    if two_jets == True:
        lengths = np.zeros((times.shape[0], 2)) * u.kpc
        volumes = np.zeros((times.shape[0], 2)) * (u.kpc ** 3)
    print(lengths.shape)

    # loop through all outuputs and calculate dynamics
    for output_number, time in enumerate(times):
        print(f'Processing {output_number}/{total}')

        # load simulation
        sim_data = pk.simulations.load_timestep_data(output_number, sim_dir, mmap=True)

        # load tracers
        tr = pk.simulations.combine_tracers(sim_data, pk.simulations.get_tracer_count_data(sim_data))

        # calculate length
        if two_jets == False:
            ind = np.argmin(tr[:, 0] > tracer_cutoff)
            lengths[output_number] = sim_data.x1[ind] * uv.length
        elif two_jets == True:
            # first jet
            ind = np.argmin(tr[:, 0] > tracer_cutoff)
            lengths[output_number,0] = sim_data.x1[ind] * uv.length
            # second jet
            ind = np.argmin(tr[:, -1] > tracer_cutoff)
            lengths[output_number,1] = sim_data.x1[ind] * uv.length

        # calculate volume
        if two_jets == False:
            volumes[output_number] = pk.simulations.calculate_cell_volume(sim_data)[tr > tracer_cutoff].sum() * (uv.length ** 3)
        elif two_jets == True:
            t_ind = tr.shape[1]//2
            # first jet
            vol_index = tr > tracer_cutoff
            vol_index[:,:t_ind] = False
            volumes[output_number, 0] = pk.simulations.calculate_cell_volume(sim_data)[vol_index].sum() * (uv.length ** 3)
            # second jet
            vol_index = tr > tracer_cutoff
            vol_index[:,t_ind:] = False
            volumes[output_number, 1] = pk.simulations.calculate_cell_volume(sim_data)[vol_index].sum() * (uv.length ** 3)

        #code.interact(local=locals())

    return (times, lengths, volumes)

def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('simulation_directory', help='Simulation directory', type=str)
    parser.add_argument('output_directory', help='Output directory', type=str)
    parser.add_argument('--trc_cutoff', help='Tracer cutoff', type=float, default=1e-14)
    parser.add_argument('--two_jets', help='Two jet', action='store_true')
    args = parser.parse_args()

    # create output directory if needed
    pathlib.Path(args.output_directory).mkdir(parents = True, exist_ok = True)

    mach, opening_angle, offset, power = os.path.basename(os.path.normpath(args.simulation_directory)).split('_')
    fname = f'{power}-{opening_angle}-{offset}-dynamics'
    print(fname)

    (t, l, v) = calculate_dynamics(sim_dir = args.simulation_directory, tracer_cutoff=args.trc_cutoff, two_jets=args.two_jets)
    np.savez(f'{args.output_directory}/{fname}.npz', times=t.to(u.Myr), lengths=l.to(u.kpc), volumes=v.to(u.kpc ** 3))

    # with h5py.File(f'{args.output_directory}/dynamics.hdf5', 'w') as f:
    #     f.create_dataset('time', data=t.to(u.Myr))
    #     f.create_dataset('length', data=l.to(u.kpc))
    #     f.create_dataset('volume', data=v.to(u.kpc ** 3))
    #     f.attrs['tracer_cutoff'] = args.trc_cutoff
    #     f.attrs['sim_dir'] = args.simulation_directory

if __name__ == '__main__':
    main()
