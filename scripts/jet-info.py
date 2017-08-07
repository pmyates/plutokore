#!/bin/env python3

import sys
import os
try:
    import plutokore as pk
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import plutokore as pk
import argparse
import pprint
from astropy import units as u
from tabulate import tabulate
import numpy as np

def print_jet_info(cfile):
    config = pk.configuration.SimulationConfiguration(cfile, None, None)
    uv = config.get_unit_values()
    env = config.env
    jet = config.jet
    print(f'Information for {cfile}')

    print('\nEnvironment:')
    print(tabulate([
        ['halo mass', f'1e{np.log10(env.halo_mass.value)}'],
        ['redshift', env.redshift],
        ['virial temp', env.T],
        ['virial radius', env.virial_radius],
        ['cosmology', env.cosmo.name],
        ['concentration method', env.concentration_method],
        ['concentration', env.concentration],
        ['scale radius', env.scale_radius],
        ['nfw parameter', env.nfw_parameter],
        ['sound speed', env.sound_speed],
        ['central density', env.central_density],

    ]))

    print('\nJet:')
    print(tabulate([
        ['half opening angle', np.rad2deg(jet.theta)],
        ['external mach number', jet.M_x],
        ['power', jet.Q],
        ['omega', jet.omega],
        ['jet velocity', jet.v_jet],
        ['L1', jet.L_1],
        ['L1a', jet.L_1a],
        ['L1b', jet.L_1b],
        ['L1c', jet.L_1c],
        ['L2', jet.L_2]
    ]))

    print('\nUnit Values:')
    print(tabulate([
        ['density', uv.density],
        ['length', uv.length],
        ['speed', uv.speed],
        ['time', uv.time],
        ['mass', uv.mass],
        ['pressure', uv.pressure],
        ['energy', uv.energy],
    ]))

    print('\ndefinitions.h unit values:')
    print(tabulate([
        ['UNIT_DENSITY', uv.density.to(u.g / u.cm **3).value],
        ['UNIT_LENGTH', f'{uv.length.to(u.kpc).value}e3*CONST_pc'],
        ['UNIT_VELOCITY', f'{uv.speed.to(u.cm / u.s).value/1e7}e7'],
    ]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file, or directory containing a config.yaml file')
    args = parser.parse_args()

    if os.path.isdir(args.config):
        cfile = os.path.join(args.config, 'config.yaml')
    else:
        cfile = args.config
    if not os.path.exists(cfile):
        print(f'Can not find config file at {cfile}')
        return 1

    print_jet_info(cfile)

if __name__ == '__main__':
    main()
