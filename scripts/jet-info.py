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
from plutokore.utilities import tcolors

def print_jet_info(cfile):
    print(f'{tcolors.HEADER}Information for {tcolors.BOLD}{cfile}{tcolors.ENDC}')

    print(f'\n{tcolors.BLUE+tcolors.BOLD}Environment:{tcolors.ENDC}')
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

    print(f'\n{tcolors.BLUE+tcolors.BOLD}Jet:{tcolors.ENDC}')
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

    print(f'\n{tcolors.BLUE+tcolors.BOLD}Unit Values:{tcolors.ENDC}')
    print(tabulate([
        ['density', uv.density],
        ['length', uv.length],
        ['speed', uv.speed],
        ['time', uv.time],
        ['mass', uv.mass],
        ['pressure', uv.pressure],
        ['energy', uv.energy],
    ]))

    print(f'\n{tcolors.BLUE+tcolors.BOLD}definitions.h unit values:{tcolors.ENDC}')
    print(tabulate([
        ['UNIT_DENSITY', uv.density.to(u.g / u.cm **3).value],
        ['UNIT_LENGTH', f'{uv.length.to(u.kpc).value}e3*CONST_pc'],
        ['UNIT_VELOCITY', f'{uv.speed.to(u.cm / u.s).value/1e7}e7'],
    ]))

def print_sim_info(cfile):
    print(f'{tcolors.HEADER}Information for {tcolors.BOLD}{cfile}{tcolors.ENDC}')
    yml = config.yaml

    print(f'\n{tcolors.BLUE+tcolors.BOLD}Time information:{tcolors.ENDC}')
    print(tabulate([
        ['Simulation time', (yml['simulation-properties']['total-time-myrs'] * u.Myr) / uv.time],
        ['Jet active time', (yml['simulation-properties']['jet-active-time-myrs'] * u.Myr) / uv.time],
    ]))

    if yml['yaml-version'] >= 3:
        print(f'\n{tcolors.BLUE+tcolors.BOLD}Grid information:{tcolors.ENDC}')
        print(tabulate([
            ['X1 grid', (yml['simulation-properties']['x1'] * u.kpc) / uv.length],
            ['X2 grid', (yml['simulation-properties']['x2'] * u.kpc) / uv.length],
            ['X3 grid', (yml['simulation-properties']['x3'] * u.kpc) / uv.length],
        ]))


    print(f'\n{tcolors.BLUE+tcolors.BOLD}definitions.h unit values:{tcolors.ENDC}')
    print(tabulate([
        ['UNIT_DENSITY', uv.density.to(u.g / u.cm **3).value],
        ['UNIT_LENGTH', f'{uv.length.to(u.kpc).value}e3*CONST_pc'],
        ['UNIT_VELOCITY', f'{uv.speed.to(u.cm / u.s).value/1e7}e7'],
    ]))


def nondimensionalise(value):
    v = u.Quantity(value)
    converted = False
    for k, uu in uv._asdict().items():
        if (uu.unit.is_equivalent(v)):
            print(f'Converted input {v} to code {uu.unit.physical_type}:')
            print((v / uu).si)
            converted = True
    if not converted:
        print(f'Could not find an equivalence for {v}')

def dimensionalise(value, type):
    if type in uv._fields:
        print(f'Converting input {value} to physical {type}:')
        print(getattr(uv, type) * value)
    else:
        print(f'Could not convet {value} to physical {type}')


def load_jet(cfile):
    global config
    global env
    global jet
    global uv

    config = pk.configuration.SimulationConfiguration(cfile, None, None)
    uv = config.get_unit_values()
    env = config.env
    jet = config.jet

class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def main():
    pp = argparse.ArgumentParser(add_help=False)
    pp.add_argument('-c', '--config', help='Config file, or directory containing a config.yaml file', default='./')

    parser = DefaultHelpParser(
        description='A program for calculating jet and environment parameters',
    )
    subparsers = parser.add_subparsers(dest='subcommand', metavar='command')
    subparsers.required = True
    parser_jinfo = subparsers.add_parser('jetinfo', parents=[pp], help='Print jet and environment parameters for the given config file')

    parser_dim = subparsers.add_parser('dim', parents=[pp], help='Convert a value in code units to a value in physical units')
    parser_dim.add_argument('value', help='Value to dimensionalise', type=float)
    parser_dim.add_argument('-u', '--unit', help='Code unit the value is in, can be one of [%(choices)s]', choices=['length', 'speed', 'energy', 'density', 'pressure', 'mass', 'time'], type=str, required=True, metavar='UNIT')

    parser_nondim = subparsers.add_parser('nondim', parents=[pp], help='Convert a physical value to a value in code units')
    parser_nondim.add_argument('value', help='Value to nondimensionalise')

    parser_siminfo = subparsers.add_parser('siminfo', parents=[pp], help='Print configuration values in simulation units')

    args = parser.parse_args()

    if os.path.isdir(args.config):
        cfile = os.path.join(args.config, 'config.yaml')
    else:
        cfile = args.config
    if not os.path.exists(cfile):
        print(f'Can not find config file at {cfile}')
        return 1
    
    load_jet(cfile)

    if args.subcommand == 'jetinfo':
        print_jet_info(cfile)
    if args.subcommand == 'nondim':
        nondimensionalise(args.value)
    if args.subcommand == 'dim':
        dimensionalise(args.value, args.unit)
    if args.subcommand == 'siminfo':
        print_sim_info(cfile)

if __name__ == '__main__':
    main()
