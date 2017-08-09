#!/bin/env python3

import sys
import os
try:
    import plutokore as pk
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import plutokore as pk
import argparse
from plutokore.utilities import tcolors

def print_jet_info(cfile):
    from tabulate import tabulate
    import numpy as np
    from astropy import units as u
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
    from astropy import units as u
    from tabulate import tabulate
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

    print(f'\n{tcolors.BLUE+tcolors.BOLD}Parameter information:{tcolors.ENDC}')
    print(tabulate([
        ['RHO_0', (env.central_density / uv.density)],
        ['R_SCALING', (env.scale_radius / uv.length)],
    ]))


    print(f'\n{tcolors.BLUE+tcolors.BOLD}definitions.h unit values:{tcolors.ENDC}')
    print(tabulate([
        ['UNIT_DENSITY', uv.density.to(u.g / u.cm **3).value],
        ['UNIT_LENGTH', f'{uv.length.to(u.kpc).value}e3*CONST_pc'],
        ['UNIT_VELOCITY', f'{uv.speed.to(u.cm / u.s).value/1e7}e7'],
    ]))


def nondimensionalise(value):
    from astropy import units as u
    v = u.Quantity(value)
    converted = False
    for k, uu in uv._asdict().items():
        if (uu.unit.is_equivalent(v)):
            print(f'Converted input {v} to code {uu.unit.physical_type}:')
            print((v / uu).si)
            converted = True
    if not converted:
        print(f'Could not find an equivalence for {v}')

def dimensionalise(value, unit):
    if unit in uv._fields:
        print(f'Converting input {value} to physical {unit}:')
        print(getattr(uv, unit) * value)
    else:
        print(f'Could not convert {value} to physical {unit}')

def check_config(cfile, cdir):
    import pprint
    print(f'{tcolors.HEADER}Checking simulation for {tcolors.BOLD}{cfile}{tcolors.ENDC}')
    paths = {
        'yaml': cfile,
        'ini': os.path.join(cdir, 'pluto.ini'),
        'definitions': os.path.join(cdir, 'definitions.h')
    }
    errors = pk.configuration.validate_yaml(paths['yaml'], paths['ini'], paths['definitions'])

    if len(errors) > 0:
        pprint.pprint(errors, width=100)
        print(f'{tcolors.LIGHT_RED}{tcolors.BOLD}Not valid{tcolors.ENDC}')
        sys.exit(1)
    else:
        print(f'{tcolors.BLUE}{tcolors.BOLD}Passed{tcolors.ENDC}')

def submit(sdir, sname):
    import subprocess
    res = subprocess.run(['qsub', os.path.join(sdir, sname)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0:
        print(res.stdout.decode().trim())
        print(f'{tcolors.BLUE}{tcolors.BOLD}Job submitted{tcolors.ENDC}')
        with open(os.path.join(sdir, 'current-run'), 'wb') as f:
            f.write(res.stdout)
    else:
        print(res.stderr.decode())
        print(f'{tcolors.LIGHT_RED}{tcolors.BOLD}Submission failed{tcolors.ENDC}')
        sys.exit(1)

def plot_yml(yml_plots, timestep, directory, output_directory):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plot
    import numpy as np
    import yaml
    def plot_slice(x1, x2, v, fig, ax, xlabel=None, ylabel=None, title=None):
        im = ax.pcolormesh(x1, x2, v.T)
        pk.plot.create_colorbar(im, ax, fig)
        ax.set_aspect('equal')
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)

    output_directory = os.path.join(output_directory, '')
    print(f'{tcolors.BOLD}{tcolors.HEADER}Loading simulation data for output {timestep}...{tcolors.ENDC}')
    d = pk.simulations.load_timestep_data(timestep, directory, mmap=mmap)
    x_index = d.n1_tot // 2
    y_index = d.n2_tot // 2
    z_index = d.n3_tot // 2

    for y in yml_plots:
        with open(y) as f:
            yml = yaml.load(f)
            fy = yml['figure']
        name = fy.get('name', os.path.splitext(os.path.basename(y))[0])
        rcount = fy['dimensions'][0]
        ccount = fy['dimensions'][1]
        xlabel = fy.get('xlabel', 'X')
        ylabel = fy.get('ylabel', 'Y')
        xlim = fy.get('xlim', None)
        ylim = fy.get('ylim', None)
        title = fy.get('title', '')
        plts = fy['plots']
        skw = {'aspect': 'equal'}
        if xlim is not None:
            skw['xlim'] = xlim
        if ylim is not None:
            skw['ylim'] = ylim
        df, axs = plot.subplots(rcount, ccount, subplot_kw=skw, figsize=fy['size'])
        df.suptitle(f'{title}: output {timestep}')
        df.subplots_adjust(wspace=0.3, hspace=0.3)
        for i,p in enumerate(plts):
            a = axs[i%rcount][i//rcount]
            if p['var'] in ['rho', 'prs', 'vx1', 'vx2', 'vx3'] or p['var'][:2] == 'tr':
                v = getattr(d, p['var'])
            if p['plane'] == 'xy':
                x = d.x1
                y = d.x2
                v = np.squeeze(v[:, :, z_index])
            elif p['plane'] == 'xz':
                x = d.x1
                y = d.x3
                v = np.squeeze(v[:, y_index, :])
            elif p['plane'] == 'yz':
                x = d.x2
                y = d.x3
                v = np.squeeze(v[x_index, :, :])
            if p.get('log', False):
                v = np.log10(v)
            ptitle = p.get('title', f'{p["var"]} ({p["plane"]} plane)')
            if 'xlim' in p:
                a.set_xlim(p['xlim'])
            if 'ylim' in p:
                a.set_ylim(p['ylim'])
            plot_slice(x, y, v, df, a, xlabel=xlabel, ylabel=ylabel, title=ptitle)
        print(f'{tcolors.BOLD}{tcolors.BLUE}Saving {name} for {timestep}{tcolors.ENDC}')
        df.savefig(f'{output_directory}{name}-{timestep:04}', dpi=1000, bbox_inches='tight')
        plot.close(df)

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
    pp.add_argument('-c', '--config', help='Config file, or directory containing a config.yaml file (default: current directory)', default='./')

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

    parser_check = subparsers.add_parser('check', parents=[pp], help='Check that simulation configuration is valid')

    parser_submit = subparsers.add_parser('submit', help='Submit a simulation')
    parser_submit.add_argument('-d', '--directory', help='Simulation directory (default: current directory)', default='./')
    parser_submit.add_argument('-n', '--name', help='Submission script name (default: %(default)s)', default='run.sh')
    parser_submit.add_argument('-c', '--check', help='Check simulation before submitting (default: %(default)s)', default='yes', choices=['yes', 'no'])

    parser_plot = subparsers.add_parser('plot', help='Plot a simulation')
    parser_plot.add_argument('-y', '--yml', nargs='+', help='Plot definition files (.yml) (required)', required=True)
    parser_plot.add_argument('-w', '--workingdir', help='Simulation directory (default: current directory)', action='store', type=str, default='./')
    parser_plot.add_argument('-o', '--output', help='Plot output directory (default: simulation directory)', action='store', type=str)
    parser_plot.add_argument('-t', '--timestep', help='Produce diagnostic plots for the specified timestep', action='store', type=int)
    parser_plot.add_argument('-s', '--start', help='Start plotting from this output (and plot all later ones)', action='store', type=int)
    parser_plot.add_argument('--mmap', help='Enable memory mapping for output files (default: %(default)s)', action='store', type=str, choices=['yes', 'no'], default='yes')


    args = parser.parse_args()

    if hasattr(args, 'config'):
        if os.path.isdir(args.config):
            cfile = os.path.join(args.config, 'config.yaml')
            cdir = args.config
        else:
            cfile = args.config
            cdir = os.path.dirname(cfile)
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
    if args.subcommand == 'check':
        check_config(cfile, cdir)
    if args.subcommand == 'submit':
        cfile = os.path.join(args.directory, 'config.yaml')
        cdir = args.directory
        load_jet(cfile)
        if args.check == 'yes':
            check_config(cfile, cdir)
        submit(cdir, args.name)
    if args.subcommand == 'plot':
        if args.output is None:
            args.output = args.workingdir

        print(f'Simulation directory is {args.workingdir}')
        print(f'Plot directory is {args.output}')

        global mmap
        if args.mmap == 'yes':
            mmap = True
        else:
            mmap = False

        if args.timestep is not None:
            plot_yml(args.yml, args.timestep, args.workingdir, args.output)
        else:
            start = 0
            if args.start is not None:
                start = args.start
            for i in range(start, pk.io.nlast_info(w_dir=args.workingdir)['nlast']+1):
                plot_yml(args.yml, i, args.workingdir, args.output)


if __name__ == '__main__':
    main()
