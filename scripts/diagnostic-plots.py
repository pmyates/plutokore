#!/usr/bin/env python3
import sys
import os
import matplotlib as mpl
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

mmap=False

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

def plot_yml(yml_plots, timestep, directory, output_directory):
    output_directory = os.path.join(output_directory, '')
    print(f'Loading simulation data for output {timestep}...')
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
        print(f'Saving {name} for {timestep}')
        df.savefig(f'{output_directory}{name}-{timestep:04}', dpi=1000, bbox_inches='tight')
        plot.close(df)

def produce_diagnostic_plots(timestep, directory, output_directory):
    output_directory = os.path.join(output_directory, '')
    print(f'Loading simulation data for output {timestep}...')
    data = pk.simulations.load_timestep_data(timestep, directory, mmap=mmap)

    x_index = data.n1_tot // 2
    y_index = data.n2_tot // 2
    z_index = data.n3_tot // 2

    print(f'Plotting diagnostic plots for output {timestep}...')
    df, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plot.subplots(2, 3, figsize=(15, 10))
    df.suptitle(f'Output {timestep}, midplane slices')

    print(data.rho.shape)
    print(data.x1.shape)
    print(data.x2.shape)
    print(data.x3.shape)
    print(x_index)
    print(y_index)
    print(z_index)
    plot_slice(data.x1, data.x2, np.log10(np.squeeze(data.rho[:, :, z_index])), df, ax1, xlabel='X', ylabel='Y', title=f'Density (x-y plane)')
    plot_slice(data.x1, data.x2, np.log10(np.squeeze(data.prs[:, :, z_index])), df, ax2, xlabel='X', ylabel='Y', title=f'Pressure (x-y plane)')

    plot_slice(data.x1, data.x3, np.log10(np.squeeze(data.rho[:, y_index, :])), df, ax3, xlabel='X', ylabel='Z', title=f'Density (x-z plane)')
    plot_slice(data.x1, data.x3, np.log10(np.squeeze(data.prs[:, y_index, :])), df, ax4, xlabel='X', ylabel='Z', title=f'Pressure (x-z plane)')

    plot_slice(data.x2, data.x3, np.log10(np.squeeze(data.rho[x_index, :, :])), df, ax5, xlabel='Y', ylabel='Z', title=f'Density (y-z plane)')
    plot_slice(data.x2, data.x3, np.log10(np.squeeze(data.prs[x_index, :, :])), df, ax6, xlabel='Y', ylabel='Z', title=f'Pressure (y-z plane)')

    df.savefig(f'{output_directory}diag-{timestep:04}', dpi=1000, bbox_inches='tight')

    df, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plot.subplots(2, 3, subplot_kw={'xlim': (-2, 2), 'ylim': (-2, 2)}, figsize=(15, 10))
    df.suptitle(f'Output {timestep}, midplane slices (jet-injection region)')

    plot_slice(data.x1, data.x2, np.log10(np.squeeze(data.rho[:, :, z_index])), df, ax1, xlabel='X', ylabel='Y', title=f'Density (x-y plane)')
    plot_slice(data.x1, data.x2, np.squeeze(data.vx3[:, :, z_index]), df, ax2, xlabel='X', ylabel='Y', title=f'Velocity in z (x-y plane)')

    plot_slice(data.x1, data.x3, np.log10(np.squeeze(data.rho[:, y_index, :])), df, ax3, xlabel='X', ylabel='Z', title=f'Density (x-z plane)')
    plot_slice(data.x1, data.x3, np.squeeze(data.vx3[:, y_index, :]), df, ax4, xlabel='X', ylabel='Z', title=f'Velocity in z (x-z plane)')

    plot_slice(data.x2, data.x3, np.log10(np.squeeze(data.rho[x_index, :, :])), df, ax5, xlabel='Y', ylabel='Z', title=f'Density (y-z plane)')
    plot_slice(data.x2, data.x3, np.squeeze(data.vx3[x_index, :, :]), df, ax6, xlabel='Y', ylabel='Z', title=f'Velocity in z (y-z plane)')

    df.savefig(f'{output_directory}diag-injection-{timestep:04}', dpi=1000, bbox_inches='tight')

    # plot_slice(data.x1, data.x2, np.log10(data.rho[:,:,z_index]), df, ax1, xlabel='X', ylabel='Y', title=f'Density')
    # plot_slice(data.x1, data.x2, np.log10(data.prs[:,:,z_index]), df, ax2, xlabel='X', ylabel='Y', title=f'Pressure')
    # plot_slice(data.x1, data.x2, data.vx1[:,:,z_index], df, ax3, xlabel='X', ylabel='Y', title=f'X velocity')
    # plot_slice(data.x1, data.x2, data.vx2[:,:,z_index], df, ax4, xlabel='X', ylabel='Y', title=f'Y velocity')

    return

def main():
    parser = argparse.ArgumentParser(
        description = 'Produce diagnostic plots for 3D simulations'
    )
    parser.add_argument('-t', help='Produce diagnostic plots for the specified timestep', action='store', type=int)
    parser.add_argument('-w', help='Simulation directory (defaults to current working directory)', action='store', type=str, default='./')
    parser.add_argument('-o', help='Plot output directory (defaults to simulation directory)', action='store', type=str)
    parser.add_argument('-s', help='Start plotting from this output (and plot all later ones)', action='store', type=int)
    parser.add_argument('--mmap', help='Enable memory mapping for output files (default: %(default)s)', action='store', type=str, choices=['yes', 'no'], default='yes')
    parser.add_argument('-y', '--yml', nargs='+', help='Plot definition files (.yml)')
    args = parser.parse_args()

    if args.o is None:
        args.o = args.w

    print(f'Simulation directory is {args.w}')
    print(f'Plot directory is {args.o}')

    global mmap
    if args.mmap == 'yes':
        mmap = True
    else:
        mmap = False

    if args.t is not None:
        if args.yml is None:
            produce_diagnostic_plots(args.t, args.w, args.o)
        else:
            plot_yml(args.yml, args.t, args.w, args.o)
    else:
        start = 0
        if args.s is not None:
            start = args.s
        for i in range(start, pk.io.nlast_info(w_dir=args.w)['nlast']+1):
            if args.yml is None:
                produce_diagnostic_plots(i, args.w, args.o)
            else:
                plot_yml(args.yml, i, args.w, args.o)

if __name__ == "__main__":
    main()
