from __future__ import print_function
from __future__ import absolute_import
import astropy.units as _u
from .environments import makino as _NFW
from . import simulations as _ps
import matplotlib.gridspec as _gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable
from collections import namedtuple as _namedtuple
import numpy as _np

def create_colorbar(im,
                    ax,
                    fig,
                    size='5%',
                    padding=0.05,
                    position='right',
                    divider=None,
                    use_ax=False):  #pragma: no cover
    if use_ax is False:
        if divider is None:
            divider = _make_axes_locatable(ax)
        cax = divider.append_axes(position, size=size, pad=padding)
    else:
        cax = ax
    ca = fig.colorbar(im, cax=cax)
    cax.yaxis.set_ticks_position(position)
    cax.yaxis.set_label_position(position)
    ca.solids.set_rasterized(True)
    return (ca, divider, cax)



def figsize(scale, ratio=None):
    fig_width_pt = 240                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    if ratio is None:
        ratio = golden_mean
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*ratio # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size


def newfig(width, ratio=None):
    import matplotlib.pyplot as _plt
    _plt.clf()
    fig = _plt.figure(figsize=figsize(width, ratio))
    ax = fig.add_subplot(111)
    return fig, ax

def savefig(filename, fig, dpi=1000, kwargs=None, png=False, eps=False):
    if kwargs is None:
        kwargs = {}
    if png == True:
        fig.savefig('{}.png'.format(filename), dpi=dpi, **kwargs)
    else:
        if eps:
            fig.savefig('{}.eps'.format(filename), dpi=dpi, **kwargs)
        fig.savefig('{}.pdf'.format(filename), dpi=dpi, **kwargs)

def get_pluto_data_direct(data_object, variable, log, simulation_directory,
                          timestep):
    variable_data = getattr(data_object, variable).T
    if log is True:
        variable_data = _np.log10(variable_data)
    return variable_data


def get_pluto_data_direct_no_log(data_object, variable, log,
                                 simulation_directory, timestep):
    return get_pluto_data_direct(data_object, variable, False,
                                 simulation_directory, timestep)


def get_animation(simulation_directory,
                  timeste_ps,
                  time_scaling,
                  length_scaling,
                  variable,
                  figure_properties,
                  log=True,
                  vmax=None,
                  vmin=None,
                  vmax2=None,
                  vmin2=None,
                  timebar_fontsize='large',
                  cbar_size='5%',
                  load_data_function=get_pluto_data_direct,
                  load_data_function2=None,
                  cmap2=None,
                  mirror_horizontally=False,
                  need_two_colorbars=True,
                  simulation_directory2=None,
                  draw_timebar=True,
                  draw_function=None,
                  length_scaling2=None):  #pragma: no cover

    from matplotlib import pyplot as _plt

    # load data function
    def load_data(pluto_data, t=-1):
        v_data = []
        if len(variable) is 2:
            v_data.append(
                load_data_function(pluto_data, variable[0], log,
                                   simulation_directory, timeste_ps[t]))

            sim_d2 = simulation_directory2
            if simulation_directory2 is None:
                sim_d2 = simulation_directory
            if load_data_function2 is not None:
                v_data.append(
                    load_data_function2(pluto_data, variable[1], log, sim_d2,
                                        timeste_ps[t]))
            else:
                v_data.append(
                    load_data_function(pluto_data, variable[1], log, sim_d2,
                                       timeste_ps[t]))
        else:
            v_data = load_data_function(pluto_data, variable, log,
                                        simulation_directory, timeste_ps[t])
        return v_data

    # plot data function
    def plot_data(variable_data):
        if multiple_var is True:
            im1 = ax.pcolormesh(
                X1, X2, variable_data[0], shading='flat', vmin=vmin, vmax=vmax)
            if mirror_horizontally is True:
                im1_mirror = ax.pcolormesh(
                    X1,
                    -X2,
                    variable_data[0],
                    shading='flat',
                    vmin=vmin,
                    vmax=vmax)
            X1_2_temp = X1
            X2_2_temp = X2
            if length_scaling2 is not None:
                X1_2_temp = X1_2
                X2_2_temp = X2_2
            if cmap2 is not None:
                im2 = ax.pcolormesh(
                    -X1_2_temp,
                    X2_2_temp,
                    variable_data[1],
                    shading='flat',
                    vmin=vmin2,
                    vmax=vmax2,
                    cmap=cmap2)
                if mirror_horizontally is True:
                    im2_mirror = ax.pcolormesh(
                        -X1_2_temp,
                        -X2_2_temp,
                        variable_data[1],
                        shading='flat',
                        vmin=vmin2,
                        vmax=vmax2,
                        cmap=cmap2)
            else:
                im2 = ax.pcolormesh(
                    -X1_2_temp,
                    X2_2_temp,
                    variable_data[1],
                    shading='flat',
                    vmin=vmin2,
                    vmax=vmax2)
                if mirror_horizontally is True:
                    im2_mirror = ax.pcolormesh(
                        -X1_2_temp,
                        -X2_2_temp,
                        variable_data[1],
                        shading='flat',
                        vmin=vmin2,
                        vmax=vmax2)
            if mirror_horizontally is True:
                return (im1, im2, im1_mirror, im2_mirror)
            else:
                return (im1, im2)
        else:
            return ax.pcolormesh(
                X1, X2, variable_data, shading='flat', vmin=vmin, vmax=vmax)

    def update_data(variable_data, im):
        if multiple_var is True:
            variable_data[0] = variable_data[0][:-1, :-1]
            variable_data[1] = variable_data[1][:-1, :-1]
            im[0].set_array(variable_data[0].ravel())
            im[1].set_array(variable_data[1].ravel())
            if mirror_horizontally is True:
                im[2].set_array(variable_data[0].ravel())
                im[3].set_array(variable_data[1].ravel())
        else:
            variable_data = variable_data[:-1, :-1]
            im.set_array(variable_data.ravel())

    # plot colobar function
    def plot_colorbar(images):
        if multiple_var is True:
            (cb1, div, cax1) = create_colorbar(
                images[0],
                ax,
                f1,
                size=cbar_size,
                padding=figure_properties.cbar_padding)
            cb1.set_label(figure_properties.cbar_label[0])
            if need_two_colorbars is True:
                (cb2, div, cax2) = create_colorbar(
                    images[1],
                    ax,
                    f1,
                    size=cbar_size,
                    padding=figure_properties.cbar_padding,
                    position='left',
                    divider=div)
                cb2.set_label(figure_properties.cbar_label[1])
            return div
        else:
            (cb1, div, cax1) = create_colorbar(
                images,
                ax,
                f1,
                size=cbar_size,
                padding=figure_properties.cbar_padding)
            cb1.set_label(figure_properties.cbar_label)
            return div

    # check if we are plotting multiple variables
    multiple_var = len(variable) is 2

    # load the last data step we've been given
    last_data = _ps.load_timestep_data(timeste_ps[-1], simulation_directory)

    if multiple_var is True:
        last_data2 = _ps.load_timestep_data(timeste_ps[-1],
                                            simulation_directory2)

    # load the last variables
    v_last = load_data(last_data)

    # get the last simulation time (unscaled!)
    last_time = last_data.SimTime

    # get colorbar limits if necessary
    if multiple_var is True:
        if vmax is None:
            vmax = _np.max(v_last[0].ravel())
        if vmax2 is None:
            vmax2 = _np.max(v_last[1].ravel())
        if vmin is None:
            vmin = _np.min(v_last[0].ravel())
        if vmin2 is None:
            vmin2 = _np.min(v_last[1].ravel())
    else:
        if vmax is None:
            vmax = _np.max(v_last.ravel())
        if vmin is None:
            vmin = _np.min(v_last.ravel())

    # get cartesian coordinates
    X1, X2 = _ps.sphericaltocartesian(last_data)
    X1 = X1 * length_scaling.value
    X2 = X2 * length_scaling.value

    if length_scaling2 is not None:
        X1_2, X2_2 = _ps.sphericaltocartesian(last_data2)
        X1_2 = X1_2 * length_scaling2.value
        X2_2 = X2_2 * length_scaling2.value

    # create the figure
    f1 = _plt.figure(
        figsize=[figure_properties.width, figure_properties.height])

    # create the axes
    ax = _plt.axes(xlim=figure_properties.xlim, ylim=figure_properties.ylim)

    # load inital data
    initial_data = _ps.load_timestep_data(timeste_ps[0], simulation_directory)

    initial_variable_data = load_data(initial_data)

    # plot initial data
    ims = plot_data(initial_variable_data)

    # set axes properties
    ax.set_aspect(figure_properties.aspect)
    ax.set_xlabel(figure_properties.xlabel)
    ax.set_ylabel(figure_properties.ylabel)
    ax.set_title(figure_properties.suptitle)

    # create colorbar
    div = plot_colorbar(ims)

    # create time bar
    if draw_timebar is True:
        tax = div.append_axes(
            'bottom', size='15%', pad=figure_properties.timebar_padding)
        tax.set_xlim([0, last_time * time_scaling.value])
        tax.spines['top'].set_visible(False)
        tax.spines['right'].set_visible(False)
        tax.spines['bottom'].set_visible(False)
        tax.spines['left'].set_visible(False)
        tax.tick_params(
            axis='y', which='both', right='off', left='off', labelleft='off')
        tax.tick_params(axis='x', which='both', top='off')

    #init function
    def init():
        ims = plot_data(initial_variable_data)
        if draw_timebar is True:
            tax.hlines(1, 0, 10)
            tax.eventplot([0], linewidths=[10], colors='k')
        ax.set_xlabel(figure_properties.xlabel)
        ax.set_ylabel(figure_properties.ylabel)
        ax.set_title(figure_properties.suptitle)
        return ims

    #animation function
    def animate(i):
        if i >= len(timeste_ps):
            i = len(timeste_ps) - 1

        # load timestep data file
        d = _ps.load_timestep_data(timeste_ps[i], simulation_directory)

        # get variable data
        var_data = load_data(d, i)

        # clear plot
        ax.cla()

        # reset limits
        ax.set_xlim(figure_properties.xlim)
        ax.set_ylim(figure_properties.ylim)

        # reset labels
        ax.set_xlabel(figure_properties.xlabel)
        ax.set_ylabel(figure_properties.ylabel)
        ax.set_title(figure_properties.suptitle)

        # plot data
        #update_data(var_data, ims)
        ims = plot_data(var_data)
        if draw_function is not None:
            draw_function(f1, ax, ims)

        # update timebar plot
        if draw_timebar is True:
            tax.cla()
            tax.set_xlim(0, last_time * time_scaling.value)
            tax.hlines(1, 0, last_time * time_scaling.value)
            tax.eventplot(
                [d.SimTime * time_scaling.value], linewidths=[2], colors='k')
            tax.set_xlabel(
                '\nt = {0}'.format(_np.round(d.SimTime * time_scaling)),
                fontsize=timebar_fontsize)

        # clear notebook output
        clear_output(wait=True)

        # print progress bar
        print('Processed {0}/{1}'.format(i, len(timeste_ps)))

        # flush progress string
        sys.stdout.flush()

        return ims

    return animation.FuncAnimation(
        f1, animate, init_func=init, frames=len(timeste_ps), blit=False)


def plot_energy(simulation_directory,
                timeste_ps,
                sim_times,
                j,
                run_code,
                ax=None,
                fig=None,
                plot_theoretical=True,
                plot_flux=True,
                width=10,
                height=10,
                energy_scaling=1 * _u.J,
                draw_legend=True,
                draw_title=True):  #pragma: no cover

    from matplotlib import pyplot as _plt

    # get figure
    if ax is None or fig is None:
        fig = _plt.figure(figsize=(width, height))
        ax = _plt.axes()

    initial_data = _ps.load_timestep_data(0, simulation_directory)
    E_sum, KE_sum, UE_sum, UTh_sum, flux_sum = _ps.calculate_total_run_energy(
        simulation_directory,
        timeste_ps,
        _np.rad2deg(j.theta),
        correct_numerical_errors=False)
    theoretical_energy = _ps.calculate_theoretical_energy(initial_data,
                                                          _np.rad2deg(j.theta),
                                                          j, sim_times)

    scaled_timeste_ps = sim_times * j.time_scaling
    ax.plot(
        scaled_timeste_ps, (UTh_sum - UTh_sum[0]) * energy_scaling.value,
        label='Thermal Energy')
    ax.plot(
        scaled_timeste_ps, (UE_sum - UE_sum[0]) * energy_scaling.value,
        label='Potential Energy')
    ax.plot(
        scaled_timeste_ps,
        KE_sum * energy_scaling.value,
        label='Kinetic Energy')
    ax.plot(
        scaled_timeste_ps, (E_sum - E_sum[0]) * energy_scaling.value,
        '.',
        label='Total Energy')

    if plot_theoretical is True:
        ax.plot(
            scaled_timeste_ps,
            theoretical_energy,
            '.',
            label='Expected Energy')

    # Plot the flux
    if plot_flux is True:
        ax.plot(
            timeste_ps,
            flux_sum * _np.asarray(sim_times),
            '-.',
            label='Energy flux accross boundary')

    # label everything
    if draw_title is True:
        ax.set_title('Energy Components for {0}'.format(run_code))

    #if draw_legend is True:
    #    _plt.legend(loc='best')

    # set axes
    #ax.set_ylim(0, 200000);

    # print percentage error
    # print('Run {0} boundary flux error: {1}'.format(run_code, ((flux_sum[-1] * _np.asarray(sim_times)[-1]) - (theoretical_energy[-1]))/(theoretical_energy[-1])))
    # print('Run {0} measured energy error: {1}'.format(run_code, ((E_sum[-1] - E_sum[0]) - (theoretical_energy[-1]))/(theoretical_energy[-1])))


FigureProperties = _namedtuple('FigureProperties', [
    'width', 'height', 'suptitle', 'aspect', 'xlim', 'ylim', 'xlabel',
    'ylabel', 'cbar_label', 'cbar_padding', 'timebar_padding', 'vmin', 'vmax'
])


def plot_multiple_timesteps(simulation_dir,
                            times,
                            ts,
                            ls,
                            var,
                            figure_properties,
                            ncol=5,
                            log=True,
                            colorbar=True,
                            vs=1):  #pragma: no cover

    from matplotlib import pyplot as _plt

    # calculate number of rows from max number of columns
    nrow = int(_np.ceil(len(times) / float(ncol)))

    # create the figure
    gs = _gridspec.GridSpec(nrow, ncol)

    fig = _plt.figure(figsize=(figure_properties.width,
                               figure_properties.height))
    fig.suptitle(figure_properties.suptitle)

    # load the coordinate data
    coord_data = _ps.load_timestep_data(times[0], simulation_dir)
    X1, X2 = _ps.sphericaltocartesian(coord_data)

    # plot the times
    for i in range(len(times)):

        # load the simulation data
        d = _ps.load_timestep_data(times[i], simulation_dir)

        # setup axes
        ax = _plt.subplot(gs[i // ncol, i % ncol])
        ax.set_aspect(figure_properties.aspect)
        ax.set_xlim(figure_properties.xlim)
        ax.set_ylim(figure_properties.ylim)
        ax.text(
            0.85,
            0.85,
            't = {0}'.format(round((d.SimTime * ts).value, 0)),
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            color='white')

        if (i // ncol) == 2:
            ax.set_xlabel(figure_properties.xlabel)

        if (i % ncol) == 0:
            ax.set_ylabel(figure_properties.ylabel)

        # load variable data
        v_data = getattr(d, var).T * vs
        if log is True:
            v_data = _np.log10(v_data)

        # plot data
        im = ax.pcolormesh(
            X1 * ls.value,
            X2 * ls.value,
            v_data,
            vmin=figure_properties.vmin,
            vmax=figure_properties.vmax)
        im.set_rasterized(True)

        # make colorbar
        if colorbar is True:
            (ca, div, cax) = create_colorbar(im, ax, fig)

            if (i % ncol) == 1:
                ca.set_label(figure_properties.cbar_label)
    return fig
