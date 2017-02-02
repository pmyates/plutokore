import os as _os
import sys as _sys
import numpy as _np
import pyPLUTO as _pp
from IPython.display import display as _display
from IPython.display import Markdown as _Markdown
from contextlib import contextmanager as _contextmanager
from astropy import units as _u
from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable
from collections import namedtuple as _namedtuple


# Define some helper functions
def video(fname, mimetype):
    from IPython.display import HTML
    video_encoded = open(fname, "rb").read().encode("base64")
    video_tag = '<video autoplay controls alt="test" src="data:video/{0};base64,{1}">'.format(
        mimetype, video_encoded)
    return HTML(data=video_tag)


@_contextmanager
def suppress_stdout():
    with open(_os.devnull, "w") as devnull:
        old_stdout = _sys.stdout
        _sys.stdout = devnull
        try:
            yield
        finally:
            _sys.stdout = old_stdout


def create_colorbar(im,
                    ax,
                    fig,
                    size='5%',
                    padding=0.05,
                    position='right',
                    divider=None,
                    use_ax=False):
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


def printmd(string):
    _display(_Markdown(string))


def figsize(scale, ratio=None):
    fig_width_pt = 418.25368  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (
        _np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    if ratio is None:
        ratio = golden_mean
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * ratio  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def newfig(width, ratio=None):
    import matplotlib.pyplot as _plt
    _plt.clf()
    fig = _plt.figure(figsize=figsize(width, ratio))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename, fig, dpi=300):
    fig.savefig('./Images/thesis/{}.pgf'.format(filename), dpi=dpi)
    fig.savefig('./Images/thesis/{}.pdf'.format(filename), dpi=dpi)


def get_last_timestep(simulation_directory):
    with suppress_stdout():
        return _pp.nlast_info(w_dir=simulation_directory)['nlast']


UnitValues = _namedtuple(
    'UnitValues',
    ['density', 'length', 'time', 'mass', 'pressure', 'energy', 'speed'])


def get_unit_values(environment, jet):

    # calculate unit values
    unit_density = environment.get_density(jet.L_1b)
    unit_length = jet.length_scaling
    unit_time = jet.time_scaling
    unit_mass = (unit_density * (unit_length**3)).to(_u.kg)
    unit_pressure = (unit_mass / (unit_length * unit_time**2)).to(_u.Pa)
    unit_energy = (unit_mass * (unit_length**2) / (unit_time**2)).to(_u.J)
    unit_speed = environment.sound_speed

    return UnitValues(
        density=unit_density,
        length=unit_length,
        time=unit_time,
        mass=unit_mass,
        pressure=unit_pressure,
        energy=unit_energy,
        speed=unit_speed)


def close_open_hdf5_files():
    import gc
    import h5py
    for obj in gc.get_objects():  # Browse through ALL objects
        if isinstance(obj, h5py.File):  # Just HDF5 files
            try:
                obj.close()
            except:
                pass  # Was already closed
