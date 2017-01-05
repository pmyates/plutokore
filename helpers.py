import os
import sys
#sys.path.append('F:/MyPython_Modules/Lib/site-packages/')
import numpy as np
import pyPLUTO as pp
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import clear_output, Math, Latex, display
from contextlib import contextmanager
from astropy import units as u
from astropy import constants as const
from contextlib2 import ExitStack
from tabulate import tabulate
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define some helper functions
def video(fname, mimetype):
    from IPython.display import HTML
    video_encoded = open(fname, "rb").read().encode("base64")
    video_tag = '<video autoplay controls alt="test" src="data:video/{0};base64,{1}">'.format(mimetype, video_encoded)
    return HTML(data=video_tag)

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def create_colorbar(im, ax, fig, size='5%', padding=0.05, position='right', divider=None):
    if divider is None:
        divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=padding)
    ca = fig.colorbar(im, cax=cax)
    cax.yaxis.set_ticks_position(position)
    cax.yaxis.set_label_position(position)
    return (ca, divider, cax)

from IPython.display import Markdown
def printmd(string):
    display(Markdown(string))

def figsize(scale, ratio=None):
    fig_width_pt = 418.25368                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    if ratio is None:
        ratio = golden_mean
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*ratio # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def newfig(width, ratio=None):
    plt.clf()
    fig = plt.figure(figsize=figsize(width, ratio))
    ax = fig.add_subplot(111)
    return fig, ax

def savefig(filename, fig, dpi=300):
    fig.savefig('./Images/thesis/{}.pgf'.format(filename), dpi=dpi)
    fig.savefig('./Images/thesis/{}.pdf'.format(filename), dpi=dpi)

def get_last_timestep(simulation_directory):
    with suppress_stdout():
        return pp.nlast_info(w_dir=simulation_directory)['nlast']

from collections import namedtuple
UnitValues = namedtuple('UnitValues', ['density', 'length', 'time', 'mass', 'pressure', 'energy', 'speed'])

def get_unit_values(environment, jet):

    # calculate unit values
    unit_density = environment.get_density(jet.L_1b)
    unit_length = jet.length_scaling
    unit_time = jet.time_scaling
    unit_mass = (unit_density * (unit_length ** 3)).to(u.kg)
    unit_pressure = (unit_mass / (unit_length * unit_time ** 2)).to(u.Pa)
    unit_energy = (unit_mass * (unit_length ** 2) / (unit_time ** 2)).to(u.J)
    unit_speed = environment.sound_speed

    return UnitValues(density=unit_density,
                      length=unit_length,
                      time=unit_time,
                      mass=unit_mass,
                      pressure=unit_pressure,
                      energy=unit_energy,
                      speed=unit_speed)
