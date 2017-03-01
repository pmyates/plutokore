from __future__ import print_function
from __future__ import absolute_import
from . import jet as _jet
from . import io as _io
from astropy.convolution import convolve as _convolve
from astropy.convolution import Box2DKernel as _Box2DKernel
from contextlib import contextmanager as _contextmanager
from .utilities import suppress_stdout as _suppress_stdout
import numpy as _np
import sys as _sys

if _sys.version_info[0] == 2:
    from contextlib2 import ExitStack as _ExitStack
else:
    from contextlib import ExitStack as _ExitStack



def load_simulation_data(ids, directory, suppress_output=None):
    data = []
    with _ExitStack() as stack:
        if suppress_output in [None, True]:
            stack.enter_context(_suppress_stdout())
        for i in ids:
            data.append(_io.pload(i, w_dir=directory))
    return data


def get_nlast_info(directory):
    with _suppress_stdout():
        return _io.nlast_info(w_dir=directory)

def get_last_timestep(simulation_directory):
    with _suppress_stdout():
        return _io.nlast_info(w_dir=simulation_directory)['nlast']


def get_output_count(directory):
    return get_nlast_info(directory)['nlast']


def get_tracer_count(directory):
    data = load_timestep_data(0, directory)
    return get_tracer_count_data(data)


def get_tracer_count_data(sim_data):
    return len([trc for trc in sim_data.vars if 'tr' in trc])


def load_timestep_data(timestep, directory, suppress_output=None):
    with _ExitStack() as stack:
        if suppress_output in [None, True]:
            stack.enter_context(_suppress_stdout())
            return _io.pload(timestep, w_dir=directory)


def load_simulation_variables(ids, directory, var_list, suppress_output=None):
    data = {}
    for v in var_list:
        data[v] = []
    for i in ids:
        current_run_data = load_simulation_data([i], directory,
                                              suppress_output)[0]
        for v in var_list:
            data[v].append(getattr(current_run_data, v))
    return data


def load_simulation_times(run_directory, run_timesteps):
    times = []
    for i in run_timesteps:
        with _suppress_stdout():
            energy_data = _io.pload(i, w_dir=run_directory)
        times.append(energy_data.SimTime)
    return times


def sphericaltocartesian(run_data, rotation=None):

    # default rotation is pi / 2 
    # (results in jet pointing up for certain simulations)
    if rotation is None:
        rotation = _np.pi / 2
    # generate the spherical polar grid
    R, Theta = _np.meshgrid(run_data.x1, run_data.x2)
    # rotate theta so that jet is pointing upwards - not necessarily needed
    Theta = Theta - rotation

    # convert spherical polar grid to cartesian
    X1 = R * _np.cos(Theta)
    X2 = R * _np.sin(-Theta)
    return X1, X2


def get_cartesian_grid(irregular_grid, x_count, y_count):
    xi = _np.linspace(irregular_grid[0].min(), irregular_grid[0].max(),
                      x_count)
    yi = _np.linspace(irregular_grid[1].min(), irregular_grid[1].max(),
                      y_count)
    return tuple(_np.meshgrid(xi, yi))


def get_gridded_data(sim_data,
                     data_variable='rho',
                     irregular_grid=None,
                     cart_grid=None,
                     method='cubic'):
    # sort out our grid
    if irregular_grid is None:
        irregular_grid = sphericaltocartesian(sim_data)
    if cart_grid is None:
        cart_grid = get_cartesian_grid(
            irregular_grid, x_count=sim_data.n1_tot, y_count=sim_data.n1_tot)
    print(len(irregular_grid))
    print(len(cart_grid))
    # interpolate data
    gridded_data = _np.ma.masked_invalid(
        griddata(
            (irregular_grid[0].ravel(), irregular_grid[1].ravel()),
            getattr(sim_data, data_variable).ravel(order='F'),
            cart_grid,
            method=method))
    return gridded_data, cart_grid


def calculate_cell_volume(sim_data):
    cell_volumes = _np.zeros((sim_data.n1_tot, sim_data.n2_tot))
    for i in range(0, cell_volumes.shape[0]):
        r = (sim_data.x1r[i + 1]**3) - (sim_data.x1r[i]**3)
        for j in range(0, cell_volumes.shape[1]):
            volume = 2 * _np.pi * (
                _np.cos(sim_data.x2r[j]) - _np.cos(sim_data.x2r[j + 1])) * (r /
                                                                            3)
            cell_volumes[i, j] = volume
    return cell_volumes


def calculate_cell_area(sim_data):
    areas = _np.zeros(sim_data.rho.shape)
    for i in range(0, areas.shape[0]):
        r = (sim_data.x1r[i + 1]**2) - (sim_data.x1r[i]**2)
        for j in range(0, areas.shape[1]):
            area = sim_data.dx2[j] * (r / 2)
            areas[i, j] = sim_data.x1[i]**2 * _np.sin(sim_data.x2[
                j]) * sim_data.dx2[j] * _np.pi * 2
            # areas[i,j] = area
    return areas


def find_last_equal_point_radial(data1, data2, epsilon=1e-5):
    """Returns the last equal points in the first dimension of the data, expects 2D arrays"""
    # find difference between 2 data sets
    difference = abs(data1 - data2)
    indicies = []
    for t_index in range(data1.shape[1]):
        indicies.append(find_last_equal_point(difference[:, t_index]))
    return _np.asarray(indicies)


def find_last_equal_point(difference, epsilon=1e-5):
    """Find the last equal point of two 1D(!) arrays, given the absolute difference between them."""
    return (_np.where(difference < epsilon)[0])[-1]


def replace_with_initial_data(initial_data, new_data, epsilon=1e-5):
    """Replace the 1D new_data array with the 1D intial_data array, from the last equal point onwards"""
    # find the last equal point
    last_index = find_last_equal_point(abs(initial_data - new_data), epsilon)

    # replace with intial data from this point onwards
    ret = _np.copy(new_data)
    ret[last_index:] = initial_data[last_index:]

    return ret


def replace_with_initial_data_radial(initial_data, new_data, epsilon=1e-5):
    """Replaces new_data with inital_data, from the last equal point in the 1st dimensions onwards. Expects 2D arrays."""
    # find the last equal point
    last_index = find_last_equal_point_radial(initial_data, new_data, epsilon)

    # replace with intial data from this point onwards
    ret = _np.copy(new_data)
    for t_index in range(new_data.shape[1]):
        ret[last_index[t_index]:, t_index] = initial_data[last_index[t_index]:,
                                                          t_index]

    return ret


def fix_numerical_errors_single_timestep(run_data, initial_data, var_list):
    for v in var_list:
        va = getattr(run_data, v)
        va = replace_with_initial_data_radial(initial_data[v], va)
        setattr(run_data, v, va)


def fix_numerical_errors(run_data, initial_data, var_list):
    for t_step in range(len(run_data)):
        fix_numerical_errors_single_timestep(run_data[t_step], initial_data,
                                             var_list)


def combine_tracers(simulation_data, ntracers):
    """Helper function to combine multiple tracers into one array. Simply adds them up"""
    ret = _np.zeros_like(simulation_data.tr1)
    for i in range(ntracers):
        ret = ret + getattr(simulation_data, 'tr{0}'.format(i + 1))
    return ret


def clamp_tracers(simulation_data,
                  ntracers,
                  tracer_threshold=1e-7,
                  tracer_effective_zero=1e-20):
    # smooth the tracer data with a 2d box kernel of width 3
    box2d = _Box2DKernel(3)
    radio_combined_tracers = _convolve(
        combine_tracers(simulation_data, ntracers), box2d, boundary='extend')
    radio_tracer_mask = _np.where(radio_combined_tracers > tracer_threshold,
                                  1.0, tracer_effective_zero)

    # create new tracer array that is clamped to tracer values
    clamped_tracers = radio_combined_tracers.copy()
    clamped_tracers[clamped_tracers <=
                    tracer_threshold] = tracer_effective_zero

    return (radio_tracer_mask, clamped_tracers, radio_combined_tracers)

def calculate_actual_jet_opening_angle(run_data, theta_deg):
    indicies = _np.where(run_data.x2 < _np.deg2rad(theta_deg))[0]
    if len(indicies) == 0:
        return (list(range(0, len(run_data.x2 - 1))), theta_deg)
    actual_angle = _np.rad2deg(run_data.x2[indicies[-1]])
    return (indicies, actual_angle)


