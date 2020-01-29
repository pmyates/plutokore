from __future__ import print_function
from __future__ import absolute_import
from . import jet as _jet
from . import io as _io
from astropy.convolution import convolve as _convolve
from astropy.convolution import Box2DKernel as _Box2DKernel
import astropy.units as _u
import astropy.constants as _const
from contextlib import contextmanager as _contextmanager
from .utilities import suppress_stdout as _suppress_stdout
import numpy as _np
import sys as _sys
import os as _os
import h5py as _h5py
from pathlib import Path as _Path

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


def get_nlast_info(directory, datatype='double'):
    with _suppress_stdout():
        return _io.nlast_info(w_dir=directory, datatype=datatype)

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

def get_times(sim_dir, time_fname='dbl.out'):
    with open(_os.path.join(sim_dir, time_fname), 'r') as f_var:
        tlist = []
        for l in f_var.readlines():
            tlist.append(float(l.split()[1]))
        return _np.asarray(tlist)

def load_timestep_data(timestep, directory, suppress_output=None, mmap=True, datatype='double'):
    with _ExitStack() as stack:
        if suppress_output in [None, True]:
            stack.enter_context(_suppress_stdout())
        return _io.pload(timestep, w_dir=directory, mmap=mmap, datatype=datatype)


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
    R, Theta = _np.meshgrid(run_data.x1r, run_data.x2r)
    # rotate theta so that jet is pointing upwards - not necessarily needed
    Theta = Theta - rotation

    # convert spherical polar grid to cartesian
    X1 = R * _np.cos(Theta)
    X2 = R * _np.sin(-Theta)
    return X1, X2


def get_cartesian_grid(irregular_grid, x_count, y_count, min_offset=0.1):
    xmin = irregular_grid[0].min()
    ymin = irregular_grid[1].min()
    xi = _np.linspace(xmin + min_offset, irregular_grid[0].max(),
                      x_count)
    yi = _np.linspace(ymin + min_offset, irregular_grid[1].max(),
                      y_count)
    return tuple(_np.meshgrid(xi, yi))


def get_gridded_data(data,
                     irregular_grid,
                     cart_grid=None,
                     method='cubic',
                     xcount=200,
                     ycount=200,
                     fill_value=0):
    from scipy.interpolate import griddata

    # sort out our grid
    if cart_grid is None:
        cart_grid = get_cartesian_grid(
            irregular_grid, x_count=xcount, y_count=ycount)
    print(len(irregular_grid))
    print(len(cart_grid))
    print('fill value is: {0}'.format(fill_value))
    # interpolate data
    gridded_data = griddata(
        (irregular_grid[0].flatten(), irregular_grid[1].flatten()),
        data.flatten('F'),
        cart_grid,
        method=method,
        fill_value=fill_value)
    #gridded_data = _np.ma.masked_invalid(gridded_data)
    return gridded_data, cart_grid


def calculate_cell_volume(sim_data):
    return calculate_cell_volume_fast(sim_data)
    # cell_volumes = _np.zeros((sim_data.n1_tot, sim_data.n2_tot))
    # if (sim_data.geometry == 'SPHERICAL'):
    #     for i in range(0, cell_volumes.shape[0]):
    #         r = (sim_data.x1r[i + 1]**3) - (sim_data.x1r[i]**3)
    #         for j in range(0, cell_volumes.shape[1]):
    #             volume = 2 * _np.pi * (
    #                 _np.cos(sim_data.x2r[j]) - _np.cos(sim_data.x2r[j + 1])) * (r /
    #                                                                             3)
    #             cell_volumes[i, j] = volume
    # elif (sim_data.geometry == 'CARTESIAN'):
    #     for i in range(0, cell_volumes.shape[0]):
    #         for j in range(0, cell_volumes.shape[1]):
    #             cell_volumes[i, j] = sim_data.x1r[i] * sim_data.x2r[j] * 1
    # return cell_volumes

def calculate_cell_volume_fast(sim_data):
    if (sim_data.geometry == 'SPHERICAL'):
        return ((2 * _np.pi) *
                _np.outer(((sim_data.x1r[1:] ** 3 - sim_data.x1r[:-1] ** 3) / 3.0),
                _np.cos(sim_data.x2r[:-1]) - _np.cos(sim_data.x2r[1:])))
    elif (sim_data.geometry == 'CARTESIAN'):
        return _np.multiply.outer(_np.multiply.outer(sim_data.dx1, sim_data.dx2), sim_data.dx3)

def calculate_cell_area(sim_data):
    areas = _np.zeros(sim_data.rho.shape)
    if (sim_data.geometry == 'SPHERICAL'):
        for i in range(0, areas.shape[0]):
            r = (sim_data.x1r[i + 1]**2) - (sim_data.x1r[i]**2)
            for j in range(0, areas.shape[1]):
                areas[i, j] = sim_data.x1[i]**2 * _np.sin(sim_data.x2[
                    j]) * sim_data.dx2[j] * _np.pi * 2
    elif (sim_data.geometry == 'CARTESIAN'):
        for i in range(0, areas.shape[0]):
            for j in range(0, areas.shape[1]):
                areas[i, j] = sim_data.x1r[i] * sim_data.x2r[j]
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

def get_hdf5_output_count(*, sim_path, data_type = 'float'):
    """
    Returns the last hdf5 output number, obtained from the out files.

    Parameters
    ----------
    sim_path : Path
        The simulation path
    data_type : str
        The data type of the output file (default: 'float')

    Returns
    -------
    integer
        The number of the final hdf5 output
    """

    if data_type == 'float':
        out_fname = 'flt.h5.out'
    elif data_type == 'double':
        out_fname = 'dbl.h5.out'

    with open(sim_path / out_fname, 'r') as f:
        last_output = int(f.readlines()[-1].split()[0])

    return last_output

def load_hdf5_data(*, sim_path, output, data_type = 'float'):
    """
    Loads the specified hdf5 data output file from the simulation directory.


    Parameters
    ----------
    sim_path : Path
        The simulation path
    output : int
        The number of the output file to be loaded
    data_type : str
        The data type of the output file (default: 'float')

    Returns
    -------
    HDF5 File
        The loaded output file
    """
    # Make sure we have the correct datatype
    if data_type == 'float':
        dext = 'flt.h5'
    elif data_type == 'double':
        dext = 'dbl.h5'


    # Load the data file
    data_file_path = sim_path / _Path(f'data.{output:04d}.{dext}')
    data_file = _h5py.File(data_file_path, mode = 'r')

    # Set some basic properties
    setattr(data_file, 'output', output)
    setattr(data_file, 'sim_time', data_file[f'Timestep_{output}'].attrs['Time'])
    setattr(data_file, 'variable_path', f'Timestep_{output}/vars')
    setattr(data_file, 'geometry', 'CARTESIAN')

    # Calculate the unit values
    unit_length = 1 * _u.kpc
    unit_density = (0.60364 * _u.u / (_u.cm ** 3)).to(_u.g / _u.cm ** 3)
    unit_speed = _const.c
    unit_time = (unit_length / unit_speed).to(_u.Myr)
    unit_pressure = (unit_density * (unit_speed ** 2)).to(_u.Pa)
    unit_mass = (unit_density * (unit_length ** 3)).to(_u.kg)
    unit_energy = (unit_mass * (unit_length**2) / (unit_time**2)).to(_u.J)

    uv = UnitValues(
    density=unit_density,
    length=unit_length,
    time=unit_time,
    mass=unit_mass,
    pressure=unit_pressure,
    energy=unit_energy,
    speed=unit_speed,
    )

    # Set the unit values
    setattr(data_file, 'unit_length', unit_length)
    setattr(data_file, 'unit_density', unit_density)
    setattr(data_file, 'unit_speed', unit_speed)
    setattr(data_file, 'unit_time', unit_time)
    setattr(data_file, 'unit_pressure', unit_pressure)
    setattr(data_file, 'unit_mass', unit_mass)
    setattr(data_file, 'unit_energy', unit_energy)
    setattr(data_file, 'unit_values', uv)

    # Set the variables
    variables = list(data_file[data_file.variable_path])
    for v in variables:
        setattr(data_file, v, data_file[f'{data_file.variable_path}/{v}'])
    setattr(data_file, 'vars', variables)

    # Set the coords
    coords = ['X', 'Y', 'Z']
    for c in coords:
        setattr(data_file, f'cc{c.lower()}', data_file[f'/cell_coords/{c}'])
        setattr(data_file, f'nc{c.lower()}', data_file[f'/node_coords/{c}'])

    # Set the 1D cell midpoint cooordinate arrays
    setattr(data_file, 'mx', data_file.ccx[0,0,:])
    setattr(data_file, 'my', data_file.ccy[0,:,0])
    setattr(data_file, 'mz', data_file.ccz[:,0,0])

    setattr(data_file, 'ex', data_file.ncx[0,0,:])
    setattr(data_file, 'ey', data_file.ncy[0,:,0])
    setattr(data_file, 'ez', data_file.ncz[:,0,0])

    setattr(data_file, 'dx', np.diff(data_file.ex))
    setattr(data_file, 'dy', np.diff(data_file.ey))
    setattr(data_file, 'dz', np.diff(data_file.ez))

    setattr(data_file, 'dx1', data_file.dx)
    setattr(data_file, 'dx2', data_file.dy)
    setattr(data_file, 'dx3', data_file.dz)

    # Set midpoint index attributes
    setattr(data_file, 'mid_x', data_file.mx.shape[0] // 2)
    setattr(data_file, 'mid_y', data_file.my.shape[0] // 2)
    setattr(data_file, 'mid_z', data_file.mz.shape[0] // 2)

    # Set the tracer count
    setattr(data_file, 'ntracers', len([t for t in variables if 'tr' in t]))

    # Return our loaded data file
    return data_file
