from __future__ import print_function
from __future__ import absolute_import
from . import jet as _jet
from . import io as _io
from astropy.convolution import convolve as _convolve
from astropy.convolution import Box2DKernel as _Box2DKernel
from contextlib import contextmanager as _contextmanager
from .utilities import suppress_stdout as _suppress_stdout
from .simulations import calculate_cell_volume, clamp_tracers
import numpy as _np
import sys as _sys

if _sys.version_info[0] == 2:
    from contextlib2 import ExitStack as _ExitStack
else:
    from contextlib import ExitStack as _ExitStack


def calculate_energy(run_data,
                     initial_data,
                     gamma=5.0 / 3.0,
                     volume=None,
                     tracer_weighted=False,
                     tracer_or_not=True,
                     ntracers=1,
                     tracer_threshold=0.005):
    """Calculates the change in energy components for the given run_data and initial_data"""

    # calculate volume if required
    if volume is None:
        volume = calculate_cell_volume(run_data)

    if tracer_weighted is True:
        (radio_tracer_mask, clamped_tracers,
         radio_combined_tracers) = clamp_tracers(
             run_data, ntracers=ntracers, tracer_threshold=tracer_threshold)
        if tracer_or_not is False:
            radio_tracer_mask = 1 - radio_tracer_mask
            radio_combined_tracers = 1 - radio_combined_tracers
            clamped_tracers = 1 - clamped_tracers

        volume = volume * radio_tracer_mask

    # kinetic energy
    # velocity term - need to find total velocity
    if tracer_weighted is True:
        velocity = _np.sqrt((run_data.vx1 * radio_tracer_mask)**2 + (
            run_data.vx2 * radio_tracer_mask)**2)
    else:
        velocity = _np.sqrt(run_data.vx1**2 + run_data.vx2**2)
    # density term
    if tracer_weighted is True:
        density = run_data.rho * radio_tracer_mask
    else:
        density = run_data.rho

    # kinetic energy calculation - for individual cells
    dKE = (1.0 / 2.0) * (velocity**2) * density * volume

    # potential energy

    # potential energy calculation - for individual cells
    dUE = -(_np.log(initial_data.rho) * density * volume) / (gamma)

    # thermal energy
    # pressure term
    if tracer_weighted is True:
        pressure = run_data.prs * radio_tracer_mask
    else:
        pressure = run_data.prs

    # thermal energy calculation - for individual cells
    dUth = (3.0 / 2.0) * pressure * volume

    # total energy
    dE = dKE + dUE + dUth

    return (dE, dKE, dUE, dUth)


def calculate_timestep_energy(run_directory,
                              timestep,
                              initial_data,
                              gamma=5.0 / 3.0,
                              correct_numerical_errors=None,
                              var_list=None):
    """Calculates the change in energy components for a single timestep"""
    total_energy, kinetic_energy, potential_energy, thermal_energy = calculate_run_energy(
        run_directory, [timestep], initial_data, gamma,
        correct_numerical_errors, var_list)

    total_energy = total_energy[0]
    kinetic_energy = kinetic_energy[0]
    potential_energy = potential_energy[0]
    thermal_energy = thermal_energy[0]
    return (total_energy, kinetic_energy, potential_energy, thermal_energy)


def calculate_run_energy(run_directory,
                         timesteps,
                         initial_data=None,
                         gamma=5.0 / 3.0,
                         correct_numerical_errors=None,
                         var_list=None,
                         tracer_weighted=False,
                         tracer_or_not=True,
                         ntracers=0):
    """Calculates the change in energy components for a range of timesteps"""
    from .simulations import load_timestep_data, calculate_cell_volume

    total_energy = []
    kinetic_energy = []
    potential_energy = []
    thermal_energy = []
    initial_var_values = {}

    if var_list is None:
        var_list = ['prs', 'rho']

    if initial_data is None:
        initial_data = load_timestep_data(timesteps[0], run_directory)

    if correct_numerical_errors in [None, True]:
        for v in var_list:
            initial_var_values[v] = getattr(initial_data, v)

    volume = calculate_cell_volume(initial_data)
    run_data = None

    for time in timesteps:
        run_data = load_timestep_data(time, run_directory)

        # perform numerical corrections
        if correct_numerical_errors in [None, True]:
            fix_numerical_errors_single_timestep(run_data, initial_var_values,
                                                 var_list)

        # calculate energy
        dE, dKE, dUE, dUth = calculate_energy(run_data, initial_data, gamma,
                                              volume, tracer_weighted,
                                              tracer_or_not, ntracers)

        # total energy
        total_energy.append(dE)
        kinetic_energy.append(dKE)
        potential_energy.append(dUE)
        thermal_energy.append(dUth)
    return (total_energy, kinetic_energy, potential_energy, thermal_energy)


def calculate_total_run_energy(run_directory,
                               timesteps,
                               theta_deg,
                               initial_data=None,
                               gamma=5.0 / 3.0,
                               correct_numerical_errors=None,
                               var_list=None):
    """Calculates the energy component sum for each timestep"""
    if initial_data is None:
        initial_data = load_simulation_data([timesteps[0]], run_directory)[0]
    # calculate energies
    total_energy, kinetic_energy, potential_energy, thermal_energy = calculate_run_energy(
        run_directory, timesteps, initial_data, gamma,
        correct_numerical_errors, var_list)

    # sum energies
    E_sum = _np.sum(_np.asarray(total_energy), (1, 2))
    KE_sum = _np.sum(_np.asarray(kinetic_energy), (1, 2))
    UE_sum = _np.sum(_np.asarray(potential_energy), (1, 2))
    UTh_sum = _np.sum(_np.asarray(thermal_energy), (1, 2))

    # calculate flux
    indicies, actual_angle = calculate_actual_jet_opening_angle(initial_data,
                                                                theta_deg)
    volume = calculate_cell_volume(initial_data)
    area = calculate_cell_area(initial_data)

    thermal_energy_density = thermal_energy / volume
    kinetic_energy_density = kinetic_energy / volume
    potential_energy_density = potential_energy / volume

    var_list = ["vx1", "vx2"]
    v = load_simulation_variables(timesteps, run_directory, var_list)
    vx1 = _np.asarray(v["vx1"])
    vx2 = _np.asarray(v["vx2"])

    flux = ((thermal_energy_density + kinetic_energy_density) *
            _np.sqrt(vx1**2 + vx2**2) * area)
    # array is timestep, r index, theta index
    flux_sum = _np.sum(flux[:, 0, 0:indicies[-1]], (1))

    return (E_sum, KE_sum, UE_sum, UTh_sum, flux_sum)

def calculate_theoretical_energy(run_data, theta_deg, run_jet, run_times):
    """Calculates the theoretical change in total energy given the jet opening angle for the run times"""
    indicies, actual_angle = calculate_actual_jet_opening_angle(run_data,
                                                                theta_deg)
    new_run_jet = _jet.AstroJet(actual_angle, run_jet.M_x, run_jet.c_x,
                                run_jet.rho_0, run_jet.Q, run_jet.gamma)
    new_run_jet.calculate_length_scales()

    theoretical_energy = (((run_jet.M_x**3) * (new_run_jet.Omega * (
        (run_jet.L_1b / run_jet.L_1)**2) / 2.0)) + (new_run_jet.Omega / 2.0) *
                          (9.0 / 10.0) * (
                              (run_jet.L_1b / run_jet.L_1)**2) * run_jet.M_x
                          ) * _np.asarray(run_times)
    return theoretical_energy


def calculate_energy_multiple_timesteps(run_data,
                                        gamma=5.0 / 3.0,
                                        initial_data=None):
    """Calculates the energy components for each timestep in run_data, where run_data is a list of pload objects"""
    total_energy = []
    kinetic_energy = []
    potential_energy = []
    thermal_energy = []

    if initial_data is None:
        initial_data = run_data[0]

    # use new method of calculating volume
    volume = calculate_cell_volume(run_data[0])

    for energy_data in run_data:
        # calculate energy
        dE, dKE, dUE, dUth = calculate_energy(energy_data, initial_data, gamma,
                                              volume)

        # total energy
        total_energy.append(dE)
        kinetic_energy.append(dKE)
        potential_energy.append(dUE)
        thermal_energy.append(dUth)

    return (total_energy, kinetic_energy, potential_energy, thermal_energy)
