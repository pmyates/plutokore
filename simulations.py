import jet as _jet
import environments.makino as _NFW
from astropy.convolution import convolve as _convolve
from astropy.convolution import Box2DKernel as _Box2DKernel
from contextlib2 import ExitStack as _ExitStack
from contextlib import contextmanager as _contextmanager
from helpers import suppress_stdout as _suppress_stdout
import pyPLUTO as _pp
import numpy as _np

def LoadSimulationData(ids, directory, suppress_output = None):
    data = []
    with _ExitStack() as stack:
        if suppress_output in [None, True]:
            stack.enter_context(_suppress_stdout())
        for i in ids:
            data.append(_pp.pload(i, w_dir=directory))
    return data

def load_timestep_data(timestep, directory, suppress_output = None):
    with _ExitStack() as stack:
        if suppress_output in [None, True]:
            stack.enter_context(_suppress_stdout())
            return _pp.pload(timestep, w_dir=directory)

def load_simulation_variables(ids, directory, var_list, suppress_output = None):
    data = {}
    for v in var_list:
        data[v] = []
    for i in ids:
        current_run_data = LoadSimulationData([i], directory, suppress_output)[0]
        for v in var_list:
            data[v].append(getattr(current_run_data, v))
    return data

def LoadSimulationTimes(run_directory, run_timesteps):
    times = []
    for i in run_timesteps:
        with _suppress_stdout():
            energy_data = _pp.pload(i, w_dir=run_directory)
        times.append(energy_data.SimTime)
    return times

def sphericaltocartesian(run_data):
    # generate the spherical polar grid
    R, Theta = _np.meshgrid(run_data.x1, run_data.x2)
    # rotate theta so that jet is pointing upwards - not necessarily needed
    Theta = Theta - _np.pi/2

    # convert spherical polar grid to cartesian
    X1 = R * _np.cos(Theta)
    X2 = R * _np.sin(-Theta)
    return X1, X2

def get_cartesian_grid(irregular_grid, x_count, y_count):
    xi = _np.linspace(irregular_grid[0].min(), irregular_grid[0].max(), x_count)
    yi = _np.linspace(irregular_grid[1].min(), irregular_grid[1].max(), y_count)
    return tuple(_np.meshgrid(xi, yi))

def get_gridded_data(sim_data, data_variable='rho', irregular_grid=None, cart_grid=None, method='cubic'):
    # sort out our grid
    if irregular_grid is None:
        irregular_grid = sphericaltocartesian(sim_data)
    if cart_grid is None:
        cart_grid = get_cartesian_grid(irregular_grid, x_count=sim_data.n1_tot, y_count=sim_data.n1_tot)
    print(len(irregular_grid))
    print(len(cart_grid))
    # interpolate data
    gridded_data = _np.ma.masked_invalid(griddata((irregular_grid[0].ravel(), irregular_grid[1].ravel()),
                                                  getattr(sim_data, data_variable).ravel(order='F'),
                                                  cart_grid, method=method))
    return gridded_data, cart_grid

def calculate_cell_volume(sim_data):
    cell_volumes = _np.zeros((sim_data.n1_tot, sim_data.n2_tot))
    for i in range(0, cell_volumes.shape[0]):
        r = (sim_data.x1r[i+1]**3) - (sim_data.x1r[i]**3)
        for j in range(0, cell_volumes.shape[1]):
            volume = 2*_np.pi * (_np.cos(sim_data.x2r[j]) - _np.cos(sim_data.x2r[j+1])) * (r/3)
            cell_volumes[i,j] = volume
    return cell_volumes

def calculate_cell_area(sim_data):
    areas = _np.zeros(sim_data.rho.shape)
    for i in range(0, areas.shape[0]):
        r = (sim_data.x1r[i+1]**2) - (sim_data.x1r[i]**2)
        for j in range(0, areas.shape[1]):
            area = sim_data.dx2[j] * (r/2)
            areas[i,j] = sim_data.x1[i] ** 2 * _np.sin(sim_data.x2[j]) * sim_data.dx2[j] * _np.pi * 2
            # areas[i,j] = area
    return areas

def find_last_equal_point_radial(data1, data2, epsilon=1e-5):
    """Returns the last equal points in the first dimension of the data, expects 2D arrays"""
    # find difference between 2 data sets
    difference = abs(data1 - data2)
    indicies = []
    for t_index in range(data1.shape[1]):
        indicies.append(find_last_equal_point(difference[:,t_index]))
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
        ret[last_index[t_index]:,t_index] = initial_data[last_index[t_index]:,t_index]
    
    return ret

def fix_numerical_errors_single_timestep(run_data, initial_data, var_list):
    for v in var_list:
        va = getattr(run_data, v)
        va = replace_with_initial_data_radial(initial_data[v], va)
        setattr(run_data, v, va)

def fix_numerical_errors(run_data, initial_data, var_list):
    for t_step in range(len(run_data)):
        fix_numerical_errors_single_timestep(run_data[t_step], initial_data, var_list)

def combine_tracers(simulation_data, ntracers):
    """Helper function to combine multiple tracers into one array. Simply adds them up"""
    ret = _np.zeros_like(simulation_data.tr1)
    for i in range(ntracers):
        ret = ret + getattr(simulation_data, 'tr{0}'.format(i+1))
    return ret

def clamp_tracers(simulation_data, ntracers,
                  tracer_threshold = 1e-7,
                  tracer_effective_zero = 1e-20):
    # smooth the tracer data with a 2d box kernel of width 3
    box2d = _Box2DKernel(3)
    radio_combined_tracers = _convolve(combine_tracers(simulation_data, ntracers), box2d, boundary='extend')
    radio_tracer_mask = _np.where(radio_combined_tracers > tracer_threshold, 1.0, tracer_effective_zero)

    # create new tracer array that is clamped to tracer values
    clamped_tracers = radio_combined_tracers.copy()
    clamped_tracers[clamped_tracers <= tracer_threshold] = tracer_effective_zero
    
    return (radio_tracer_mask, clamped_tracers, radio_combined_tracers)

def calculate_energy(run_data, initial_data, gamma=5.0/3.0, volume=None, tracer_weighted = False, tracer_or_not = True, ntracers=1,
                    tracer_threshold=0.005):
    # calculate volume if required
    if volume is None:
        volume = calculate_cell_volume(run_data)
    
    if tracer_weighted is True:
        (radio_tracer_mask, clamped_tracers, radio_combined_tracers) = clamp_tracers(run_data, ntracers=ntracers, 
                                                                                     tracer_threshold=tracer_threshold)
        if tracer_or_not is False:
            radio_tracer_mask = 1 - radio_tracer_mask
            radio_combined_tracers = 1 - radio_combined_tracers
            clamped_tracers = 1 - clamped_tracers
            
        volume = volume * radio_tracer_mask 
        
    # kinetic energy
    # velocity term - need to find total velocity
    if tracer_weighted is True:
        velocity = _np.sqrt((run_data.vx1 * radio_tracer_mask)**2 + (run_data.vx2 * radio_tracer_mask)**2)
    else:
        velocity = _np.sqrt(run_data.vx1**2 + run_data.vx2**2)
    # density term
    if tracer_weighted is True:
        density = run_data.rho * radio_tracer_mask
    else:
        density = run_data.rho

    # kinetic energy calculation - for individual cells
    dKE = (1.0/2.0) * (velocity ** 2) * density * volume

    # potential energy

    # potential energy calculation - for individual cells
    dUE = - (_np.log(initial_data.rho) * density * volume) / (gamma)

    # thermal energy
    # pressure term
    if tracer_weighted is True:
        pressure = run_data.prs * radio_tracer_mask
    else:
        pressure = run_data.prs

    # thermal energy calculation - for individual cells
    dUth = (3.0/2.0) * pressure * volume
    
    # total energy
    dE = dKE + dUE + dUth
    
    return (dE, dKE, dUE, dUth)

def calculate_timestep_energy(run_directory, timestep, initial_data, gamma=5.0/3.0, correct_numerical_errors = None, var_list = None):
    total_energy, kinetic_energy, potential_energy, thermal_energy = calculate_run_energy(run_directory,
                                                                                          [timestep],
                                                                                          initial_data,
                                                                                          gamma,
                                                                                          correct_numerical_errors,
                                                                                          var_list)
    
    total_energy = total_energy[0]
    kinetic_energy = kinetic_energy[0]
    potential_energy = potential_energy[0]
    thermal_energy = thermal_energy[0]
    return (total_energy, kinetic_energy, potential_energy, thermal_energy) 

def calculate_run_energy(run_directory, timesteps, initial_data=None, gamma=5.0/3.0, correct_numerical_errors = None, var_list = None,
                         tracer_weighted = False,
                         tracer_or_not = True,
                         ntracers=0):
    total_energy = []
    kinetic_energy = []
    potential_energy = []
    thermal_energy = []
    initial_var_values = {}
    
    if var_list is None:
        var_list = ['prs', 'rho']
    
    if initial_data is None:
        initial_data = LoadSimulationData(timesteps[0], run_directory)[0]
        
    if correct_numerical_errors in [None, True]:
        for v in var_list:
            initial_var_values[v] = getattr(initial_data, v)
    
    volume = calculate_cell_volume(initial_data)
    run_data = None
    
    for time in timesteps:
        run_data = LoadSimulationData([time], run_directory)[0]
        
        # perform numerical corrections
        if correct_numerical_errors in [None, True]:
            fix_numerical_errors_single_timestep(run_data, initial_var_values, var_list)
        
        # calculate energy
        dE, dKE, dUE, dUth = calculate_energy(run_data, initial_data, gamma, volume, tracer_weighted, tracer_or_not, ntracers)
        
        # total energy
        total_energy.append(dE)
        kinetic_energy.append(dKE)
        potential_energy.append(dUE)
        thermal_energy.append(dUth)
    return (total_energy, kinetic_energy, potential_energy, thermal_energy)

def calculate_total_run_energy(run_directory, timesteps, theta_deg, initial_data=None, gamma=5.0/3.0, correct_numerical_errors=None,
                               var_list=None):
    if initial_data is None:
        initial_data = LoadSimulationData([timesteps[0]], run_directory)[0]
    # calculate energies
    total_energy, kinetic_energy, potential_energy, thermal_energy = calculate_run_energy(run_directory, timesteps, initial_data, gamma, correct_numerical_errors, var_list)
    
    # sum energies
    E_sum = _np.sum(_np.asarray(total_energy), (1,2))
    KE_sum = _np.sum(_np.asarray(kinetic_energy), (1,2))
    UE_sum = _np.sum(_np.asarray(potential_energy), (1,2))
    UTh_sum = _np.sum(_np.asarray(thermal_energy), (1,2))
    
    # calculate flux
    indicies, actual_angle = calculate_actual_jet_opening_angle(initial_data, theta_deg)
    volume = calculate_cell_volume(initial_data)
    area = calculate_cell_area(initial_data)
    
    thermal_energy_density = thermal_energy / volume
    kinetic_energy_density = kinetic_energy / volume
    potential_energy_density = potential_energy / volume
    
    var_list = ["vx1", "vx2"]
    v = load_simulation_variables(timesteps, run_directory, var_list)
    vx1 = _np.asarray(v["vx1"])
    vx2 = _np.asarray(v["vx2"])
    
    flux = ( (thermal_energy_density + kinetic_energy_density) * _np.sqrt(vx1 ** 2 + vx2 ** 2) * area )
    # array is timestep, r index, theta index
    flux_sum = _np.sum(flux[:, 0, 0:indicies[-1]], (1))
    
    return (E_sum, KE_sum, UE_sum, UTh_sum, flux_sum)

def calculate_actual_jet_opening_angle(run_data, theta_deg):
    indicies = _np.where(run_data.x2 < _np.deg2rad(theta_deg))[0]
    if len(indicies) == 0:
        return (range(0,len(run_data.x2-1)), theta_deg)
    actual_angle = _np.rad2deg(run_data.x2[indicies[-1]])
    return (indicies, actual_angle)

def calculate_theoretical_energy(run_data, theta_deg, run_jet, run_times):
    indicies, actual_angle = calculate_actual_jet_opening_angle(run_data, theta_deg)
    new_run_jet = _jet.AstroJet(actual_angle, run_jet.M_x, run_jet.c_x, run_jet.rho_0, run_jet.Q, run_jet.gamma)
    new_run_jet.calculate_length_scales()
    
    theoretical_energy = (((run_jet.M_x ** 3)*(new_run_jet.Omega*((run_jet.L_1b/run_jet.L_1)**2)/2.0)) 
                    + (new_run_jet.Omega / 2.0) * (9.0/10.0) * ((run_jet.L_1b/run_jet.L_1) ** 2) * run_jet.M_x) * _np.asarray(run_times)
    return theoretical_energy

def calculate_energy_multiple_timesteps(run_data, gamma=5.0/3.0, initial_data=None):
    """Calculates the energy components for each timestep in run_data"""
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
        dE, dKE, dUE, dUth = calculate_energy(energy_data, initial_data, gamma, volume)
        
        # total energy
        total_energy.append(dE)
        kinetic_energy.append(dKE)
        potential_energy.append(dUE)
        thermal_energy.append(dUth)
        
    return (total_energy, kinetic_energy, potential_energy, thermal_energy)
