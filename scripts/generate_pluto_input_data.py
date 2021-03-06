#!/usr/bin/env python3

# system
import os
import sys

# utilities
from numba import jit
from pathlib import Path
import h5py
from IPython.display import display,clear_output
from datetime import date

# science imports
import numpy as np
import scipy.interpolate
import scipy.integrate
from scipy.integrate import trapz

# matplotlib imports
import matplotlib as mpl
import matplotlib.pyplot as plot

# astropy imports
from astropy.table import Table
from astropy import units as u # Astropy units
from astropy import cosmology as cosmo # Astropy cosmology
from astropy import constants as const # Astropy constants
from astropy.convolution import convolve, Gaussian2DKernel # Astropy convolutions

# plutokore
if os.path.exists(os.path.expanduser('~/plutokore')):
    sys.path.append(os.path.expanduser('~/plutokore'))
else:
    sys.path.append(os.path.expanduser('~/uni/plutokore'))
import plutokore as pk
import plutokore.radio as radio
from plutokore.jet import UnitValues

unit_length = 1 * u.kpc
unit_density = (0.60364 * u.u / (u.cm ** 3)).to(u.g / u.cm ** 3)
unit_speed = const.c
unit_time = (unit_length / unit_speed).to(u.Myr)
unit_pressure = (unit_density * (unit_speed ** 2)).to(u.Pa)
unit_mass = (unit_density * (unit_length ** 3)).to(u.kg)
unit_energy = (unit_mass * (unit_length**2) / (unit_time**2)).to(u.J)

uv = UnitValues(
    density=unit_density,
    length=unit_length,
    time=unit_time,
    mass=unit_mass,
    pressure=unit_pressure,
    energy=unit_energy,
    speed=unit_speed,
)

def write_pluto_grid_information(*, grid_fname, dimensions, geometry, grid_list, extra_info = None):
    """
    Writes out a PLUTO grid file in the same format as grid.out, for the given grid information

    Parameters
    ----------
    grid_fname : str
        The grid filename
    dimensions : int
        The number of grid dimensions [1, 2, or 3]
    geometry : str
        The grid geometry [CARTESIAN, SPHERICAL, CYLINDRICAL, or POLAR]
    grid_list : List[1D numpy.ndarray]
        A list of n numpy arrays for an n-dimensional grid, where each numpy array contains the cell edges in that coordinate
    extra_info : List[str]
        A list of extra information that will be printed to the grid.out file
    """
    # check arguments
    if dimensions not in [1, 2, 3]:
        raise Exception('Invalid dimensions')
    if geometry not in ['CARTESIAN', 'SPHERICAL', 'CYLINDRICAL', 'POLAR']:
        raise Exception('Invalid geometry')

    # Write out our grid file
    with open(grid_fname, 'w') as f:
        # Write header info
        f.write('# ' + '*'*50 + '\n')
        f.write('# PLUTO 4.3 Grid File\n')
        f.write(f'# Manually generated on {date.today()}\n')
        f.write('#\n')
        f.write('# Info:\n')
        f.write('# Input data generated by python\n')
        f.write(f'# Endianess: {sys.byteorder}\n')
        if extra_info is not None:
            f.writelines([f'# {line}\n' for line in extra_info])
        f.write('#\n')
        f.write(f'# DIMENSIONS: {dimensions}\n')
        f.write(f'# GEOMETRY: {geometry}\n')
        for dim in range(dimensions):
            f.write(f'# X{dim+1}: [ {grid_list[dim][0]}, {grid_list[dim][-1]}], {grid_list[dim].shape[0] - 1} point(s), 0 ghosts\n')
        f.write('# ' + '*'*50 + '\n')
        
        # Write out our grid points
        for dim in range(3):
            if (dim < dimensions):
                f.write(f'{grid_list[dim].shape[0] - 1}\n')
                for pn in np.arange(grid_list[dim].shape[0]-1) + 1:
                    f.write(f'\t{pn}\t{grid_list[dim][pn-1]:.12e}\t{grid_list[dim][pn]:.12e}\n')
            else:
                f.write(f'{1}\n')
                f.write(f'\t{1}\t{0:.12e}\t{1:.12e}\n')
    pass

def write_pluto_data(*, data_fname, data_list):
    """
    Writes out a PLUTO data file in the same format as *.dbl, for the given variables

    Parameters
    ----------
    data_fname : str
        The data filename (must end in .dbl)
    data_list : List[n-dimensional numpy.ndarray]
        A list of numpy arrays (one per variable) to be written out to the data file.
        Each numpy array should have the same dimensions as the grid, and contain the cell-centered values
        of that variable.
    """
    with open(data_fname, 'w') as f:
        for vdata in data_list:
            vdata.astype(np.double).flatten(order = 'F').tofile(f)

def write_pluto_initial_conditions(*, grid_fname, data_fname, grid_list, data_dict, dimensions, geometry, extra_info = None):
    """
    Writes out PLUTO grid and data files to be used as intial conditions

    Parameters
    ----------
    grid_fname : str
        The grid filename
    data_fname : str
        The data filename (must end in .dbl)
    grid_list : List[1D numpy.ndarray]
        A list of n numpy arrays for an n-dimensional grid, where each numpy array contains the cell edges in that coordinate
    data_dict : Dictionary{str, n-dimensional numpy.ndarray}
        A dictionary of numpy variable arrays.
        Each variable should have it's own entry, e.g. {'rho' : rho_numpy_array }, where the variable array
        should have the same dimensions as the grid, and contain the cell-centered values of that variable
    dimensions : int
        The number of grid dimensions [1, 2, or 3]
    geometry : str
        The grid geometry [CARTESIAN, SPHERICAL, CYLINDRICAL, or POLAR]
    extra_info : List[str]
        A list of extra information that will be printed to the grid.out file
    """
    if extra_info is None:
        extra_info = []
    write_pluto_grid_information(grid_fname = grid_fname,
                                 dimensions = dimensions,
                                 geometry = geometry,
                                 grid_list = grid_list,
                                 extra_info = extra_info + ['Variables in dbl file:'] + list(data_dict.keys()))
    write_pluto_data(data_fname = data_fname,
                     data_list = list(data_dict.values()))

def main():
    # first we generate our data

    # for testing, I've created a grid ranging from -100 to 100 in each direction, with random edge sampling
    # note that the ex, ey, ez coordinate arrays are EDGES
    nx, ny, nz = (101, 51, 201)
    ex = np.sort(np.random.uniform(low = -55, high = 55, size = nx))
    ey = np.sort(np.random.uniform(low = -55, high = 55, size = ny))
    ez = np.sort(np.random.uniform(low = -55, high = 55, size = nz))

    # now we create our midpoint arrays
    mx = (np.diff(ex)*0.5) + ex[:-1]
    my = (np.diff(ey)*0.5) + ey[:-1]
    mz = (np.diff(ez)*0.5) + ez[:-1]

    print(mx.shape)

    # let's generate some sort of density data

    # first we create our meshgrid
    # note the 'ij' indexing - THIS IS IMPORTANT. Need to use this if working in 3D
    mesh_x,mesh_y,mesh_z = np.meshgrid(mx, my, mz, indexing = 'ij')

    # create our radius array
    r = np.sqrt(mesh_x**2 + mesh_y**2 + mesh_z**2)

    # create our density array
    rho = 1 * np.power(1 + (r/144), -3/2 * 0.5)

    # for fun, let's also create a vx3 velocity array that is proportional to the current radius
    vx3 = r * 1e-5

    # now we save our data
    write_pluto_initial_conditions(grid_fname = 'simple-3D.grid.out',
                                   data_fname = 'simple-3D.dbl',
                                   grid_list = [ex, ey, ez],
                                   data_dict = {'rho': rho, 'vx3': vx3},
                                   dimensions = 3,
                                   geometry = 'CARTESIAN')

    f,a = plot.subplots()
    a.pcolormesh(mx, mz, rho[:,0,:].T)
    plot.show()

if __name__ == "__main__":
    main()
