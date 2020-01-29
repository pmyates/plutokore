from __future__ import absolute_import
import astropy.units as _u
from . import simulations as _ps
import numpy as _np
import scipy as _scipy
from scipy import special as _sp
from astropy import constants as _const
from astropy.cosmology import Planck15 as _cosmo
from astropy.convolution import convolve as _convolve
from astropy.convolution import Box2DKernel as _Box2DKernel
from astropy.convolution import Gaussian2DKernel as _Gaussian2DKernel
from scipy.integrate import trapz as _trapz


def get_A(q):
    return (_sp.gamma(q / 4.0 + 19.0 / 12.0) *
            _sp.gamma(q / 4.0 - 1.0 / 12.0) *
            _sp.gamma(q / 4.0 + 5.0 / 4.0)) / (_sp.gamma(q / 4.0 + 7.0 / 4.0))


def get_K(q, gamma_c):
    return get_A(q) * (1.0 / ((gamma_c - 1)**(
        (q + 5.0) / 4.0))) * (2.0 * _np.pi / 3.0)**(-(q - 1) / 2.0) * (
            (_np.sqrt(3.0 * _np.pi)) / ((16.0 * (_np.pi**2.0)) * (q + 1.0)))


def get_I(q, gamma_min=10, gamma_max=1e5):
    return ((_const.m_e * (_const.c**2.0))**(2 - q)) * ((
        (gamma_max**(2.0 - q)) - (gamma_min**(2.0 - q))) / (2.0 - q))


def get_L0(q,
           gamma_c,
           eta=0.1,
           gamma_min=10,
           gamma_max=1e5,
           freq=1 * _u.GHz,
           prs=1e-11 * _u.Pa,
           vol=1 * _u.kpc**3):
    log_inverse_I = -1.0 * _np.log10(get_I(q, gamma_min, gamma_max).si.value)
    log_eta_term = ((q + 1) / 4.0) * _np.log10(eta) + (-(q + 5.0) / 4.0
                                                       ) * _np.log10(1.0 + eta)
    log_mu0_term = ((q + 1) / 4.0) * _np.log10(2 * _const.mu0.si.value)
    log_nu_term = (-(q - 1) / 2.0) * _np.log10(
        freq.si.value * (_const.m_e.si.value**3.0) * (_const.c.si.value**4.0) *
        (1.0 / _const.e.si.value))
    log_const_term = _np.log10(
        get_K(q, gamma_c) *
        ((_const.e.si.value**3.0) /
         (_const.eps0.si.value * _const.c.si.value * _const.m_e.si.value)))
    log_p0_term = ((q + 5.0) / 4.0) * _np.log10(prs.si.value)
    L0 = 4 * _np.pi * vol.si.value * 10**(
        log_const_term + log_nu_term + log_mu0_term + log_inverse_I +
        log_eta_term + log_p0_term)
    return L0 * (_u.W / _u.Hz)


def combine_tracers(simulation_data, ntracers):
    """Helper function to combine multiple tracers into one array. Simply adds them up"""
    ret = _np.zeros_like(simulation_data.tr1)
    for i in range(ntracers):
        ret = ret + getattr(simulation_data, 'tr{0}'.format(i + 1))
    return ret


def clamp_tracers(simulation_data,
                  ntracers,
                  tracer_threshold=1e-7,
                  tracer_effective_zero=1e-10):

    return clamp_tracers_internal(
        combine_tracers(simulation_data, ntracers),
        tracer_threshold=tracer_threshold,
        tracer_effective_zero=tracer_effective_zero)


def clamp_tracers_internal(tracers,
                           tracer_threshold=1e-7,
                           tracer_effective_zero=1e-10):

    # smooth the tracer data with a 2d box kernel of width 3
    #box2d = _Box2DKernel(3)
    #radio_combined_tracers = _convolve(tracers, box2d, boundary='extend')
    radio_combined_tracers = tracers
    radio_tracer_mask = _np.where(radio_combined_tracers > tracer_threshold,
                                  1.0, tracer_effective_zero)

    # create new tracer array that is clamped to tracer values
    clamped_tracers = radio_combined_tracers.copy()
    clamped_tracers[clamped_tracers <=
                    tracer_threshold] = tracer_effective_zero

    return (radio_tracer_mask, clamped_tracers, radio_combined_tracers)


def get_luminosity_old(simulation_data,
                       unit_density,
                       unit_length,
                       unit_time,
                       redshift,
                       beam_FWHM_arcsec,
                       ntracers,
                       q=2.2,
                       gamma_c=4.0 / 3.0,
                       eta=0.1,
                       freq=1.4 * _u.GHz,
                       prs_scale=1e-11 * _u.Pa,
                       vol_scale=(1 * _u.kpc)**3,
                       alpha=0.6,
                       tracer_threshold=1.0e-7,
                       tracer_effective_zero=1e-10,
                       radio_cell_volumes=None,
                       radio_cell_areas=None,
                       L0=None,
                       calculate_luminosity=True,
                       convolve_flux=False):

    # units
    unit_mass = (unit_density * (unit_length**3)).to(_u.kg)
    unit_pressure = unit_mass / (unit_length * unit_time**2)

    # distance information and conversions
    Dlumin = _cosmo.luminosity_distance(redshift)
    kpc_per_arcsec = _cosmo.kpc_proper_per_arcmin(redshift).to(_u.kpc /
                                                               _u.arcsec)

    # simulation data
    if radio_cell_volumes is None:
        radio_cell_volumes = _ps.calculate_cell_volume(simulation_data)
    if radio_cell_areas is None:
        radio_cell_areas = _ps.calculate_cell_area(simulation_data)

    # in physical units
    radio_cell_areas_physical = radio_cell_areas * unit_length**2
    radio_cell_volumes_physical = radio_cell_volumes * unit_length**3

    # pressure in physical units
    radio_prs_scaled = simulation_data.prs * unit_pressure

    # luminosity scaling
    if L0 is None:
        L0 = get_L0(q, gamma_c, eta, freq=freq, prs=prs_scale, vol=vol_scale)

    # beam information
    sigma_beam_arcsec = beam_FWHM_arcsec / 2.355
    area_beam_kpc2 = (_np.pi * (sigma_beam_arcsec * kpc_per_arcsec)
                      **2).to(_u.kpc**2)

    # n beams per cell
    n_beams_per_cell = (radio_cell_areas_physical / area_beam_kpc2).si

    (radio_tracer_mask, clamped_tracers,
     radio_combined_tracers) = clamp_tracers(
         simulation_data, ntracers, tracer_threshold, tracer_effective_zero)

    radio_luminosity_tracer_weighted = None
    if calculate_luminosity is True:
        radio_luminosity = (L0 * (radio_prs_scaled / prs_scale)**(
            (q + 5.0) / 4.0) * radio_cell_volumes_physical /
                            vol_scale).to(_u.W / _u.Hz)
        radio_luminosity_tracer_weighted = radio_luminosity * radio_tracer_mask * clamped_tracers

    flux_const_term = (L0 / (4 * _np.pi * (Dlumin**2))) * ((1 + redshift)
                                                           **(1 + alpha))
    flux_prs_term = (radio_prs_scaled / prs_scale)**((q + 5.0) / 4.0)
    flux_vol_term = radio_cell_volumes_physical / vol_scale
    flux_beam_term = 1 / n_beams_per_cell
    flux_density = (flux_const_term * flux_prs_term * flux_vol_term *
                    flux_beam_term).to(_u.Jy)
    flux_density_tracer_weighted = flux_density * radio_tracer_mask * clamped_tracers

    if convolve_flux is True:
        beam_kernel = _Gaussian2DKernel((sigma_beam_arcsec * kpc_per_arcsec
                                         ).to(_u.kpc).value)
        flux_density_tracer_weighted = _convolve(
            flux_density_tracer_weighted.to(_u.Jy),
            beam_kernel,
            boundary='extend') * _u.Jy

    return (radio_luminosity_tracer_weighted, flux_density_tracer_weighted)


def get_luminosity(
        simulation_data,
        unit_values,
        redshift,
        beam_FWHM_arcsec,
        q=2.2,
        gamma_c=4.0 / 3.0,
        eta=0.1,
        freq=1.4 * _u.GHz,
        prs_scale=1e-11 * _u.Pa,
        vol_scale=(1 * _u.kpc)**3,
        tracer_threshold=1.0e-7,
        tracer_effective_zero=1e-10,
        radio_cell_volumes=None,
        tracer_mask=None,):
    """Calculates the radio luminosity of the given simulation data,
    for the specified unit values, redshift, beam information,
    observing frequency and departure from equipartition factor"""

    # distance information and conversions
    Dlumin = _cosmo.luminosity_distance(redshift)
    kpc_per_arcsec = _cosmo.kpc_proper_per_arcmin(redshift).to(_u.kpc /
                                                               _u.arcsec)

    # simulation data
    if radio_cell_volumes is None:
        radio_cell_volumes = _ps.calculate_cell_volume(simulation_data)

    # in physical units
    radio_cell_volumes_physical = radio_cell_volumes * unit_values.length**3

    # pressure in physical units
    radio_prs_scaled = simulation_data.prs * unit_values.pressure

    # luminosity scaling
    L0 = get_L0(q, gamma_c, eta, freq=freq, prs=prs_scale, vol=vol_scale)

    # beam information
    sigma_beam_arcsec = beam_FWHM_arcsec / 2.355
    area_beam_kpc2 = (_np.pi * (sigma_beam_arcsec * kpc_per_arcsec)
                      **2).to(_u.kpc**2)

    ntracers = _ps.get_tracer_count_data(simulation_data)

    (radio_tracer_mask, clamped_tracers,
     radio_combined_tracers) = clamp_tracers(
         simulation_data, ntracers, tracer_threshold, tracer_effective_zero)

    radio_luminosity = (L0 * (radio_prs_scaled / prs_scale)**(
        (q + 5.0) / 4.0) * radio_cell_volumes_physical /
                        vol_scale).to(_u.W / _u.Hz)

    if tracer_mask is None:
        tracer_mask = radio_tracer_mask * clamped_tracers

    return radio_luminosity * tracer_mask

def get_flux_density(radio_luminosity, redshift, alpha=0.6):
    """Calculates the flux density from a given radio luminosity"""

    # distance information and conversions
    Dlumin = _cosmo.luminosity_distance(redshift)

    flux_density = ((radio_luminosity / (4 * _np.pi * (Dlumin**2))) * (
        (1 + redshift)**(1 + alpha))).to(_u.Jy)

    return flux_density


def get_convolved_flux_density(flux_density, redshift, beam_FWHM_arcsec):

    kpc_per_arcsec = _cosmo.kpc_proper_per_arcmin(redshift).to(_u.kpc /
                                                               _u.arcsec)

    # beam information
    sigma_beam_arcsec = beam_FWHM_arcsec / 2.355
    area_beam_kpc2 = (_np.pi * (sigma_beam_arcsec * kpc_per_arcsec)
                      **2).to(_u.kpc**2)

    beam_kernel = _Gaussian2DKernel((sigma_beam_arcsec * kpc_per_arcsec).to(
        _u.kpc).value)

    flux_density = _convolve(
        flux_density.to(_u.Jy), beam_kernel, boundary='extend') * _u.Jy

    return flux_density


def get_surface_brightness(flux_density, simulation_data, unit_values,
                           redshift, beam_FWHM_arcsec):
    """Calculates the surface brightness from a given flux density"""

    kpc_per_arcsec = _cosmo.kpc_proper_per_arcmin(redshift).to(_u.kpc /
                                                               _u.arcsec)
    # beam information
    sigma_beam_arcsec = beam_FWHM_arcsec / 2.355
    area_beam_kpc2 = (_np.pi * (sigma_beam_arcsec * kpc_per_arcsec)
                      **2).to(_u.kpc**2)

    radio_cell_areas = _ps.calculate_cell_area(simulation_data)

    # in physical units
    radio_cell_areas_physical = radio_cell_areas * unit_values.length**2

    # n beams per cell
    n_beams_per_cell = (radio_cell_areas_physical / area_beam_kpc2).si

    return flux_density / n_beams_per_cell

def convolve_surface_brightness(sb, unit_values, redshift, beam_FWHM_arcsec):
    kpc_per_arcsec = _cosmo.kpc_proper_per_arcmin(redshift).to(_u.kpc /
                                                               _u.arcsec)
    # beam information
    sigma_beam_arcsec = beam_FWHM_arcsec / 2.355
    area_beam_kpc2 = (_np.pi * (sigma_beam_arcsec * kpc_per_arcsec)
                      **2).to(_u.kpc**2)
    stddev = ((sigma_beam_arcsec * kpc_per_arcsec) / unit_values.length).si
    beam_kernel = _Gaussian2DKernel(stddev)

    return _convolve(sb.to(_u.Jy), beam_kernel, boundary='extend') * _u.Jy

def calculate_volume_emissivity_coefficient(*, q = 2.2, gamma_c = 4.0/3.0, eta = 0.1, gamma_min = 10, gamma_max = 1e5, freq = 1 * _u.GHz, prs = 1e-11 * _u.Pa):
    '''
    Calculates the volume emissivity coefficient

    Parameters
    ---------
    q : float
        The electron energy power law index (default: 2.2, see Hardcastle & Krause 2013)
    gamma_c : float
        The adiabatic index (default: 4/3 for a relativistic plasma)
    eta : float
        The ratio between energy densities of the magnetic field and electrons (default: 0.1, see Turner, Shabala, & Krause 2018)
    gamma_min : float
        The minimum lorentz factor for the electron energy distribution (default: 10)
    gamma_max : float
        The maximum lorentz factor for the electron energy distribution (default: 10)
    freq : float * u.GHz
        The scale frequency (default: 1 * u.GHz)
    prs : float * u.Pa
        The scale pressure (default: 1e-11 * u.Pa)

    Returns
    ------
    j0 : float * u.W / (u.Hz * u.sr * u.m ** 3)
        The volume emissivity coefficient in SI units
    '''
    log_inverse_I = -1.0 * _np.log10(get_I(q, gamma_min, gamma_max).si.value)
    log_eta_term = ((q + 1) / 4.0) * _np.log10(eta) + (-(q + 5.0) / 4.0
                                                       ) * _np.log10(1.0 + eta)
    log_mu0_term = ((q + 1) / 4.0) * _np.log10(2 * _const.mu0.si.value)
    log_nu_term = (-(q - 1) / 2.0) * _np.log10(
        freq.si.value * (_const.m_e.si.value**3.0) * (_const.c.si.value**4.0) *
        (1.0 / _const.e.si.value))
    log_const_term = _np.log10(
        get_K(q, gamma_c) *
        ((_const.e.si.value**3.0) /
         (_const.eps0.si.value * _const.c.si.value * _const.m_e.si.value)))
    log_p0_term = ((q + 5.0) / 4.0) * _np.log10(prs.si.value)
    j0 = _np.power(10, log_const_term + log_nu_term + log_mu0_term + log_inverse_I + log_eta_term + log_p0_term)
    return j0 * (_u.W / (_u.Hz * (_u.m ** 3) * _u.sr))

def calculate_3d_sb(*,
                    sim_data,
                    gamma_max = 1e5,
                    freq = 1.4 * _u.GHz,
                    q = 2.2,
                    gamma_c = 4.0/3.0,
                    eta = 0.1,
                    gamma_min = 10,
                    tracer_threshold = 1e-6,
                    small_tracer_val = 1e-20,
                    ):
    '''
    Calculates the smoothed, gridded surface brightness

    Parameters
    ---------
    sim_data : HDF5 File
        The simulation data file
    freq : float * u.GHz
        The surface brightness frequency (default: 1.4 * u.GHz)
    q : float
        The electron energy power law index (default: 2.2, see Hardcastle & Krause 2013)
    gamma_c : float
        The adiabatic index (default: 4/3 for a relativistic plasma)
    eta : float
        The ratio between energy densities of the magnetic field and electrons (default: 0.1, see Turner, Shabala, & Krause 2018)
    gamma_min : float
        The minimum lorentz factor for the electron energy distribution (default: 10)
    gamma_max : float
        The maximum lorentz factor for the electron energy distribution (default: 10)
    tracer_threshold : float
        The minimum tracer threshold to classify as jet material
    small_tracer_val : float
        The value assigned to tracers below the threshold

    Returns
    ------
    grid_x : u.arcsec
        X grid in arcseconds
    grid_y : u.arsec
        Y grid in arcseconds
    sb : u.mJy / u.beam
        Surface brightness, in units of mJy per beam
    '''
    # calculate our emissivity coefficient
    p0 = 1e-11 * _u.Pa
    freq0 = 1 * _u.GHz
    
    pressure_exp = (q + 5) / 4
    freq_exp = -(q - 1) / 2
    
    # calculate volume emissivity coefficient
    j0 = calculate_volume_emissivity_coefficient(q = q, prs = p0, freq = freq0)
    # calculate surface brightness coefficient
    B0 = j0 * _np.power(freq / freq0, freq_exp) / (4 * _np.pi * _np.power(p0, pressure_exp))
    # calculate tracer mask
    tracer_mask = _np.where(_np.greater(sim_data.tr1, tracer_threshold), 1, small_tracer_val)
    
    # integrate our pressure
    integrated_pressure_term = _trapz(y = _np.power(_np.multiply(sim_data.prs, tracer_mask), pressure_exp), x = sim_data.my * sim_data.unit_length, axis = 1)
    integrated_pressure_term = integrated_pressure_term * _np.power(sim_data.unit_pressure, pressure_exp)
    
    # multiply by our B0 to get surface brightness
    surf_brightness = B0 * integrated_pressure_term

    return surf_brightness

def regrid_3d_sb(*,
                 sim_data,
                 sb,
                 pixel_size = 1.8 * _u.arcsec,
                 beam_fwhm = 5 * _u.arcsec,
                 z = 0.05,
                 ):
    '''
    Grids & smooths the surface brightness, for a given redshift,
    pixel size, and observing beam information (currently assumes observing
    beam is a circular 2D Gaussian with constant & FWHM)

    Parameters
    ---------
    sim_data : HDF5 File
        The simulation data file
    sb : u.mJy / u.beam
        Surface brightness, in units of mJy per beam
    pixel_size : u.arcsec
        The gridded pixel size, in arcseconds
    beam_fwhm : u.arcsec
        The 2D observing beam Gaussian fwhm, in arcseconds
    z : float
        The redshift this sourced is observed at

    Returns
    -------
    grid_x : u.arcsec
        X grid in arcseconds
    grid_y : u.arsec
        Y grid in arcseconds
    sb : u.mJy / u.beam
        Gridded and smoothed surface brightness, in units of mJy per beam
    '''
    fwhm_to_sigma = 1 / (8 * _np.log(2)) ** 0.5
    beam_sigma = beam_fwhm * fwhm_to_sigma
    omega_beam = 2 * _np.pi * beam_sigma ** 2 # Area for a circular 2D gaussian
    
    z = 0.05
    kpc_per_arcsec = _cosmo.Planck15.kpc_proper_per_arcmin(z)
    
    # Create our grid
    grid_res = pixel_size.to(_u.arcsec)
    x_min = (sim_data.mx[0] * sim_data.unit_length / kpc_per_arcsec).to(_u.arcsec)
    x_max = (sim_data.mx[-1] * sim_data.unit_length / kpc_per_arcsec).to(_u.arcsec)
    y_min = (sim_data.mz[0] * sim_data.unit_length / kpc_per_arcsec).to(_u.arcsec)
    y_max = (sim_data.mz[-1] * sim_data.unit_length / kpc_per_arcsec).to(_u.arcsec)
    
    new_x = _np.arange(x_min.value, x_max.value, grid_res.value) # in arcsec now
    new_y = _np.arange(y_min.value, y_max.value, grid_res.value) # in arcsec now
    grid_x, grid_y = _np.meshgrid(new_x, new_y, indexing = 'xy') # in arcsec
    
    old_x = (sim_data.mx * sim_data.unit_length / kpc_per_arcsec).to(_u.arcsec).value # in arcsec
    old_z = (sim_data.mz * sim_data.unit_length / kpc_per_arcsec).to(_u.arcsec).value # in arcsec
    
    # Regrid data
    # everything is in arcsec
    # save our units first, to add them back after
    sb_units = surf_brightness.unit
    sb_gridded = _scipy.interpolate.interpn(points = (old_z, old_x),
                               values = surf_brightness.value,
                               xi = (grid_y, grid_x),
                               method = 'linear',
                               bounds_error = False,
                               fill_value = 0) * sb_units
    
    # Smooth data
    stddev = beam_sigma / pixel_size
    kernel = _Gaussian2DKernel(x_stddev = stddev.value)
    sb_units = sb_gridded.unit
    sb_smoothed = _convolve(sb_gridded.value, kernel) * sb_units
    
    # Convert from per sr to per beam
    sb_final = sb_smoothed.to(_u.mJy / _u.beam, equivalencies = _u.beam_angular_area(omega_beam))

    return (grid_x * _u.arcsec, grid_y * _u.arcsec, sb_final)

def change_sb_freq(*, sb, old_freq, new_freq, q = 2.2):
    '''
    Scale a surface brightness from one frequency to another

    Parameters
    ----------
    sb : u.mJy / u.beam
        Surface brightness, in units of mJy per beam
    old_freq : u.GHz
        The current frequency of the surface brightness
    new_freq : u.GHz
        The desired new frequency of the surface brightness
    q : float
        The electron energy power law index (default: 2.2, see Hardcastle & Krause 2013)

    Returns
    -------
    sb : u.mJy / u.beam
        The original surface brightness scaled to the new frequency
    '''
    # set up our frequency exponent
    freq_exp = -(q - 1) / 2
    
    # multiply original sb by new_freq / old_freq, to change scaling
    return sb * np.power((new_freq / old_freq).to(u.dimensionless_unscaled), freq_exp)
