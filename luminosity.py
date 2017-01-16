import astropy.units as _u
import simulations as _ps
import numpy as _np
from scipy import special as _sp
from astropy import constants as _const
from astropy.cosmology import Planck15 as _cosmo
from astropy.convolution import convolve as _convolve
from astropy.convolution import Box2DKernel as _Box2DKernel
from astropy.convolution import Gaussian2DKernel as _Gaussian2DKernel


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
    box2d = _Box2DKernel(3)
    radio_combined_tracers = _convolve(tracers, box2d, boundary='extend')
    radio_tracer_mask = _np.where(radio_combined_tracers > tracer_threshold,
                                  1.0, tracer_effective_zero)

    # create new tracer array that is clamped to tracer values
    clamped_tracers = radio_combined_tracers.copy()
    clamped_tracers[clamped_tracers <=
                    tracer_threshold] = tracer_effective_zero

    return (radio_tracer_mask, clamped_tracers, radio_combined_tracers)


def get_luminosity(simulation_data,
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


def get_luminosity_new(
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
        radio_cell_areas=None,):

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
    radio_cell_areas_physical = radio_cell_areas * unit_values.length**2
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
    radio_luminosity_tracer_weighted = radio_luminosity * radio_tracer_mask * clamped_tracers

    return radio_luminosity_tracer_weighted


def get_flux_density(radio_luminosity, redshift, alpha=0.6):

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


def get_surface_brightness(flux_density, simulation_data, unit_values, redshift, beam_FWHM_arcsec):

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
