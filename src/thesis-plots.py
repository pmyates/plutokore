
# coding: utf-8

# In[1]:

import matplotlib as mpl
mpl.use('PGF')
import matplotlib.pyplot as plt
plt.style.use(['seaborn-deep', './honours.mplstyle', './thesis.mplstyle'])

# init
get_ipython().magic(u'run Shared_Code.ipynb')

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

import jet as jet
jet = reload(jet)
import NFW as NFW
NFW = reload(NFW)
import pluto_simulations as ps
ps = reload(ps)
import plotting as plotting
plotting = reload(plotting)


# In[21]:

base_dir = 'E:/honours-project/runs/t15/'
base_dir_old = 'D:/Patrick/Documents/Uni/Honours/Pluto Runs/Jet/'

relative_standard_dirs = np.array([
    'm14.5/M25/n1/',
    'm14.5/M25/n2/',
    'm14.5/M25/n3/',
    'm14.5/M25/v2/n4/',
    'm12.5/M25/n1/',
    'm12.5/M25/n2/',
    'm12.5/M25/n3/',
    'm12.5/M25/v2/n4/',
    'env-tracer/m14.5/M25/n1/',
    'env-tracer/m14.5/M25/n4/',
    'env-tracer/m12.5/M25/n1/',
    'env-tracer/m12.5/M25/n4/',
    'm14.5/M35/n1/',
    'm14.5/M35/n4/',
    'm12.5/M25/fast-ramp/n1/',
    'm12.5/M25/fast-ramp/n4/',
    'm12.5/M25/close-injection/n1/',
    'm12.5/M25/close-injection/n4/',
    'king-fixed/m14.5/M25/n1/',
    'king-fixed/m14.5/M25/n4/',
    'king-fixed/m12.5/M25/n1/',
    'king-fixed/m12.5/M25/n4/',
    'm14.5/M25/high-res/n2/',
    'm14.5/M25/high-res/n3/',
])

relative_old_dirs = np.array([
        '01/',
        '02/',
        '03/',
        '04/',
        '05/',
        '06/',
        '07/',
        '08/',
    ])

run_dirs = np.array([os.path.join(base_dir, p) for p in relative_standard_dirs])
old_run_dirs = np.array([os.path.join(base_dir_old, p) for p in relative_old_dirs])

run_codes = np.array([
    'm14.5-M25-n1',
    'm14.5-M25-n2',
    'm14.5-M25-n3',
    'm14.5-M25-n4',
    'm12.5-M25-n1',
    'm12.5-M25-n2',
    'm12.5-M25-n3',
    'm12.5-M25-n4',
    'm14.5-M25-n1-env-tracer',
    'm14.5-M25-n4-env-tracer',
    'm12.5-M25-n1-env-tracer',
    'm12.5-M25-n4-env-tracer',
    'm14.5-M35-n1',
    'm14.5-M35-n4',
    'm12.5-M25-n1-fast-ramp',
    'm12.5-M25-n4-fast-ramp',
    'm12.5-M25-n1-close-injection',
    'm12.5-M25-n4-close-injection',
    'm14.5-M25-n1-king',
    'm14.5-M25-n4-king',
    'm12.5-M25-n1-king',
    'm12.5-M25-n4-king',
    'm14.5-M25-n2-high-res',
    'm14.5-M25-n3-high-res',
])

old_run_codes = np.array([
        'old-01',
        'old-02',
        'old-03',
        'old-04',
        'old-05',
        'old-06',
        'old-07',
        'old-08',
    ])

standard_14p5 = range(0,4)
standard_12p5 = range(4,8)
standard_all = np.concatenate((standard_14p5, standard_12p5))

env_tracer_14p5 = range(8,10)
env_tracer_12p5 = range(10,12)
env_tracer_all = np.concatenate((env_tracer_14p5, env_tracer_12p5))

higher_mach = range(12,14)

fast_ramp = range(14,16)

close_injection = range(16,18)

king_runs_14p5 = range(18,20)
king_runs_12p5 = range(20,22)
king_runs_all = np.concatenate((king_runs_14p5, king_runs_12p5))

high_res = range(22,24)

resolution_test = [1,2,0,3]


# In[12]:

# test
h = ['Index', 'Run Code', 'Run Directory']
print('All standard runs')
print(tabulate(zip(standard_all, run_codes[standard_all], run_dirs[standard_all]), headers=h))
print('')
print('All environment tracer runs')
print(tabulate(zip(env_tracer_all, run_codes[env_tracer_all], run_dirs[env_tracer_all]), headers=h))
print('')
print('Higher mach runs')
print(tabulate(zip(higher_mach, run_codes[higher_mach], run_dirs[higher_mach]), headers=h))
print('')
print('Fast ramp')
print(tabulate(zip(fast_ramp, run_codes[fast_ramp], run_dirs[fast_ramp]), headers=h))
print('')
print('Close injection')
print(tabulate(zip(close_injection, run_codes[close_injection], run_dirs[close_injection]), headers=h))
print('')
print('King profile')
print(tabulate(zip(king_all, run_codes[king_all], run_dirs[king_all]), headers=h))


# In[13]:

import numpy as np
import astropy.units as u
import astropy.constants as const
import astropy.cosmology as cosmology
import scipy.integrate as integrate
from tabulate import tabulate

class king_profile:
    
    def __init__(self, halo_mass, redshift, T=None, delta_vir=None, mu=None, f_gas=None, omega_b=None, gamma=None, concentration_method=None,
                cosmo=None):
        """Creates a King profile for the given halo mass and redshift"""
        
        self.halo_mass = halo_mass
        self.redshift = redshift
        
        if delta_vir is None:
            delta_vir = 200
        self.delta_vir = delta_vir
        
        
        if mu is None:
            mu = 0.6
        self.mu = mu
        
        if f_gas is None:
            f_gas = 1.0
        self.f_gas = f_gas
        
        if cosmo is None:
            cosmo = cosmology.Planck15
        self.cosmo = cosmo
        
        if omega_b is None:
            omega_b = self.cosmo.Ob(self.redshift) / self.cosmo.Om(self.redshift)
        self.omega_b = omega_b
        
        if gamma is None:
            gamma = 5.0 / 3.0
        self.gamma = gamma
        
        if concentration_method is None:
            concentration_method = 'klypin'
            
        self.critical_density = self.calculate_critical_density(self.redshift, cosmo=self.cosmo)
        self.virial_radius = self.calculate_virial_radius(self.halo_mass, self.delta_vir, self.critical_density)
        
        if T is None:
            T = self.calculate_temperature(self.halo_mass, self.mu, self.virial_radius)
        self.T = T
        
        # calculate sound speed
        self.sound_speed = self.calculate_sound_speed(self.gamma, self.T, self.mu)
        
        # set king beta parameter from Vikhlinin+06 cluster observations
        self.beta = 0.38
        
        # take 0.1 R_vir for consistency with Vikhlinin and HK13
        self.core_radius = 0.1 * self.virial_radius
        
        # calculate concentration
        self.concentration = self.calculate_concentration(self.halo_mass, 
                                redshift=self.redshift, 
                                method=concentration_method, 
                                cosmo=self.cosmo)
        
        self.central_density = self.calculate_central_density(self.f_gas, 
                                self.omega_b, self.halo_mass, 
                                self.virial_radius, self.beta, 
                                self.core_radius)
        
    def get_density(self, r):
        return self.calculate_density_at_radius(self.central_density, self.beta, self.core_radius, r)
        
    @staticmethod
    def calculate_temperature(halo_mass, mu, virial_radius, gamma_isothermal=1.5):
        """Calculates the virial temperature using an isothermal approximation from Makino+98"""
        t_vir = (gamma_isothermal/3.0) * (mu * const.m_p) * (const.G * halo_mass) / (virial_radius)
        return t_vir.to(u.keV)
        
    @staticmethod
    def calculate_critical_density(redshift, cosmo=cosmology.Planck15):
        """Calculates the critical density parameter for the given redshift,
        using the Planck (15) survey results for the hubble constant"""
        H_squared = cosmo.H(redshift) ** 2
        critical_density = (3.0 * H_squared) / (8.0 * np.pi * const.G)
        return critical_density.to(u.g / u.cm ** 3)
    
    @staticmethod
    def calculate_virial_radius(virial_mass, delta_vir, critical_density):
        """Calculates the virial radius for the given virial mass, delta_vir and critical density"""
        r_vir = ((3.0 * virial_mass)/(4.0 * np.pi * delta_vir * critical_density)) ** (1.0/3.0)
        return r_vir.to(u.kpc)
    
    @staticmethod
    def calculate_concentration(virial_mass, method = 'bullock', redshift = 0, cosmo=cosmology.Planck15):
        """Calculates the concentration parameter for a given virial mass and redshift"""
        # parameters for Dolag+04, delta_vir is 200 using MEAN density. Cosmology closely matches WMAP9
        c_0_dolag = 9.59
        alpha_dolag = -0.102
        
        # parameters for Klypin+16 Planck, all halos, delta_vir is 200 using critical density
        c_0_klypin_planck_all = 7.40
        gamma_klypin_planck_all = 0.120
        m_0_klypin_planck_all = 5.5e5
        
        # parameters for Klypin+16 Planck, relaxed halos, delta_vir is 200 using critical density
        c_0_klypin_planck_relaxed = 7.75
        gamma_klypin_planck_relaxed = 0.100
        m_0_klypin_planck_relaxed = 4.5e5
        
        # parameters for Klypin WMAP7 all halos, delta_vir is 200 using critical density
        c_0_wmap_all = 6.60
        gam_wmap_all = 0.110
        m_0_wmap_all = 2e6
        
        # parameters for Klypin WMAP7 relaxed halos, delta_vir is 200 using critical density
        c_0_wmap_relaxed = 6.90
        gam_wmap_relaxed = 0.090
        m_0_wmap_relaxed = 5.5e5
        
        # parameters for Dutton+14, NFW model, delta_vir = 200 using critical density, Planck cosmology
        b_dutton = -0.101 + 0.026 * redshift
        a_dutton = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * (redshift ** 1.21))
        
        # parameters for Maccio+08, NFW model, WMAP5, delta_vir = 200 using critical density
        zero_maccio = 0.787
        slope_maccio = -0.110
        
        
        c = 0.0
        
        # Dolag+04 method
        if method == 'dolag':
            c = (c_0_dolag)/(1 + redshift) * ((virial_mass / (10.0 ** 14 * u.M_sun * (1.0/cosmo.h))) ** alpha_dolag)
        # Bullock+01 method, cosmological parameters agree with WMAP9, delta_vir = 180 using 337 using mean density (for LambdaCDM)
        elif method == 'bullock':
            c = (8.0/(1+redshift)) * (10 ** ((-0.13)*(np.log10(virial_mass.to(u.M_sun).value)-14.15)))
        # Klypin+16 methods
        elif method == 'klypin-planck-all':
            c = (c_0_klypin_planck_all * ((virial_mass / (1e12 * u.M_sun * (cosmo.h ** (-1)))) ** -gamma_klypin_planck_all) * 
                 (1 + (virial_mass / (m_0_klypin_planck_all * (1e12 * u.M_sun * (cosmo.h ** (-1))))) ** 0.4))
        elif method == 'klypin-planck-relaxed':
            c = (c_0_klypin_planck_relaxed * ((virial_mass / (1e12 * u.M_sun * (cosmo.h ** (-1)))) ** -gamma_klypin_planck_relaxed) * 
                 (1 + (virial_mass / (m_0_klypin_planck_relaxed * (1e12 * u.M_sun * (cosmo.h ** (-1))))) ** 0.4))
        elif method == 'klypin-wmap-all':
            c = ((c_0_wmap_all * ((virial_mass / (1e12 * u.M_sun * (cosmo.h ** (-1)))) ** -gam_wmap_all)) *
            ((1 + (virial_mass / (m_0_wmap_all * (1e12 * u.M_sun * (cosmo.h ** (-1))))) ** 0.4)))
        elif method == 'klypin-wmap-relaxed':
            c = ((c_0_wmap_relaxed * ((virial_mass / (1e12 * u.M_sun * (cosmo.h ** (-1)))) ** -gam_wmap_relaxed)) *
            ((1 + (virial_mass / (m_0_wmap_relaxed * (1e12 * u.M_sun * (cosmo.h ** (-1))))) ** 0.4)))
        # Dutton+14 method
        elif method == 'dutton':
            logc = a_dutton + b_dutton * np.log10(virial_mass / (1e12 * (cosmo.h ** -1) * u.M_sun))
            c = 10 ** logc
        # Maccio+08 method
        elif method == 'maccio':
            logc = zero_maccio + slope_maccio * (np.log10(virial_mass / (u.M_sun * (1.0/cosmo.h))) - 12)
            c = 10 ** logc
        return c
    
    @staticmethod
    def calculate_central_density(f_gas, omega_b, virial_mass, virial_radius, beta, core_radius):
        """Calculates the central density for the NFW profile, given the hot gas fraction, baryonic matter percentage and
        virial mass of the halo"""
        func = lambda r: 4*np.pi*(r**2) * np.power(1 + ((r / core_radius.value) ** 2), -3.0 * beta / 2.0)
        integral = (u.kpc ** 3) * integrate.quad(func, 0, virial_radius.value)
        denom = integral[0]
        rho_0 = (f_gas * omega_b * virial_mass) / denom
        return rho_0.to(u.g * u.cm ** (-3))
    
    @staticmethod
    def calculate_density_at_radius(central_density, beta, core_radius, r):
        return central_density * np.power(1 + ((r / core_radius) ** 2), -3.0 * beta / 2.0)
        #return central_density * np.exp(-delta_nfw)*(1+r/r_s) ** (delta_nfw/(r/r_s))
    
    @staticmethod
    def calculate_sound_speed(gamma, T, mu):
        return np.sqrt((gamma * T)/(mu * const.m_p)).to(u.km / u.s)


# In[14]:

# 14.5 mass halo
# environment parameters
M_vir = (10 ** 14.5) * u.M_sun
z = 0

king_14p5 = king_profile(M_vir, z, delta_vir=200, cosmo=cosmology.Planck15, concentration_method='klypin-planck-relaxed')
makino_14p5 = NFW.NFW_Profile(M_vir, z, delta_vir=200, cosmo=cosmology.Planck15, concentration_method='klypin-planck-relaxed')

# 12.5 mass halo
# environment parameters
M_vir = (10 ** 12.5) * u.M_sun
z = 0

king_12p5 = king_profile(M_vir, z, delta_vir=200, cosmo=cosmology.Planck15, concentration_method='klypin-planck-relaxed')
makino_12p5 = NFW.NFW_Profile(M_vir, z, delta_vir=200, cosmo=cosmology.Planck15, concentration_method='klypin-planck-relaxed')

# jet parameters
theta_deg = 15
M_x = 25
Q = 1e37 * u.W

jet_king_12p5_m25 = jet.Jet(theta_deg, M_x, king_12p5.sound_speed, king_12p5.central_density, Q, king_12p5.gamma)
jet_king_12p5_m25.calculate_length_scales()

jet_king_14p5_m25 = jet.Jet(theta_deg, M_x, king_14p5.sound_speed, king_14p5.central_density, Q, king_14p5.gamma)
jet_king_14p5_m25.calculate_length_scales()

jet_makino_12p5_m25 = jet.Jet(theta_deg, M_x, makino_12p5.sound_speed, makino_12p5.central_density, Q, makino_12p5.gamma)
jet_makino_12p5_m25.calculate_length_scales()

jet_makino_14p5_m25 = jet.Jet(theta_deg, M_x, makino_14p5.sound_speed, makino_14p5.central_density, Q, makino_14p5.gamma)
jet_makino_14p5_m25.calculate_length_scales()

simulation_total_length = 400 * u.kpc
jet_total_active = 40 * u.Myr
sim_total_time = jet_total_active * 5


# In[15]:

from scipy import special as sp
from astropy.cosmology import Planck15 as cosmo
from astropy.convolution import convolve, Box2DKernel, Gaussian2DKernel

def get_A(q):
    return (sp.gamma(q/4.0 + 19.0/12.0) * sp.gamma(q/4.0 - 1.0/12.0) * sp.gamma(q/4.0 + 5.0/4.0)) / (sp.gamma(q/4.0 + 7.0/4.0))

def get_K(q, gamma_c):
    return get_A(q) * (1.0 / ((gamma_c - 1) ** ((q + 5.0)/4.0))) * (2.0 * np.pi / 3.0) ** (-(q - 1) / 2.0) * ((np.sqrt(3.0 * np.pi)) / ((16.0 * (np.pi ** 2.0)) * (q + 1.0)))

def get_I(q, gamma_min = 10, gamma_max = 1e5):
    return ((const.m_e * (const.c ** 2.0)) ** (2 - q)) * (((gamma_max ** (2.0 - q)) - (gamma_min ** (2.0 - q))) / (2.0 - q))

def get_L0(q, gamma_c, eta=0.1, gamma_min = 10, gamma_max = 1e5, freq = 1 * u.GHz, prs = 1e-11 * u.Pa, vol = 1 * u.kpc ** 3):
    log_inverse_I = -1.0 * np.log10(get_I(q, gamma_min, gamma_max).si.value)
    log_eta_term = ((q+1) / 4.0) * np.log10(eta) + (-(q + 5.0) / 4.0) * np.log10(1.0 + eta)
    log_mu0_term = ((q + 1) / 4.0) * np.log10(2 * const.mu0.si.value)
    log_nu_term = (-(q - 1) / 2.0) * np.log10(freq.si.value * (const.m_e.si.value ** 3.0) * (const.c.si.value ** 4.0) * (1.0 / const.e.si.value))
    log_const_term = np.log10(get_K(q, gamma_c) * ((const.e.si.value ** 3.0) / (const.eps0.si.value * const.c.si.value * const.m_e.si.value)))
    log_p0_term = ((q + 5.0) / 4.0) * np.log10(prs.si.value)
    L0 = 4 * np.pi * vol.si.value * 10 ** (log_const_term + log_nu_term + log_mu0_term + log_inverse_I + log_eta_term + log_p0_term)
    return L0 * (u.W / u.Hz)

def combine_tracers(simulation_data, ntracers):
    """Helper function to combine multiple tracers into one array. Simply adds them up"""
    ret = np.zeros_like(simulation_data.tr1)
    for i in range(ntracers):
        ret = ret + getattr(simulation_data, 'tr{0}'.format(i+1))
    return ret

def clamp_tracers(simulation_data, ntracers,
                  tracer_threshold = 1e-7,
                  tracer_effective_zero = 1e-10):
    
    return clamp_tracers_internal(combine_tracers(simulation_data, ntracers), tracer_threshold=tracer_threshold, 
                                  tracer_effective_zero=tracer_effective_zero)

def clamp_tracers_internal(tracers,
                  tracer_threshold = 1e-7,
                  tracer_effective_zero = 1e-10):
    
    # smooth the tracer data with a 2d box kernel of width 3
    box2d = Box2DKernel(3)
    radio_combined_tracers = convolve(tracers, box2d, boundary='extend')
    radio_tracer_mask = np.where(radio_combined_tracers > tracer_threshold, 1.0, tracer_effective_zero)

    # create new tracer array that is clamped to tracer values
    clamped_tracers = radio_combined_tracers.copy()
    clamped_tracers[clamped_tracers <= tracer_threshold] = tracer_effective_zero
    
    return (radio_tracer_mask, clamped_tracers, radio_combined_tracers)

def get_luminosity(simulation_data, 
                   unit_density, unit_length, unit_time,
                   redshift,
                   beam_FWHM_arcsec,
                   ntracers,
                   q=2.2,
                   gamma_c=4.0/3.0,
                   eta=0.1,
                   freq=1.4 * u.GHz,
                   prs_scale=1e-11 * u.Pa,
                   vol_scale=(1*u.kpc) ** 3,
                   alpha=0.6,
                   tracer_threshold=1.0e-7,
                   tracer_effective_zero=1e-10,
                   radio_cell_volumes = None,
                   radio_cell_areas = None,
                   L0 = None,
                   calculate_luminosity = True,
                   convolve_flux = False):
    
    # units
    unit_mass = (unit_density * (unit_length ** 3)).to(u.kg)
    unit_pressure = unit_mass / (unit_length * unit_time ** 2)
    
    # distance information and conversions
    Dlumin = cosmo.luminosity_distance(redshift)
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(redshift).to(u.kpc / u.arcsec)
    
    # simulation data
    if radio_cell_volumes is None:
        radio_cell_volumes = ps.calculate_cell_volume(simulation_data)
    if radio_cell_areas is None:
        radio_cell_areas = ps.calculate_cell_area(simulation_data)
    
    # in physical units
    radio_cell_areas_physical = radio_cell_areas * unit_length ** 2
    radio_cell_volumes_physical = radio_cell_volumes * unit_length ** 3
    
    # pressure in physical units
    radio_prs_scaled = simulation_data.prs * unit_pressure

    # luminosity scaling
    if L0 is None:
        L0 = get_L0(q, gamma_c, eta, freq=freq, prs=prs_scale, vol=vol_scale)
    
    # beam information
    sigma_beam_arcsec = beam_FWHM_arcsec / 2.355
    area_beam_kpc2 = (np.pi * (sigma_beam_arcsec* kpc_per_arcsec) ** 2).to(u.kpc ** 2)
    
    # n beams per cell
    n_beams_per_cell = (radio_cell_areas_physical / area_beam_kpc2).si
    
    (radio_tracer_mask, clamped_tracers, radio_combined_tracers) = clamp_tracers(simulation_data, 
                                                                                 ntracers, 
                                                                                 tracer_threshold,
                                                                                 tracer_effective_zero)
    
    radio_luminosity_tracer_weighted = None
    if calculate_luminosity is True:
        radio_luminosity = (L0 * (radio_prs_scaled / prs_scale) ** ((q + 5.0) / 4.0) * radio_cell_volumes_physical / vol_scale).to(u.W / u.Hz)
        radio_luminosity_tracer_weighted = radio_luminosity * radio_tracer_mask * clamped_tracers
    
    flux_const_term = (L0 / (4 * np.pi * (Dlumin ** 2))) * ((1+redshift) ** (1+alpha))
    flux_prs_term = (radio_prs_scaled / prs_scale) ** ((q + 5.0) / 4.0)
    flux_vol_term = radio_cell_volumes_physical / vol_scale
    flux_beam_term = 1 / n_beams_per_cell
    flux_density = (flux_const_term * flux_prs_term * flux_vol_term * flux_beam_term).to(u.Jy)
    flux_density_tracer_weighted = flux_density * radio_tracer_mask * clamped_tracers
    
    if convolve_flux is True:
        beam_kernel = Gaussian2DKernel((sigma_beam_arcsec * kpc_per_arcsec).to(u.kpc).value)
        flux_density_tracer_weighted = convolve(flux_density_tracer_weighted.to(u.Jy), beam_kernel, boundary='extend') * u.Jy
    
    return (radio_luminosity_tracer_weighted, flux_density_tracer_weighted)


# In[16]:

def get_last_timestep(simulation_directory):
    with suppress_stdout():
        return pp.nlast_info(w_dir=simulation_directory)['nlast']


# In[17]:

# unit values King profile

unit_density_king_14p5 = king_14p5.get_density(jet_king_14p5_m25.L_1b)
unit_length_king_14p5 = jet_king_14p5_m25.length_scaling
unit_time_king_14p5 = jet_king_14p5_m25.time_scaling
unit_mass_king_14p5 = (unit_density_king_14p5 * (unit_length_king_14p5 ** 3)).to(u.kg)
unit_pressure_king_14p5 = (unit_mass_king_14p5 / (unit_length_king_14p5 * unit_time_king_14p5 ** 2)).to(u.Pa)
unit_energy_king_14p5 = (unit_mass_king_14p5 * (unit_length_king_14p5 ** 2) / (unit_time_king_14p5 ** 2)).to(u.J)
unit_speed_king_14p5 = king_14p5.sound_speed

unit_density_king_12p5 = king_12p5.get_density(jet_king_12p5_m25.L_1b)
unit_length_king_12p5 = jet_king_12p5_m25.length_scaling
unit_time_king_12p5 = jet_king_12p5_m25.time_scaling
unit_mass_king_12p5 = (unit_density_king_12p5 * (unit_length_king_12p5 ** 3)).to(u.kg)
unit_pressure_king_12p5 = (unit_mass_king_12p5 / (unit_length_king_12p5 * unit_time_king_12p5 ** 2)).to(u.Pa)
unit_energy_king_12p5 = (unit_mass_king_12p5 * (unit_length_king_12p5 ** 2) / (unit_time_king_12p5 ** 2)).to(u.J)
unit_speed_king_12p5 = king_12p5.sound_speed

# unit values Makino profile

unit_density_makino_14p5 = makino_14p5.get_density(jet_makino_14p5_m25.L_1b)
unit_length_makino_14p5 = jet_makino_14p5_m25.length_scaling
unit_time_makino_14p5 = jet_makino_14p5_m25.time_scaling
unit_mass_makino_14p5 = (unit_density_makino_14p5 * (unit_length_makino_14p5 ** 3)).to(u.kg)
unit_pressure_makino_14p5 = (unit_mass_makino_14p5 / (unit_length_makino_14p5 * unit_time_makino_14p5 ** 2)).to(u.Pa)
unit_energy_makino_14p5 = (unit_mass_makino_14p5 * (unit_length_makino_14p5 ** 2) / (unit_time_makino_14p5 ** 2)).to(u.J)
unit_speed_makino_14p5 = makino_14p5.sound_speed

unit_density_makino_12p5 = makino_12p5.get_density(jet_makino_12p5_m25.L_1b)
unit_length_makino_12p5 = jet_makino_12p5_m25.length_scaling
unit_time_makino_12p5 = jet_makino_12p5_m25.time_scaling
unit_mass_makino_12p5 = (unit_density_makino_12p5 * (unit_length_makino_12p5 ** 3)).to(u.kg)
unit_pressure_makino_12p5 = (unit_mass_makino_12p5 / (unit_length_makino_12p5 * unit_time_makino_12p5 ** 2)).to(u.Pa)
unit_energy_makino_12p5 = (unit_mass_makino_12p5 * (unit_length_makino_12p5 ** 2) / (unit_time_makino_12p5 ** 2)).to(u.J)
unit_speed_makino_12p5 = makino_12p5.sound_speed


# # Code for Plots

# In[18]:

def plot_simulation_grid():
    
    directory = run_dirs[standard_all[0]]
    timestep = 0
    
    fig,ax = newfig(1, 1)

    # load timestep data file
    d = ps.load_timestep_data(timestep, directory)

    X1, X2 = ps.sphericaltocartesian(d)
    X1 = X1 * jet_14p5_m25.length_scaling.value
    X2 = X2 * jet_14p5_m25.length_scaling.value

    r_step = 25
    theta_step = 10

    x1_points = X1[::r_step, ::theta_step]
    x2_points = X2[::r_step, ::theta_step]

    r_stepped = np.concatenate((d.x1[::r_step], [d.x1[-1]]))*jet_14p5_m25.length_scaling.value
    theta_stepped = np.concatenate((d.x2[::theta_step], [d.x2[-1]]))
    s = (r_stepped.shape[0], theta_stepped.shape[0])

    x1_lines = np.zeros(s)
    x2_lines = np.zeros(s)

    for i in range(0, s[0]):
        x1_lines[i,:] = r_stepped[i] * np.cos(theta_stepped - (np.pi / 2))
        x2_lines[i,:] = r_stepped[i] * np.sin(-(theta_stepped - (np.pi/2)))

    # plot data
    ax.set_xlim((0, 400))
    ax.set_ylim((0, 400))
    #with plt.style.context('density-plot.mplstyle'):
        #im1 = ax.pcolormesh(X1, X2, np.log10((d.rho.T * unit_density).si.value), shading='flat');
        #(ca1, div, cax1) = create_colorbar(im1, ax, fig, padding=0.65)
        #ax.plot(x1_points, x2_points, '.', color='w', alpha=0.5, linewidth=0.5)
        
    ax.plot(x1_lines.T, x2_lines.T, 'k', alpha=1, linewidth=1)
    ax.plot(x1_lines, x2_lines, 'k', alpha=1, linewidth=1)

    #ca1.set_label(r'$\log_{10}{kg / m^3}$')

    # reset limits
    ax.set_xlabel(r'X, $\theta = \pi / 2$')
    ax.set_ylabel(r'Y, $\theta = 0$')
    ax.set_aspect('equal')
    
    savefig('simulation-grid', fig)
    plt.close();
    
def plot_density_with_simulation_grid():
    
    directory = run_dirs[standard_all[0]]
    timestep = 0
    
    fig,ax = newfig(1, 1)

    # load timestep data file
    d = ps.load_timestep_data(timestep, directory)

    X1, X2 = ps.sphericaltocartesian(d)
    X1 = X1 * jet_14p5_m25.length_scaling.value
    X2 = X2 * jet_14p5_m25.length_scaling.value

    r_step = 25
    theta_step = 10

    x1_points = X1[::r_step, ::theta_step]
    x2_points = X2[::r_step, ::theta_step]

    r_stepped = np.concatenate((d.x1[::r_step], [d.x1[-1]]))*jet_14p5_m25.length_scaling.value
    theta_stepped = np.concatenate((d.x2[::theta_step], [d.x2[-1]]))
    s = (r_stepped.shape[0], theta_stepped.shape[0])

    x1_lines = np.zeros(s)
    x2_lines = np.zeros(s)

    for i in range(0, s[0]):
        x1_lines[i,:] = r_stepped[i] * np.cos(theta_stepped - (np.pi / 2))
        x2_lines[i,:] = r_stepped[i] * np.sin(-(theta_stepped - (np.pi/2)))

    # plot data
    ax.set_xlim((0, 400))
    ax.set_ylim((0, 400))
    with plt.style.context('density-plot.mplstyle'):
        im1 = ax.pcolormesh(X1, X2, np.log10((d.rho.T * unit_density_14p5).si.value), shading='flat');
        im1.set_rasterized(True)
        (ca1, div, cax1) = create_colorbar(im1, ax, fig, padding=0.65)
        #ax.plot(x1_points, x2_points, '.', color='w', alpha=0.5, linewidth=0.5)
        
    ax.plot(x1_lines.T, x2_lines.T, 'w', alpha=0.5, linewidth=1)
    ax.plot(x1_lines, x2_lines, 'w', alpha=0.5, linewidth=1)

    ca1.set_label(r'Density ($\log_{10}{kg / m^3})$')

    # reset limits
    ax.set_xlabel('X (kpc)')
    ax.set_ylabel('Y (kpc)')
    ax.set_aspect('equal')
    
    savefig('density-with-simulation-grid', fig)
    plt.close();
    
def plot_concentration_profiles():
    import astropy.cosmology as cosmology
    mass_range_exponents = np.arange(10, 15, 0.25)
    mass_range = (10 ** mass_range_exponents) * u.M_sun

    z = 0

    c_klypin_planck_all = []
    c_klypin_planck_relaxed = []
    c_dutton = []

    for m in mass_range:
        c_klypin_planck_all.append(NFW.NFW_Profile.calculate_concentration(m, method='klypin-planck-all', redshift=z, cosmo=cosmology.Planck15))
        c_klypin_planck_relaxed.append(NFW.NFW_Profile.calculate_concentration(m, method='klypin-planck-relaxed', redshift=z, cosmo=cosmology.Planck15))
        c_dutton.append(NFW.NFW_Profile.calculate_concentration(m, method='dutton', redshift=z, cosmo=cosmology.Planck15))
    
    fig,ax=newfig(0.5,1)

    ax.plot(mass_range, c_klypin_planck_all, label='Klypin+2016 - All')
    ax.plot(mass_range, c_klypin_planck_relaxed, label='Klypin+2016 - Relaxed')
    ax.plot(mass_range, c_dutton, label='Dutton+2014')
    ax.set_xlabel('Virial Mass ($M_\odot$)')
    ax.set_ylabel('Concentration')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best')
    
    savefig('concentration-profiles', fig)
    plt.close();

def plot_nfw_density_different_concentration_profiles():
    import astropy.cosmology as cosmology

    z = 0
    virial_mass = 10 ** 14.5 * u.M_sun

    nfw_klypin_planck_all = NFW.NFW_Profile(virial_mass, z, delta_vir=200, cosmo=cosmology.Planck15, concentration_method='klypin-planck-all')
    nfw_klypin_planck_relaxed = NFW.NFW_Profile(virial_mass, z, delta_vir=200, cosmo=cosmology.Planck15, concentration_method='klypin-planck-relaxed')
    nfw_dutton = NFW.NFW_Profile(virial_mass, z, delta_vir=200, cosmo=cosmology.Planck15, concentration_method='dutton')

    steps = np.arange(0.01, 10, 0.001) # steps in terms of percentage of the virial radius

    rho_klypin_planck_all = []
    rho_klypin_planck_relaxed = []
    rho_dutton = []

    for i in steps:
        rho_klypin_planck_all.append(nfw_klypin_planck_all.get_density(i*nfw_klypin_planck_all.virial_radius).value)
        rho_klypin_planck_relaxed.append(nfw_klypin_planck_relaxed.get_density(i*nfw_klypin_planck_relaxed.virial_radius).value)
        rho_dutton.append(nfw_dutton.get_density(i*nfw_dutton.virial_radius).value)
    
    fig,ax=newfig(0.5,1)

    ax.set_xlabel('Fraction of virial radius')
    ax.set_ylabel('Gas density ($g / cm^{-3}$)')
    ax.plot(steps, rho_klypin_planck_all, label='Klypin+2016 - All')
    ax.plot(steps, rho_klypin_planck_relaxed, label='Klypin+2016 - Relaxed')
    ax.plot(steps, rho_dutton, label='Dutton+2014')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticklabels([0, 0.01, 0.1, 1.0, 10.0]);
    ax.legend(loc='lower left');
    
    savefig('nfw-density-different-concentration-profiles', fig)
    plt.close();
    
def plot_density_profiles():
    fig,ax = newfig(1)

    x = np.arange(0, 1e4) * u.kpc
    rho_14p5_makino = []
    rho_12p5_makino = []
    rho_14p5_king = []
    rho_12p5_king = []

    for i in x:
        rho_14p5_makino.append(makino_14p5.get_density(i).si.value)
        rho_12p5_makino.append(makino_12p5.get_density(i).si.value)
        rho_14p5_king.append(king_14p5.get_density(i).si.value)
        rho_12p5_king.append(king_12p5.get_density(i).si.value)

    # plot makino curves
    ax.loglog(x, rho_12p5_makino, label=r'$10^{12.5} M_{\odot}$ Makino')
    ax.loglog(x, rho_14p5_makino, label=r'$10^{14.5} M_{\odot}$ Makino')
    
    # reset colour cycle
    ax.set_color_cycle(None)
    
    # plot king curves
    ax.loglog(x, rho_12p5_king, label=r'$10^{12.5} M_{\odot}$ King', linestyle='--')
    ax.loglog(x, rho_14p5_king, label=r'$10^{14.5} M_{\odot}$ King', linestyle='--')

    # reset limits
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel(r'Density ($\log_{10}{kg / m^3}$)')
    ax.legend(loc='best')
    
    savefig('density-profiles', fig)
    plt.close();
    
def plot_velocity_ramp():
    fig,ax = newfig(1)
    
    delta_t_14p5 = 4.01E-3
    delta_t_12p5 = 83.3E-3
    step=0.01E-3
    x_ramp_up_14p5 = np.arange(-delta_t_14p5,0,step)
    x_ramp_up_12p5 = np.arange(-delta_t_12p5,0,step)
    x_ramp_down_14p5 = np.arange(0, delta_t_14p5, step)
    x_ramp_down_12p5 = np.arange(0, delta_t_12p5, step)
    
    ramp_up_14p5 = (1 - np.cos((np.pi / 2) * (x_ramp_up_14p5+delta_t_14p5)/delta_t_14p5))
    ramp_up_12p5 = (1 - np.cos((np.pi / 2) * (x_ramp_up_12p5+delta_t_12p5)/delta_t_12p5))
    ramp_down_14p5 = np.cos((np.pi/2) * x_ramp_down_14p5/delta_t_14p5)
    ramp_down_12p5 = np.cos((np.pi/2) * x_ramp_down_12p5/delta_t_12p5)
    
    # plot ramp up curves
    ax.plot(x_ramp_up_12p5, ramp_up_12p5, label=r'$10^{12.5} M_{\odot}$')
    ax.plot(x_ramp_up_14p5, ramp_up_14p5, label=r'$10^{14.5} M_{\odot}$')
    
    # reset colour cycle
    ax.set_color_cycle(None)
    
    # plot ramp down curves
    ax.plot(x_ramp_down_12p5, ramp_down_12p5)
    ax.plot(x_ramp_down_14p5, ramp_down_14p5)
    
    # add labels
    ax.set_xlabel('Difference from switch on/off time (Myr)')
    ax.set_ylabel('Fraction of desired jet velocity')
    ax.legend(loc='best')
    
    savefig('velocity-ramp', fig)
    plt.close();

def plot_density_thumbnail_grid(sim_id, time_scaling, length_scaling, var_scaling, xlim, ylim, vmin, vmax, step=150, max_time=1000):
    fs = figsize(1.3, 1.3)
    ncol = 2
    fp = plotting.FigureProperties(fs[0], fs[1], '', 'equal', xlim, ylim, 'X (kpc)', 'Y (kpc)', r'Density ($\log_{10}{kg / m^3})$', 0.2, 0,
                                   vmin, vmax)
    times = np.arange(150, max_time, step)
    fig = plotting.plot_multiple_timesteps(run_dirs[sim_id], times, time_scaling, length_scaling, 'rho', fp, ncol=ncol, vs=var_scaling)
    
    savefig('density-thumbnail-grid-{0}'.format(run_codes[sim_id]), fig)
    plt.close();
    
def plot_pressure_thumbnail_grid(sim_id, time_scaling, length_scaling, var_scaling, xlim, ylim, vmin, vmax, step=150, max_time=1000):
    fs = figsize(1.3, 1.3)
    ncol = 2
    fp = plotting.FigureProperties(fs[0], fs[1], '', 'equal', xlim, ylim, 'X (kpc)', 'Y (kpc)', r'Pressure ($\log_{10}{Pa}$)', 0.2, 0,
                                   vmin, vmax)
    times = np.arange(150, max_time, step)
    with plt.style.context('pressure-plot.mplstyle'):
        fig = plotting.plot_multiple_timesteps(run_dirs[sim_id], times, time_scaling, length_scaling, 'prs', fp, ncol=ncol, vs=var_scaling)
    
    savefig('pressure-thumbnail-grid-{0}'.format(run_codes[sim_id]), fig)
    plt.close();
    
def plot_energy_components(sim_id, unit_energy, jet, step=10, dl=False, loc='upper center', frameon=False):
    
    timesteps = range(0,get_last_timestep(run_dirs[sim_id]),step)

    # load times
    times = ps.LoadSimulationTimes(run_dirs[sim_id], timesteps)

    sim_times = np.asarray(times)
    
    fig,ax = newfig(1,0.5)

    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    
    plotting.plot_energy(run_dirs[sim_id], timesteps, sim_times, jet, run_codes[sim_id], plot_theoretical=False, plot_flux=False,
                         fig=fig, ax=ax, energy_scaling = unit_energy, draw_legend=False, draw_title=False)
    ax.set_yticks([0, 0.5e52, 1.0e52])
    ax.set_xlabel('Time (Myr)')
    ax.set_ylabel('Energy (J)')
    ax.set_xlim(0,200)
    
    if dl is True:
        ax.legend(loc=loc, frameon=frameon)
    
    savefig('energy-components-{0}'.format(run_codes[sim_id]), fig)
    plt.close();
    
def plot_feedback_efficiency(sim_id1, sim_id2, label1, label2, ts1, ts2, ntrc1, ntrc2,
                             trc_cutoff=0.005, step=10, start=1, draw_legend=True):
    
    def get_scaled_times(sim_id, tsteps, unit_t):
        times = ps.LoadSimulationTimes(run_dirs[sim_id], tsteps)

        sim_times = np.asarray(times)
        return (sim_times * unit_t)
    
    def get_tracer_weighted_energy_sum(sim_id, tsteps, trc_cutoff, ntrc):
        initial_data = ps.load_timestep_data(0, run_dirs[sim_id])
        vol = ps.calculate_cell_volume(initial_data)
        
        e_trc = np.zeros((4, len(tsteps)))
        e_tot = np.zeros_like(e_trc)
        
        for i in range(len(tsteps)):
            sim_data = ps.load_timestep_data(tsteps[i], run_dirs[sim_id])
            e_trc_single = ps.calculate_energy(sim_data, initial_data, volume=vol, tracer_weighted=True, tracer_or_not=False, 
                                ntracers=ntrc, tracer_threshold=trc_cutoff)
            e_tot_single = ps.calculate_energy(sim_data, initial_data, volume=vol, tracer_weighted=False)
            
            e_trc[0,i] = e_trc_single[0].sum()
            e_trc[1,i] = e_trc_single[1].sum()
            e_trc[2,i] = e_trc_single[2].sum()
            e_trc[3,i] = e_trc_single[3].sum()
            
            e_tot[0,i] = e_tot_single[0].sum()
            e_tot[1,i] = e_tot_single[1].sum()
            e_tot[2,i] = e_tot_single[2].sum()
            e_tot[3,i] = e_tot_single[3].sum()
        
        e_trc = e_trc - (e_trc[:,0])[:, np.newaxis]
        e_tot = e_tot - (e_tot[:,0])[:, np.newaxis]
        
        e_ratio = e_trc / e_tot[0,:]
        
        return e_ratio
            
    # generate timesteps
    timesteps1 = range(start,get_last_timestep(run_dirs[sim_id1]),step) + [get_last_timestep(run_dirs[sim_id1])]
    timesteps2 = range(start,get_last_timestep(run_dirs[sim_id2]),step) + [get_last_timestep(run_dirs[sim_id2])]
    
    # calculate scaled times
    s_times_1 = get_scaled_times(sim_id1, timesteps1, ts1)
    s_times_2 = get_scaled_times(sim_id2, timesteps2, ts2)
    
    # calculate energy ratios
    e_ratio_1 = get_tracer_weighted_energy_sum(sim_id1, timesteps1, trc_cutoff, ntrc1)
    e_ratio_2 = get_tracer_weighted_energy_sum(sim_id2, timesteps2, trc_cutoff, ntrc2)
    
    fig,ax=newfig(1)

    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    
    linewidths=[2.5, 2.0, 1.5, 2.0]
    dash_list=[(), (5,4,3,4), (1,1), (2,1)]
    linestyles=['-', '--', '--', '--']
    
    for i in range(e_ratio_1.shape[0]):
        # plot energy
        ax.plot(s_times_1, e_ratio_1[i,:], linewidth=linewidths[i], dashes=dash_list[i], linestyle=linestyles[i])
        ax.plot(s_times_2, e_ratio_2[i,:], linewidth=linewidths[i], dashes=dash_list[i], linestyle=linestyles[i])
        
        # reset colour cycle
        ax.set_color_cycle(None)

    #Create custom artists
    colour_cycle = ax._get_lines.prop_cycler
    n1_color_legend = plt.Line2D((0,1),(0,0), color=colour_cycle.next()['color'])
    n4_color_legend = plt.Line2D((0,1),(0,0), color=colour_cycle.next()['color'])

    tot_legend = plt.Line2D((0,1),(0,0), color='k', linewidth=linewidths[0], dashes=dash_list[0], linestyle=linestyles[0])
    kin_legend = plt.Line2D((0,1),(0,0), color='k', linewidth=linewidths[1], dashes=dash_list[1], linestyle=linestyles[1])
    pot_legend = plt.Line2D((0,1),(0,0), color='k', linewidth=linewidths[2], dashes=dash_list[2], linestyle=linestyles[2])
    therm_legend = plt.Line2D((0,1),(0,0), color='k', linewidth=linewidths[3], dashes=dash_list[3], linestyle=linestyles[3])

    #Create legend from custom artist/label lists
    leg1 = ax.legend([n1_color_legend,n4_color_legend],[label1, label2], loc='upper center')
    if draw_legend is True:
        leg2 = ax.legend([tot_legend, kin_legend, pot_legend, therm_legend], 
                         ['total', 'kinetic', 'potential', 'thermal'], loc='upper left')
        ax.add_artist(leg1)
    ax.set_xlabel('Time (Myr)')
    ax.set_ylabel(r'$E_\mathrm{ambient} / E_\mathrm{injected}$')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 200)
    
    savefig('feedback-efficiency-{0}-vs-{1}'.format(run_codes[sim_id1], run_codes[sim_id2]), fig)
    plt.close();
    
def plot_feedback_efficiency_different_tracer_cutoff(sim_id1, sim_id2, label1, label2, ts1, ts2, ntrc1, ntrc2,
                             trc_cutoff=0.005, step=10, start=1):
    
    def get_scaled_times(sim_id, tsteps, unit_t):
        times = ps.LoadSimulationTimes(run_dirs[sim_id], tsteps)

        sim_times = np.asarray(times)
        return (sim_times * unit_t)
    
    def get_tracer_weighted_energy_sum(sim_id, tsteps, trc_cutoff, ntrc):
        initial_data = ps.load_timestep_data(0, run_dirs[sim_id])
        vol = ps.calculate_cell_volume(initial_data)
        
        e_trc = np.zeros((4, len(tsteps)))
        e_tot = np.zeros_like(e_trc)
        
        for i in range(len(tsteps)):
            sim_data = ps.load_timestep_data(tsteps[i], run_dirs[sim_id])
            e_trc_single = ps.calculate_energy(sim_data, initial_data, volume=vol, tracer_weighted=True, tracer_or_not=False, 
                                ntracers=ntrc, tracer_threshold=trc_cutoff)
            e_tot_single = ps.calculate_energy(sim_data, initial_data, volume=vol, tracer_weighted=False)
            
            e_trc[0,i] = e_trc_single[0].sum()
            e_trc[1,i] = e_trc_single[1].sum()
            e_trc[2,i] = e_trc_single[2].sum()
            e_trc[3,i] = e_trc_single[3].sum()
            
            e_tot[0,i] = e_tot_single[0].sum()
            e_tot[1,i] = e_tot_single[1].sum()
            e_tot[2,i] = e_tot_single[2].sum()
            e_tot[3,i] = e_tot_single[3].sum()
        
        e_trc = e_trc - (e_trc[:,0])[:, np.newaxis]
        e_tot = e_tot - (e_tot[:,0])[:, np.newaxis]
        
        e_ratio = e_trc / e_tot[0,:]
        
        return e_ratio
            
    # generate timesteps
    timesteps1 = range(start,get_last_timestep(run_dirs[sim_id1]),step) + [get_last_timestep(run_dirs[sim_id1])]
    timesteps2 = range(start,get_last_timestep(run_dirs[sim_id2]),step) + [get_last_timestep(run_dirs[sim_id2])]
    
    # calculate scaled times
    s_times_1 = get_scaled_times(sim_id1, timesteps1, ts1)
    s_times_2 = get_scaled_times(sim_id2, timesteps2, ts2)
    
    # calculate energy ratios
    e_ratio_1 = get_tracer_weighted_energy_sum(sim_id1, timesteps1, trc_cutoff, ntrc1)
    e_ratio_2 = get_tracer_weighted_energy_sum(sim_id2, timesteps2, trc_cutoff, ntrc2)
    
    fig,ax=newfig(1)

    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    
    linewidths=[2.5, 2.0, 1.5, 2.0]
    dash_list=[(), (5,4,3,4), (1,1), (2,1)]
    linestyles=['-', '--', '--', '--']
    
    for i in range(e_ratio_1.shape[0]):
        # plot energy
        ax.plot(s_times_1, e_ratio_1[i,:], linewidth=linewidths[i], dashes=dash_list[i], linestyle=linestyles[i])
        ax.plot(s_times_2, e_ratio_2[i,:], linewidth=linewidths[i], dashes=dash_list[i], linestyle=linestyles[i])
        
        # reset colour cycle
        ax.set_color_cycle(None)

    #Create custom artists
    colour_cycle = ax._get_lines.prop_cycler
    n1_color_legend = plt.Line2D((0,1),(0,0), color=colour_cycle.next()['color'])
    n4_color_legend = plt.Line2D((0,1),(0,0), color=colour_cycle.next()['color'])

    tot_legend = plt.Line2D((0,1),(0,0), color='k', linewidth=linewidths[0], dashes=dash_list[0], linestyle=linestyles[0])
    kin_legend = plt.Line2D((0,1),(0,0), color='k', linewidth=linewidths[1], dashes=dash_list[1], linestyle=linestyles[1])
    pot_legend = plt.Line2D((0,1),(0,0), color='k', linewidth=linewidths[2], dashes=dash_list[2], linestyle=linestyles[2])
    therm_legend = plt.Line2D((0,1),(0,0), color='k', linewidth=linewidths[3], dashes=dash_list[3], linestyle=linestyles[3])

    #Create legend from custom artist/label lists
    leg1 = ax.legend([n1_color_legend,n4_color_legend],[label1, label2], loc='upper center')
    leg2 = ax.legend([tot_legend, kin_legend, pot_legend, therm_legend], 
                     ['total', 'kinetic', 'potential', 'thermal'], loc='upper left')
    ax.add_artist(leg1)
    ax.set_xlabel('Time (Myr)')
    ax.set_ylabel(r'$E_{ambient} / E_{injected}$')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 200)
    
    savefig('feedback-efficiency-{0}-{1}-vs-{2}'.format(trc_cutoff, run_codes[sim_id1], run_codes[sim_id2]), fig)
    plt.close();
    
def plot_feedback_efficiency_different_tracer_cutoff_grid(sim_id1, sim_id2, label1, label2, ts1, ts2, ntrc1, ntrc2,
                             trc_cutoffs, step=10, start=1, ncol=3):
    
    def get_scaled_times(sim_id, tsteps, unit_t):
        times = ps.LoadSimulationTimes(run_dirs[sim_id], tsteps)

        sim_times = np.asarray(times)
        return (sim_times * unit_t)
    
    def get_tracer_weighted_energy_sum(sim_id, tsteps, trc_cutoff, ntrc):
        initial_data = ps.load_timestep_data(0, run_dirs[sim_id])
        vol = ps.calculate_cell_volume(initial_data)
        
        e_trc = np.zeros((4, len(tsteps)))
        e_tot = np.zeros_like(e_trc)
        
        for i in range(len(tsteps)):
            sim_data = ps.load_timestep_data(tsteps[i], run_dirs[sim_id])
            e_trc_single = ps.calculate_energy(sim_data, initial_data, volume=vol, tracer_weighted=True, tracer_or_not=False, 
                                ntracers=ntrc, tracer_threshold=trc_cutoff)
            e_tot_single = ps.calculate_energy(sim_data, initial_data, volume=vol, tracer_weighted=False)
            
            e_trc[0,i] = e_trc_single[0].sum()
            e_trc[1,i] = e_trc_single[1].sum()
            e_trc[2,i] = e_trc_single[2].sum()
            e_trc[3,i] = e_trc_single[3].sum()
            
            e_tot[0,i] = e_tot_single[0].sum()
            e_tot[1,i] = e_tot_single[1].sum()
            e_tot[2,i] = e_tot_single[2].sum()
            e_tot[3,i] = e_tot_single[3].sum()
        
        e_trc = e_trc - (e_trc[:,0])[:, np.newaxis]
        e_tot = e_tot - (e_tot[:,0])[:, np.newaxis]
        
        e_ratio = e_trc / e_tot[0,:]
        
        return e_ratio
            
    # calculate number of rows from max number of columns
    nrow = int(np.ceil(len(trc_cutoffs) / float(ncol)))
    
    # create the figure
    gs = gridspec.GridSpec(nrow, ncol)
    
    # generate timesteps
    timesteps1 = range(start,get_last_timestep(run_dirs[sim_id1]),step) + [get_last_timestep(run_dirs[sim_id1])]
    timesteps2 = range(start,get_last_timestep(run_dirs[sim_id2]),step) + [get_last_timestep(run_dirs[sim_id2])]
    
    # calculate scaled times
    s_times_1 = get_scaled_times(sim_id1, timesteps1, ts1)
    s_times_2 = get_scaled_times(sim_id2, timesteps2, ts2)
    
    fs = figsize(1.25, 0.75)
    fig = plt.figure(figsize=fs)
    #fig,ax=newfig(1,1.5)
    
    for j,tcut in enumerate(trc_cutoffs):
        
        ax = plt.subplot(gs[j / ncol, j % ncol])
    
        # calculate energy ratios
        e_ratio_1 = get_tracer_weighted_energy_sum(sim_id1, timesteps1, tcut, ntrc1)
        e_ratio_2 = get_tracer_weighted_energy_sum(sim_id2, timesteps2, tcut, ntrc2)

        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False))

        linewidths=[2.5, 2.0, 1.5, 2.0]
        dash_list=[(), (5,4,3,4), (1,1), (2,1)]
        linestyles=['-', '--', '--', '--']

        for i in range(e_ratio_1.shape[0]):
            # plot energy
            ax.plot(s_times_1, e_ratio_1[i,:], linewidth=linewidths[i], dashes=dash_list[i], linestyle=linestyles[i])
            ax.plot(s_times_2, e_ratio_2[i,:], linewidth=linewidths[i], dashes=dash_list[i], linestyle=linestyles[i])

            # reset colour cycle
            ax.set_color_cycle(None)

        ax.set_ylim(0, 1)
        ax.set_xlim(0, 200)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.15, 0.9, '{0}'.format(tcut), horizontalalignment='center',
                verticalalignment='center',transform=ax.transAxes, color='black')
    
    fig.subplots_adjust(hspace=0.001, wspace=0.001)
    savefig('feedback-efficiency-grid-{0}-vs-{1}'.format(run_codes[sim_id1], run_codes[sim_id2]), fig)
    plt.close();
    
def plot_pd_tracks(sim_id1, sim_id2, label1, label2, times1, times2, ntrc1, ntrc2, ts1, ts2, ls1, ls2, ds1, ds2, 
                   trc_cutoff=0.005, xlim=(0,200), ylim=(10 ** 25.5,10 ** 26.5),
                   xticks=[2, 10, 100, 200], xtick_labels=['2', '10', '100', '200'],
                   yticks=[np.float(10**25.5), np.float(10**26), np.float(10**26.5)],
                   ytick_labels=[r'$10^{25.5}$', r'$10^{26}$', r'$10^{26.5}$'],
                   legend_loc='upper left'):
    
    def get_lobe_length(simulation_data, tracer_threshold, ntracers):
        combine_tracers(simulation_data, ntracers, )
        radio_tracer_mask, clamped_tracers, radio_combined_tracers = clamp_tracers(simulation_data, 
                                                                                   ntracers, 
                                                                                   tracer_threshold=tracer_threshold)

        theta_index = 0
        r_indicies = np.where(radio_tracer_mask[:, theta_index] == 1)[-1]
        if len(r_indicies) == 0:
            final_r_index = 0
        else:
            final_r_index = r_indicies[-1]
        return simulation_data.x1[final_r_index], final_r_index
    
    def calculate_length_and_luminosity(sim_dir, tsteps, ntrc, ds, ls, ts):
        lengths = []
        lumin = []
        
        for i in tsteps:
            d = ps.load_timestep_data(i, sim_dir)
            lobe_length, ind = get_lobe_length(d, trc_cutoff, ntrc)
            (r_lumin, flux_d) = get_luminosity(d, 
                                           ds, 
                                           ls, 
                                           ts,
                                           0.1,
                                           5 * u.arcsec,
                                           1)

            lengths.append(lobe_length)
            lumin.append(r_lumin.sum().to(u.W / u.Hz).value)
        return (lengths, lumin)
        
    lengths1, lumin1 = calculate_length_and_luminosity(run_dirs[sim_id1], times1, ntrc1, ds1, ls1, ts1)
    lengths2, lumin2 = calculate_length_and_luminosity(run_dirs[sim_id2], times2, ntrc2, ds2, ls2, ts2)
    
    lengths1 = np.asarray(lengths1)
    lengths2 = np.asarray(lengths2)
    lumin1 = np.asarray(lumin1)
    lumin2 = np.asarray(lumin2)
    
    fig,ax = newfig(1)

    ax.loglog(lengths1 * ls1.value, lumin1, label=label1)
    ax.loglog(lengths2 * ls2.value, lumin2, label=label2)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.set_xlabel('Lobe length (kpc)')
    ax.set_ylabel(r'1.4-GHz luminosity (W Hz$^{-1}$)')
    
    ax.set_xticks(xticks, minor=False)
    ax.set_xticklabels(xtick_labels)
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    
    ax.legend(loc=legend_loc)
    
    savefig('p-d-tracks-{0}-{1}'.format(run_codes[sim_id1], run_codes[sim_id2]), fig)
    plt.close();
    
def plot_surface_brightness(sim_id, timestep, ntrc, ts, ls, ds, redshift=0.1, beamsize=5*u.arcsec, showbeam=True,
                           xlim=(-1,1), ylim=(-1.5,1.5), width=2):
    from astropy.cosmology import Planck15 as cosmo
 
    fig,ax=newfig(width)
    
    # calculate beam radius
    sigma_beam = (beamsize / 2.355)
    
    # calculate kpc per arcsec
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(redshift).to(u.kpc / u.arcsec)

    # load timestep data file
    d = ps.load_timestep_data(timestep, run_dirs[sim_id])

    X1, X2 = ps.sphericaltocartesian(d)
    X1 = X1 * (ls / kpc_per_arcsec).to(u.arcmin).value
    X2 = X2 * (ls / kpc_per_arcsec).to(u.arcmin).value

    (r_lumin, flux_d) = get_luminosity(d, 
                                   ds, 
                                   ls, 
                                   ts,
                                   redshift,
                                   beamsize,
                                   ntrc,
                                   calculate_luminosity=False,
                                   convolve_flux=True)

    # plot data
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    with plt.style.context('flux-plot.mplstyle'):
        im = ax.pcolormesh(X1, X2, np.log10(flux_d.to(u.mJy).value).T, shading='flat', vmin=-3.0, vmax=2.0);
        im.set_rasterized(True)
        
        im = ax.pcolormesh(-X1, X2, np.log10(flux_d.to(u.mJy).value).T, shading='flat', vmin=-3.0, vmax=2.0);
        im.set_rasterized(True)
        
        im = ax.pcolormesh(-X1, -X2, np.log10(flux_d.to(u.mJy).value).T, shading='flat', vmin=-3.0, vmax=2.0);
        im.set_rasterized(True)
        
        im = ax.pcolormesh(X1, -X2, np.log10(flux_d.to(u.mJy).value).T, shading='flat', vmin=-3.0, vmax=2.0);
        im.set_rasterized(True)
        (ca, div, cax) = create_colorbar(im, ax, fig, position='right', padding=0.2)

    ca.set_label(r'$\log_{10}{mJy / beam}$')
    
    circ = plt.Circle((xlim[1]*0.8, ylim[0]*0.8), color='w', fill=True, radius=sigma_beam.to(u.arcmin).value, alpha=0.7)
    circ.set_rasterized(True)
    ax.add_artist(circ)

    # reset limits
    ax.set_xlabel('X ($\'$)')
    ax.set_ylabel('Y ($\'$)')
    ax.set_aspect('equal')
    
    savefig('surface-brightness-{0}-{1}'.format(run_codes[sim_id], timestep), fig)
    plt.close();
    
def plot_resolution_test():
    fig,ax1 = newfig(1.2)
    fig.delaxes(ax1)
        
    for i in resolution_test:
        ax = fig.add_subplot(1, len(resolution_test), resolution_test.index(i) + 1)
        d = ps.load_timestep_data(get_last_timestep(old_run_dirs[i]), old_run_dirs[i])
        with plt.style.context('density-plot.mplstyle'):
            im = ax.pcolormesh(d.x1, d.x2, np.log10(d.rho).T)
        im.set_rasterized(True)
        ax.set_aspect('equal')
        ax.axis([0, 10, 0, 40])
        (ca, div, cax) = create_colorbar(im, ax, fig, size='10%', padding=0.05)
        if resolution_test.index(i)+1 == len(resolution_test):
            ca.set_label('$\log_{10}$ density (simulation units)')
        
        savefig('resolution-test', fig, dpi=600)
        plt.close();

def plot_lobe_lengths(sim_ids, labels, ntrcs, timesteps, ts, ls, trc_cutoff=0.005, width=1):
    def get_lobe_length(simulation_data, tracer_threshold, ntracers):
        combine_tracers(simulation_data, ntracers, )
        radio_tracer_mask, clamped_tracers, radio_combined_tracers = clamp_tracers(simulation_data, 
                                                                                   ntracers, 
                                                                                   tracer_threshold=tracer_threshold)

        theta_index = 0
        r_indicies = np.where(radio_tracer_mask[:, theta_index] == 1)[-1]
        if len(r_indicies) == 0:
            final_r_index = 0
        else:
            final_r_index = r_indicies[-1]
        return simulation_data.x1[final_r_index], final_r_index
    
    def inner_loop(i):
        for j in range(len(timesteps)):
            d = ps.load_timestep_data(timesteps[j], run_dirs[sim_ids[i]])
            l, ind = get_lobe_length(d, trc_cutoff, ntrcs[i])
            lengths[i,j] = l
            sim_times[i,j] = d.SimTime * ts[i].value
        
    fig,ax=newfig(width)
    
    lengths = np.zeros((len(sim_ids), len(timesteps)))
    sim_times = np.zeros_like(lengths)
    
    for i in range(len(sim_ids)):
        inner_loop(i)
        #for j in range(len(timesteps)):
        
        ax.plot(sim_times[i,:], lengths[i,:]*ls[i], label=labels[i])
    
    ax.set_xlabel('Simulation Time (Myr)')
    ax.set_ylabel('Lobe length (kpc)')
    ax.legend(loc='best')
    
    savefig('lobe-length-{0}'.format('-'.join(run_codes[sim_ids])), fig)
    plt.close(); 
    
def plot_lobe_lengths_diff_cutoff(sim_ids, labels, ntrcs, timesteps, ts, ls, trc_cutoff=0.005, width=1):
    def get_lobe_length(simulation_data, tracer_threshold, ntracers):
        combine_tracers(simulation_data, ntracers, )
        radio_tracer_mask, clamped_tracers, radio_combined_tracers = clamp_tracers(simulation_data, 
                                                                                   ntracers, 
                                                                                   tracer_threshold=tracer_threshold)

        theta_index = 0
        r_indicies = np.where(radio_tracer_mask[:, theta_index] == 1)[-1]
        if len(r_indicies) == 0:
            final_r_index = 0
        else:
            final_r_index = r_indicies[-1]
        return simulation_data.x1[final_r_index], final_r_index
    
    def inner_loop(i):
        for j in range(len(timesteps)):
            d = ps.load_timestep_data(timesteps[j], run_dirs[sim_ids[i]])
            l, ind = get_lobe_length(d, trc_cutoff, ntrcs[i])
            lengths[i,j] = l
            sim_times[i,j] = d.SimTime * ts[i].value
        
    fig,ax=newfig(width)
    
    lengths = np.zeros((len(sim_ids), len(timesteps)))
    sim_times = np.zeros_like(lengths)
    
    for i in range(len(sim_ids)):
        inner_loop(i)
        #for j in range(len(timesteps)):
        
        ax.plot(sim_times[i,:], lengths[i,:]*ls[i], label=labels[i])
    
    ax.set_xlabel('Simulation Time (Myr)')
    ax.set_ylabel('Lobe length (kpc)')
    ax.legend(loc='best')
    
    savefig('lobe-length-cutoff-{0}-{1}'.format(trc_cutoff, '-'.join(run_codes[sim_ids])), fig)
    plt.close(); 
    
def plot_lobe_lengths_different_timesteps(sim_ids, labels, ntrcs, max_timestep_length, timesteps, ts, ls, trc_cutoff=0.005):
    def get_lobe_length(simulation_data, tracer_threshold, ntracers):
        combine_tracers(simulation_data, ntracers, )
        radio_tracer_mask, clamped_tracers, radio_combined_tracers = clamp_tracers(simulation_data, 
                                                                                   ntracers, 
                                                                                   tracer_threshold=tracer_threshold)

        theta_index = 0
        r_indicies = np.where(radio_tracer_mask[:, theta_index] == 1)[-1]
        if len(r_indicies) == 0:
            final_r_index = 0
        else:
            final_r_index = r_indicies[-1]
        return simulation_data.x1[final_r_index], final_r_index
    
    def inner_loop(i, tsteps):
        for j in range(len(tsteps)):
            d = ps.load_timestep_data(tsteps[j], run_dirs[sim_ids[i]])
            l, ind = get_lobe_length(d, trc_cutoff, ntrcs[i])
            lengths[i,j] = l
            sim_times[i,j] = d.SimTime * ts[i].value
        
    fig,ax=newfig(1)
    
    lengths = np.zeros((len(sim_ids), max_timestep_length))
    sim_times = np.zeros_like(lengths)
    
    for i in range(len(sim_ids)):
        inner_loop(i, timesteps[i])
        #for j in range(len(timesteps)):
        
        ax.plot(sim_times[i,:len(timesteps[i])], lengths[i,:len(timesteps[i])]*ls[i], label=labels[i])
    
    ax.set_xlabel('Simulation Time (Myr)')
    ax.set_ylabel('Lobe length (kpc)')
    ax.legend(loc='best')
    
    savefig('lobe-length-diff-tsteps-{0}'.format('-'.join(run_codes[sim_ids])), fig)
    plt.close(); 
    
def plot_lobe_volumes(sim_ids, labels, ntrcs, timesteps, ts, ls, trc_cutoff=0.005, width=1):
    def get_lobe_volume(simulation_data, volume, tracer_threshold, ntracers):
        combine_tracers(simulation_data, ntracers)
        radio_tracer_mask, clamped_tracers, radio_combined_tracers = clamp_tracers(simulation_data, 
                                                                                   ntracers, 
                                                                                   tracer_threshold=tracer_threshold)
        
        v = np.sum(volume * radio_tracer_mask)
        return v
    
    def inner_loop(i):
        vol = ps.calculate_cell_volume(ps.load_timestep_data(timesteps[0], run_dirs[sim_ids[i]]))
        for j in range(len(timesteps)):
            d = ps.load_timestep_data(timesteps[j], run_dirs[sim_ids[i]])
            v = get_lobe_volume(d, vol, trc_cutoff, ntrcs[i])
            volumes[i,j] = v
            sim_times[i,j] = d.SimTime * ts[i].value
        
    fig,ax=newfig(width)
    
    volumes = np.zeros((len(sim_ids), len(timesteps)))
    sim_times = np.zeros_like(volumes)
    
    for i in range(len(sim_ids)):
        inner_loop(i)
        
        ax.plot(sim_times[i,:], volumes[i,:]*((ls[i]**3).to(u.kpc ** 3).value), label=labels[i])
    
    ax.set_xlabel('Simulation Time (Myr)')
    ax.set_ylabel('Lobe volume ($\mathrm{kpc}^3$)')
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,5))
    ax.set_yticks([0, 1e5, 2e5, 3e5, 4e5])
    ax.legend(loc='best')
    
    savefig('lobe-volume-{0}'.format('-'.join(run_codes[sim_ids])), fig)
    plt.close(); 
    
def plot_lobe_masses(sim_ids, labels, ntrcs, timesteps, ts, ls, ds, trc_cutoff=0.005, width=1):
    def get_lobe_mass(simulation_data, volume, density, length_scale, density_scale, tracer_threshold, ntracers):
        combine_tracers(simulation_data, ntracers)
        radio_tracer_mask, clamped_tracers, radio_combined_tracers = clamp_tracers(simulation_data, 
                                                                                   ntracers, 
                                                                                   tracer_threshold=tracer_threshold)
        
        m = np.sum(volume * (length_scale ** 3) * density * density_scale * radio_tracer_mask)
        return m
    
    def inner_loop(i):
        vol = ps.calculate_cell_volume(ps.load_timestep_data(timesteps[0], run_dirs[sim_ids[i]]))
        for j in range(len(timesteps)):
            d = ps.load_timestep_data(timesteps[j], run_dirs[sim_ids[i]])
            m = get_lobe_mass(d, vol, d.rho, ls[i], ds[i], trc_cutoff, ntrcs[i])
            masses[i,j] = m.to(u.M_sun).value
            sim_times[i,j] = d.SimTime * ts[i].value
        
    fig,ax=newfig(width)
    
    masses = np.zeros((len(sim_ids), len(timesteps)))
    sim_times = np.zeros_like(masses)
    
    for i in range(len(sim_ids)):
        inner_loop(i)
        
        ax.plot(sim_times[i,:], masses[i,:], label=labels[i])
    
    ax.set_xlabel('Simulation Time (Myr)')
    ax.set_ylabel('Lobe mass ($M_\odot$)')
    ax.set_yticks([0, 1e9, 2e9, 3e9, 4e9])
    ax.legend(loc='best')
    
    savefig('lobe-mass-{0}'.format('-'.join(run_codes[sim_ids])), fig)
    plt.close(); 

def plot_lobe_widths(sim_ids, labels, ntrcs, timesteps, ts, ls, trc_cutoff=0.005, width=1):
    def get_lobe_length(simulation_data, tracer_threshold, ntracers):
        combine_tracers(simulation_data, ntracers, )
        radio_tracer_mask, clamped_tracers, radio_combined_tracers = clamp_tracers(simulation_data, 
                                                                                   ntracers, 
                                                                                   tracer_threshold=tracer_threshold)

        theta_index = 0
        r_indicies = np.where(radio_tracer_mask[:, theta_index] == 1)[-1]
        if len(r_indicies) == 0:
            final_r_index = 0
        else:
            final_r_index = r_indicies[-1]
        return simulation_data.x1[final_r_index], final_r_index
    
    def inner_loop(i):
        for j in range(len(timesteps)):
            d = ps.load_timestep_data(timesteps[j], run_dirs[sim_ids[i]])
            l, ind = get_lobe_length(d, trc_cutoff, ntrcs[i])
            lengths[i,j] = l
            sim_times[i,j] = d.SimTime * ts[i].value
        
    fig,ax=newfig(width)
    
    lengths = np.zeros((len(sim_ids), len(timesteps)))
    sim_times = np.zeros_like(lengths)
    
    for i in range(len(sim_ids)):
        inner_loop(i)
        #for j in range(len(timesteps)):
        
        ax.plot(sim_times[i,:], lengths[i,:]*ls[i], label=labels[i])
    
    ax.set_xlabel('Simulation Time (Myr)')
    ax.set_ylabel('Lobe length (kpc)')
    ax.legend(loc='best')
    
    savefig('lobe-length-{0}'.format('-'.join(run_codes[sim_ids])), fig)
    plt.close(); 
    
def plot_lobe_widths(sim_ids, labels, ntrcs, timesteps, ts, ls, trc_cutoff=0.005, width=1):
    
    def get_lobe_width(simulation_data, tracer_threshold, ntracers):
        
        def get_lobe_length(simulation_data, tracer_threshold, ntracers):
            radio_tracer_mask, clamped_tracers, radio_combined_tracers = clamp_tracers(simulation_data, 
                                                                                       ntracers, 
                                                                                       tracer_threshold=tracer_threshold)

            theta_index = 0
            r_indicies = np.where(radio_tracer_mask[:, theta_index] == 1)[-1]
            if len(r_indicies) == 0:
                final_r_index = 0
            else:
                final_r_index = r_indicies[-1]
            return simulation_data.x1[final_r_index], final_r_index
        
        alpha, ind = get_lobe_length(simulation_data, tracer_threshold, ntracers)
        y = alpha / 2.0
        x = 0

        x_steps = range(1, simulation_data.n1_tot)

        radio_tracer_mask, clamped_tracers, radio_combined_tracers = clamp_tracers(simulation_data, 
                                                                                   ntracers,
                                                                                   tracer_threshold=tracer_threshold)
        
        r1 = 0
        for i in x_steps:
            r = np.sqrt((simulation_data.x1[i] **2) + (y ** 2))
            theta = (np.pi / 2) - np.arctan(y / d.x1[i])
            r_ind = np.argmax(simulation_data.x1 > r) - 1
            theta_ind = np.argmax(simulation_data.x2 > theta) - 1
            if r_ind is 0 or theta_ind is 0:
                break
            if radio_tracer_mask[r_ind, theta_ind] == 1:
                r1 = r
                x = simulation_data.x1[i]
        width = x
        return width
    
    def inner_loop(i):
        for j in range(len(timesteps)):
            d = ps.load_timestep_data(timesteps[j], run_dirs[sim_ids[i]])
            w = get_lobe_width(d, trc_cutoff, ntrcs[i])
            widths[i,j] = w
            sim_times[i,j] = d.SimTime * ts[i].value
        
    fig,ax=newfig(width)
    
    widths = np.zeros((len(sim_ids), len(timesteps)))
    sim_times = np.zeros_like(widths)
    
    for i in range(len(sim_ids)):
        inner_loop(i)
        #for j in range(len(timesteps)):
        
        ax.plot(sim_times[i,:], widths[i,:]*ls[i], label=labels[i])
    
    ax.set_xlabel('Simulation Time (Myr)')
    ax.set_ylabel('Lobe width (kpc)')
    ax.legend(loc='best')
    
    savefig('lobe-width-{0}'.format('-'.join(run_codes[sim_ids])), fig)
    plt.close(); 
    
def plot_lobe_widths_slow(sim_ids, labels, ntrcs, timesteps, ts, ls, trc_cutoff=0.005, width=1):
    
    def get_lobe_width(simulation_data, tracer_threshold, ntracers):
        
        def get_lobe_length(simulation_data, tracer_threshold, ntracers):
            radio_tracer_mask, clamped_tracers, radio_combined_tracers = clamp_tracers(simulation_data, 
                                                                                       ntracers, 
                                                                                       tracer_threshold=tracer_threshold)

            theta_index = 0
            r_indicies = np.where(radio_tracer_mask[:, theta_index] == 1)[-1]
            if len(r_indicies) == 0:
                final_r_index = 0
            else:
                final_r_index = r_indicies[-1]
            return simulation_data.x1[final_r_index], final_r_index
        
        x_steps = range(1, simulation_data.n1_tot)

        radio_tracer_mask, clamped_tracers, radio_combined_tracers = clamp_tracers(simulation_data, 
                                                                                   ntracers,
                                                                                   tracer_threshold=tracer_threshold)
        
        max_y, ind = get_lobe_length(simulation_data, tracer_threshold, ntracers)
        r1 = 0
        x = 0
        
        for y_ind in x_steps:
            y = simulation_data.x1[y_ind]
            
            if y > max_y:
                break
            
            for i in x_steps:
                r = np.sqrt((simulation_data.x1[i] **2) + (y ** 2))
                theta = (np.pi / 2) - np.arctan(y / d.x1[i])
                r_ind = np.argmax(simulation_data.x1 > r) - 1
                theta_ind = np.argmax(simulation_data.x2 > theta) - 1
                
                if r_ind is 0 or theta_ind is 0:
                    break
                    
                if radio_tracer_mask[r_ind, theta_ind] == 1 and simulation_data.x1[i] > x:
                    r1 = r
                    x = simulation_data.x1[i]
                    
        width = x
        return width
    
    def inner_loop(i):
        for j in range(len(timesteps)):
            d = ps.load_timestep_data(timesteps[j], run_dirs[sim_ids[i]])
            w = get_lobe_width(d, trc_cutoff, ntrcs[i])
            widths[i,j] = w
            sim_times[i,j] = d.SimTime * ts[i].value
        
    fig,ax=newfig(width)
    
    widths = np.zeros((len(sim_ids), len(timesteps)))
    sim_times = np.zeros_like(widths)
    
    for i in range(len(sim_ids)):
        inner_loop(i)
        #for j in range(len(timesteps)):
        
        ax.plot(sim_times[i,:], widths[i,:]*ls[i], label=labels[i])
    
    ax.set_xlabel('Simulation Time (Myr)')
    ax.set_ylabel('Lobe width (kpc)')
    ax.legend(loc='best')
    
    savefig('lobe-width-slow-{0}'.format('-'.join(run_codes[sim_ids])), fig)
    plt.close(); 
    
def plot_morphology_comparisons(var, unit_vars, label, style, timestep=100, log=True):
    runs = [standard_14p5[0], standard_12p5[0]]
    unit_lengths = [unit_length_makino_14p5, unit_length_makino_12p5]
    unit_times = [unit_time_makino_14p5, unit_time_makino_12p5]
    xlims = [(0, 30), (0, 24)]
    ylims = [(0, 50), (0, 40)]
    
    for i, run in enumerate(runs):
        d = ps.load_timestep_data(timestep, run_dirs[run])

        X1, X2 = ps.sphericaltocartesian(d)
        X1 = X1 * unit_lengths[i].value
        X2 = X2 * unit_lengths[i].value
        
        fig,ax=newfig(1.25)

        data = getattr(d, var).T*unit_vars[i]
        if log is True:
            data = np.log10(data)
        with plt.style.context(style):
            im = ax.pcolormesh(X1, X2, data)
        im.set_rasterized(True)
        (ca, div, cax) = create_colorbar(im, ax, fig)
        ca.set_label(label)
        ax.set_xlabel('X (kpc)')
        ax.set_ylabel('Y (kpc)')

        ax.set_xlim(xlims[i])
        ax.set_ylim(ylims[i])
        ax.set_aspect('equal')
        
        savefig('morphology-{0}-{1}-{2}'.format(var, run_codes[run], timestep), fig)
        plt.close();
        
def plot_lobe_lengths_multiple_tracers(sim_ids, labels, ntrcs, on_times, off_times, step, ts, ls, trc_cutoff=0.005, width=1):
    def clamp_single_tracer(tracer,
                  tracer_threshold = 1e-7,
                  tracer_effective_zero = 1e-10):
    
        # smooth the tracer data with a 2d box kernel of width 3
        box2d = Box2DKernel(3)
        smoothed_tracer = convolve(tracer, box2d, boundary='extend')
        tracer_mask = np.where(smoothed_tracer > tracer_threshold, 1.0, tracer_effective_zero)

        # create new tracer array that is clamped to tracer values
        clamped_tracer = smoothed_tracer.copy()
        clamped_tracer[clamped_tracer <= tracer_threshold] = tracer_effective_zero
    
        return (tracer_mask, clamped_tracer, smoothed_tracer)

    def get_lobe_length(simulation_data, tracer_threshold, tracer_number):
        
        # get clamped tracer
        tracer_mask, clamped_tracer, smoothed_tracer = clamp_single_tracer(getattr(simulation_data, 'tr{0}'.format(tracer_number)),
                                                                           tracer_threshold=tracer_threshold)
        theta_index = 0
        r_indicies = np.where(tracer_mask[:, theta_index] == 1)[-1]
        if len(r_indicies) == 0:
            final_r_index = 0
        else:
            final_r_index = r_indicies[-1]
        return simulation_data.x1[final_r_index], final_r_index
    
    def inner_loop(i,j):
        t_start = (j-1) * (on_times[i] + off_times[i])
        t_start_next = j * (on_times[i] + off_times[i])
        t_end = t_start + on_times[i]
        
        t_start_ind = int((t_start * (5 / u.Myr)).value)
        t_end_ind = int((t_start_next * (5 / u.Myr)).value)
        if t_end_ind > 1000:
            t_end_ind = 1000
        print('{0}:{1}-{2}'.format(j,t_start_ind, t_end_ind))
        r = range(t_start_ind, t_end_ind, step)
        
        lengths = np.zeros(len(r))
        sim_times = np.zeros_like(lengths)
        
        ind = 0
        for k in r:
            d = ps.load_timestep_data(k, run_dirs[sim_ids[i]])
            l, _ = get_lobe_length(d, trc_cutoff, j)
            lengths[ind] = l
            sim_times[ind] = (d.SimTime * ts[i].value) - t_start.value
            ind+=1
        
        return lengths, sim_times
        
    fig,ax=newfig(width)
    
    for i in range(len(sim_ids)):
        for j in range(1,ntrcs[i]+1):
            lengths, sim_times = inner_loop(i,j)
            
            ax.plot(sim_times, lengths*ls[i], label='{0}, Outburst {1}'.format(labels[i], j))
    
    ax.set_xlabel('Simulation Time (Myr)')
    ax.set_ylabel('Lobe length (kpc)')
    ax.legend(loc='best')
    
    savefig('lobe-length-multiple-tracers-{0}'.format('-'.join(run_codes[sim_ids])), fig)
    plt.close(); 
    
def plot_four_outbursts_tracer(sim_id, ls):
    fig,ax = newfig(1.2, 0.5)
    fig.delaxes(ax)
    timesteps = [50, 300, 550, 800]
    for i,t in enumerate(timesteps):
        ax = fig.add_subplot(1, len(timesteps), i + 1)
        d = ps.load_timestep_data(t, run_dirs[sim_id])

        X1, X2 = ps.sphericaltocartesian(d)
        X1 = X1 * ls.value
        X2 = X2 * ls.value

        with plt.style.context('tracer-plot.mplstyle'):
            im = ax.pcolormesh(X1, X2, getattr(d, 'tr{0}'.format(i+1)).T, vmin=0, vmax=1)
            im.set_rasterized(True)
        ax.set_aspect('equal')
        ax.set_xticks([0,4,8,12])
        ax.axis([0, 15, 0, 60])
        if i+1 == len(timesteps):
            (ca, div, cax) = create_colorbar(im, ax, fig, size='10%', padding=0.05)
            ca.set_label('Outburst tracer value')
    savefig('four-outburst-tracer-comparison-{0}'.format(run_codes[sim_id]),fig)
    
def plot_radial_surface_brightness_pressure(sim_id, timestep, ntrc, ts, ls, ds, redshift=0.1, beamsize=5*u.arcsec,
                           width=2):
    from astropy.cosmology import Planck15 as cosmo
 
    fig,ax=newfig(width)
    
    # calculate beam radius
    sigma_beam = (beamsize / 2.355)
    
    # calculate kpc per arcsec
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(redshift).to(u.kpc / u.arcsec)

    # load timestep data file
    d = ps.load_timestep_data(timestep, run_dirs[sim_id])

    X1, X2 = ps.sphericaltocartesian(d)
    X1 = X1 * (ls / kpc_per_arcsec).to(u.arcmin).value
    X2 = X2 * (ls / kpc_per_arcsec).to(u.arcmin).value

    (r_lumin, flux_d) = get_luminosity(d, 
                                   ds, 
                                   ls, 
                                   ts,
                                   redshift,
                                   beamsize,
                                   ntrc,
                                   calculate_luminosity=False,
                                   convolve_flux=True)

    # plot data
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    line1 = ax.plot(d.x1 * ls, flux_d[:,0], label='Surface brightness')
    print(d.x1.shape)
    ax.set_xlim((0, 400))
    
    
    ax2 = fig.add_subplot(111, sharex=ax, frameon=False)
    
    colour_cycle = ax2._get_lines.prop_cycler
    colour_cycle.next()
    
    line2 = ax2.plot(d.x1 * ls, d.prs[:,0], '-', label='Pressure')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    
    ax2.set_ylabel('Pressure (Pa)')
    ax.set_ylabel(r'Surface brightness (mJy / beam)')
    ax.set_xlabel('Radius (kpc)')
    
    # show legend
    plt.legend(handles=[line1, line2], loc='best')
    
    savefig('radial-surface-brightness-pressure-{0}-{1}'.format(run_codes[sim_id], timestep), fig)
    plt.close();
    
def plot_r_theta_n2_cluster():
    fig,ax = newfig(1, 0.5)
    fig.delaxes(ax)
    
    tstep = 700
    xlim = (100, 300)
    ylim = (0, 1.5)
    yticks = [0, 0.5, 1]

    ax = fig.add_subplot(211)

    d = ps.load_timestep_data(tstep, run_dirs[standard_14p5[1]])
    im = ax.pcolormesh(d.x1r*unit_length_makino_14p5.value, np.rad2deg(d.x2r), np.log10(d.rho.T), vmax=0, vmin=-2.5)
    im.set_rasterized(True)
    ax.text(0.95, 0.85, 'standard', color='white', horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    ax.set_ylim(ylim);
    ax.set_xlim(xlim)
    ax.set_yticks(yticks)
    ax.set_xticklabels([])
    ax.set_ylabel(r'$\theta$')
    ax1 = ax
    im1 = im

    ax = fig.add_subplot(212)

    d = ps.load_timestep_data(tstep, run_dirs[high_res[0]])
    im = ax.pcolormesh(d.x1r*unit_length_makino_14p5.value, np.rad2deg(d.x2r), np.log10(d.rho.T), vmax=0, vmin=-2.5)
    im.set_rasterized(True)
    ax.text(0.95, 0.85, 'high resolution', color='white', horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    ax.set_ylim(ylim);
    ax.set_xlim(xlim)
    ax.set_yticks(yticks)
    ax.set_ylabel(r'$\theta$')
    ax.set_xlabel(r'$r$')

    ax2 = ax
    im2 = im

    fig.subplots_adjust(hspace=0.001)
    ca = fig.colorbar(im1, ax=(ax1, ax2), location='top',pad=0);
    ca.set_ticks([-2.5, -2, -1.5, -1, -0.5, 0]);
    
    savefig('res-comparison-n2-cluster', fig)
    plt.close();


# # Run Plots

# In[26]:

# misc plots

#plot_simulation_grid()
#plot_density_with_simulation_grid()
#plot_concentration_profiles()
#plot_nfw_density_different_concentration_profiles()
#plot_density_profiles()
#plot_velocity_ramp()

# thumbnail morphology plots

# density standard 14p5

#plot_density_thumbnail_grid(standard_14p5[0], unit_time_14p5, unit_length_14p5, unit_density_14p5.to(u.kg / (u.m ** 3)).value, (0,250), (0,250),
#                           -24, -22.5)
#plot_density_thumbnail_grid(standard_14p5[1], unit_time_14p5, unit_length_14p5, unit_density_14p5.to(u.kg / (u.m ** 3)).value, (0,250), (0,250))
#plot_density_thumbnail_grid(standard_14p5[2], unit_time_14p5, unit_length_14p5, unit_density_14p5.to(u.kg / (u.m ** 3)).value, (0,250), (0,250))
#plot_density_thumbnail_grid(standard_14p5[3], unit_time_14p5, unit_length_14p5, unit_density_14p5.to(u.kg / (u.m ** 3)).value, (0,250), (0,250),
#                           -24, -22.5)

# pressure standard 14p5

#plot_pressure_thumbnail_grid(standard_14p5[0], unit_time_14p5, unit_length_14p5, unit_pressure_14p5.to(u.Pa).value, (0,250), (0,250),
#                            -12.2, -10.6)
#plot_pressure_thumbnail_grid(standard_14p5[1], unit_time_14p5, unit_length_14p5, unit_pressure_14p5.to(u.Pa).value, (0,250), (0,250),
#                            -12.2, -10.6)
#plot_pressure_thumbnail_grid(standard_14p5[2], unit_time_14p5, unit_length_14p5, unit_pressure_14p5.to(u.Pa).value, (0,250), (0,250),
#                            -12.2, -10.6)
#plot_pressure_thumbnail_grid(standard_14p5[3], unit_time_14p5, unit_length_14p5, unit_pressure_14p5.to(u.Pa).value, (0,250), (0,250),
#                            -12.2, -10.6)

# density standard 12p5

#plot_density_thumbnail_grid(standard_12p5[0], unit_time_12p5, unit_length_12p5, unit_density_12p5.to(u.kg / (u.m ** 3)).value, (0,250), (0,250),
#                            -25, -22)
#plot_density_thumbnail_grid(standard_12p5[1], unit_time_12p5, unit_length_12p5, unit_density_12p5.to(u.kg / (u.m ** 3)).value, (0,250), (0,250))
#plot_density_thumbnail_grid(standard_12p5[2], unit_time_12p5, unit_length_12p5, unit_density_12p5.to(u.kg / (u.m ** 3)).value, (0,250), (0,250))
#plot_density_thumbnail_grid(standard_12p5[3], unit_time_12p5, unit_length_12p5, unit_density_12p5.to(u.kg / (u.m ** 3)).value, (0,250), (0,250),
#                            -25, -22)

# pressure standard 12p5

#plot_pressure_thumbnail_grid(standard_12p5[0], unit_time_12p5, unit_length_12p5, unit_pressure_12p5.to(u.Pa).value, (0,250), (0,250),
#                            -15.2, -11.4)
#plot_pressure_thumbnail_grid(standard_12p5[1], unit_time_12p5, unit_length_12p5, unit_pressure_12p5.to(u.Pa).value, (0,250), (0,250),
#                            -15.2, -11.4)
#plot_pressure_thumbnail_grid(standard_12p5[2], unit_time_12p5, unit_length_12p5, unit_pressure_12p5.to(u.Pa).value, (0,250), (0,250),
#                            -15.2, -11.4)
#plot_pressure_thumbnail_grid(standard_12p5[3], unit_time_12p5, unit_length_12p5, unit_pressure_12p5.to(u.Pa).value, (0,250), (0,250),
#                            -15.2, -11.4)

# # energy component plots for standard 14p5
# 
# plot_energy_components(standard_14p5[0], unit_energy_makino_14p5, jet_makino_14p5_m25, step=10, dl=True, frameon=True)
# plot_energy_components(standard_14p5[1], unit_energy_14p5, jet_14p5_m25, step=10, dl=False)
# plot_energy_components(standard_14p5[2], unit_energy_14p5, jet_14p5_m25, step=10, dl=False)
# plot_energy_components(standard_14p5[3], unit_energy_makino_14p5, jet_makino_14p5_m25, step=10, dl=True, loc='upper left')
# 
# # energy component plots for standard 12p5
# 
# plot_energy_components(standard_12p5[0], unit_energy_12p5, jet_12p5_m25, step=10, dl=False)
# plot_energy_components(standard_12p5[1], unit_energy_12p5, jet_12p5_m25, step=10, dl=False)
# plot_energy_components(standard_12p5[2], unit_energy_12p5, jet_12p5_m25, step=10, dl=False)
# plot_energy_components(standard_12p5[3], unit_energy_12p5, jet_12p5_m25, step=10, dl=False)

# # energy component plots for fast ramp 12p5
# 
# plot_energy_components(fast_ramp[0], unit_energy_12p5, jet_12p5_m25, step=10, dl=True)

# feedback efficiencies 12p5 vs 14p5
# plot_feedback_efficiency(standard_12p5[0], standard_14p5[0], r'$10^{12.5}M_\odot$', r'$10^{14.5}M_\odot$', unit_time_12p5, unit_time_14p5,
#                         1, 1, step=10)

# feedback efficiency 14p5 n1 vs n4
# plot_feedback_efficiency(standard_14p5[0], standard_14p5[3], r'$n=1$', r'$n=4$', unit_time_makino_14p5, unit_time_makino_14p5,
#                          1, 4, step=10)
# 
# # feedback efficiency 12p5 n1 vs n4
# plot_feedback_efficiency(standard_12p5[0], standard_12p5[3], r'$n=1$', r'$n=4$', unit_time_makino_12p5, unit_time_makino_12p5,
#                         1, 4, step=10, draw_legend=False)

# # feedback efficiency 14p5 n2 vs n3
# plot_feedback_efficiency(standard_14p5[1], standard_14p5[2], r'$n=2$', r'$n=3$', unit_time_14p5, unit_time_14p5,
#                         2, 3, step=10)
# 
# # feedback efficiency 12p5 n2 vs n3
# plot_feedback_efficiency(standard_12p5[1], standard_12p5[2], r'$n=2$', r'$n=3$', unit_time_12p5, unit_time_12p5,
#                         2, 3, step=10)

# feedback efficiencies 12p5 vs 14p5 various cutoffs
# for i in np.arange(0.001,0.01,0.001):
#     plot_feedback_efficiency_different_tracer_cutoff(standard_12p5[0], standard_14p5[0], r'$10^{12.5}M_\odot$', 
#                                                  r'$10^{14.5}M_\odot$', unit_time_makino_12p5, 
#                                                  unit_time_makino_14p5,
#                                                  1, 1, step=10, trc_cutoff=i)

# plot_feedback_efficiency_different_tracer_cutoff_grid(standard_12p5[0], standard_14p5[0], r'$10^{12.5}M_\odot$', 
#                                                      r'$10^{14.5}M_\odot$', unit_time_makino_12p5, 
#                                                      unit_time_makino_14p5,
#                                                      1, 1, np.arange(0.001,0.010,0.001), step=10)
# 
# plot_feedback_efficiency_different_tracer_cutoff_grid(standard_12p5[0], standard_12p5[3], r'$n=1$', 
#                                                      r'$n=4$', unit_time_makino_12p5, 
#                                                      unit_time_makino_12p5,
#                                                      1, 4, np.arange(0.001,0.010,0.001), step=10)
# 
# plot_feedback_efficiency_different_tracer_cutoff_grid(standard_14p5[0], standard_14p5[3], r'$n=1$', 
#                                                      r'$n=4$', unit_time_makino_14p5, 
#                                                      unit_time_makino_14p5,
#                                                      1, 4, np.arange(0.001,0.010,0.001), step=10)

# surface brightness

# surface brightness m14.5-M25-n1
#plot_surface_brightness(standard_14p5[0], 200, 1, unit_time_14p5, unit_length_14p5, unit_density_14p5, xlim=(-0.67, 0.67), ylim=(-1, 1))

# surface brightness m12.5-M25-n1
#plot_surface_brightness(standard_12p5[0], 200, 1, unit_time_12p5, unit_length_12p5, unit_density_12p5, xlim=(-0.35, 0.35), ylim=(-0.5, 0.5))

# # surface brightness m14.5-M25 at t=1000
# plot_surface_brightness(standard_14p5[0], 1000, 1, unit_time_14p5, unit_length_14p5, unit_density_14p5, xlim=(-0.75, 0.75), ylim=(-2.5, 2.5))
# plot_surface_brightness(standard_14p5[1], 1000, 2, unit_time_14p5, unit_length_14p5, unit_density_14p5, xlim=(-0.75, 0.75), ylim=(-2.5, 2.5))
# plot_surface_brightness(standard_14p5[2], 1000, 3, unit_time_14p5, unit_length_14p5, unit_density_14p5, xlim=(-0.75, 0.75), ylim=(-2.5, 2.5))
# plot_surface_brightness(standard_14p5[3], 1000, 4, unit_time_14p5, unit_length_14p5, unit_density_14p5, xlim=(-0.75, 0.75), ylim=(-2.5, 2.5))
# 
# # surface brightness m12.5-M25 at t=1000
# plot_surface_brightness(standard_12p5[0], 1000, 1, unit_time_12p5, unit_length_12p5, unit_density_12p5, xlim=(-0.5625, 0.5625), ylim=(-1.875, 1.875))
# plot_surface_brightness(standard_12p5[1], 1000, 2, unit_time_12p5, unit_length_12p5, unit_density_12p5, xlim=(-0.5625, 0.5625), ylim=(-1.875, 1.875))
# plot_surface_brightness(standard_12p5[2], 1000, 3, unit_time_12p5, unit_length_12p5, unit_density_12p5, xlim=(-0.5625, 0.5625), ylim=(-1.875, 1.875))
# plot_surface_brightness(standard_12p5[3], 1000, 4, unit_time_12p5, unit_length_12p5, unit_density_12p5, xlim=(-0.5625, 0.5625), ylim=(-1.875, 1.875))

# # p-d tracks for n=1, m12.5 and m14.5
# plot_pd_tracks(standard_12p5[0], standard_14p5[0], r'$10^{12.5}M_\odot$', r'$10^{14.5}M_\odot$', range(1,200,1), range(1,200,1), 1, 1,
#               unit_time_12p5, unit_time_14p5, unit_length_12p5, unit_length_14p5, unit_density_12p5, unit_density_14p5)

# # p-d tracks for n=4, m12.5 and m14.5
# plot_pd_tracks(standard_12p5[3], standard_14p5[3], r'$10^{12.5}M_\odot$', r'$10^{14.5}M_\odot$', 
#                range(1,200,1) + range(200,1000,10),
#                range(1,200,1) + range(200,1000,10),
#                4, 4,
#                unit_time_makino_12p5, unit_time_makino_14p5, unit_length_makino_12p5, 
#                unit_length_makino_14p5, unit_density_makino_12p5, unit_density_makino_14p5,
#                xlim=(0,275),ylim=(10 ** 23.5, 10 ** 26.5),
#                xticks=[2, 10, 100, 200], xtick_labels=['2', '10', '100', '200'],
#                yticks=[np.float(10**23.5), np.float(10**24.5), np.float(10**25.5), np.float(10**26.5)],
#                ytick_labels=[r'$10^{23.5}$', r'$10^{24.5}$', r'$10^{25.5}$', r'$10^{26.5}$'],
#                legend_loc='lower left')

# plot resolution test
#plot_resolution_test()

# # plot lobe lenghts for n=1 14.5 and 12.5
# plot_lobe_lengths([standard_12p5[0], standard_14p5[0]], [r'$10^{12.5}M_\odot$', r'$10^{14.5}M_\odot$'], [1,1], range(1,1000,10), 
#                 [unit_time_12p5, unit_time_14p5], [unit_length_12p5, unit_length_14p5])

# # plot lobe lenghts for n=1 12.5 fast ramp
# plot_lobe_lengths([standard_12p5[0], fast_ramp[0]], [r'Regular', r'Fast ramp'], [1,1], range(1,1000,10), 
#                 [unit_time_makino_12p5]*2, [unit_length_makino_12p5]*2, width=0.5)
# 
# # plot lobe lenghts for n=4 12.5 fast ramp
# plot_lobe_lengths([standard_12p5[3], fast_ramp[1]], [r'Regular', r'Fast ramp'], [4,4], range(1,500,10), 
#                 [unit_time_makino_12p5]*2, [unit_length_makino_12p5]*2, width=0.5)
# 
# # plot lobe lengths for n=1 and n=4 12.5 fast ramp
# plot_lobe_lengths([standard_12p5[0], fast_ramp[0], standard_12p5[3], fast_ramp[1]], 
#                   [r'Regular: $n=1$', r'Fast ramp: $n=1$', r'Regular: $n=4$', r'Fast ramp: $n=4$'], [1,1, 4, 4], range(1,500,10), 
#                 [unit_time_12p5]*4, [unit_length_12p5]*4)

# # plot lobe lengths for n=1 12.5 close injection
# plot_lobe_lengths([standard_12p5[0], close_injection[0]], [r'Regular', r'Close injection'], [1,1], range(1,100,5), 
#                 [unit_time_12p5]*2, [unit_length_12p5]*2, width=0.5)
# 
# # plot lobe lengths for n=4 12.5 close injection
# plot_lobe_lengths([standard_12p5[3], close_injection[1]], [r'Regular', r'Close injection'], [4,4], range(1,260,5), 
#                 [unit_time_12p5]*2, [unit_length_12p5]*2, width=0.5)

# # plot lobe lengths for n=1 and n=4 12.5 close injection all timesteps
# plot_lobe_lengths_different_timesteps([standard_12p5[0], close_injection[0], standard_12p5[3], close_injection[1]], 
#                   [r'Regular: $n=1$', r'Close injection: $n=1$', r'Regular: $n=4$', r'Close injection: $n=4$'], [1,1, 4, 4], 
#                                       len(range(1,260,10)),
#                                       [range(1,100,10), range(1, 100, 10), range(1, 260, 10), range(1, 260, 10)],
#                 [unit_time_12p5]*4, [unit_length_12p5]*4)

# # plot lobe lengths for n=1 and n=4 12.5 close injection clamped to shortest timestep
# plot_lobe_lengths([standard_12p5[0], close_injection[0], standard_12p5[3], close_injection[1]], 
#                   [r'Regular: $n=1$', r'Close injection: $n=1$', r'Regular: $n=4$', r'Close injection: $n=4$'], [1,1, 4, 4], 
#                                       range(1,100,1),
#                 [unit_time_12p5]*4, [unit_length_12p5]*4)

# # plot lobe lengths for m14.5-M25 runs
# plot_lobe_lengths(standard_14p5, ['$n=1$', '$n=2$', '$n=3$', '$n=4$'], [1,2,3,4], range(1,1000,10), 
#                 [unit_time_14p5]*4, [unit_length_14p5]*4)
# 
# # plot lobe lengths for m12.5-M25 runs
# plot_lobe_lengths(standard_12p5, ['$n=1$', '$n=2$', '$n=3$', '$n=4$'], [1,2,3,4], range(1,1000,10), 
#                 [unit_time_12p5]*4, [unit_length_12p5]*4)

# Lobe lengths different cutoffs

# # plot lobe lengths for n=1 m12.5 and m14.5, different tracer cutoffs
# plot_lobe_lengths_diff_cutoff([standard_12p5[0], standard_14p5[0]], [r'$10^{12.5}M_\odot$', r'$10^{14.5}M_\odot$'], [1,1], range(1,1000,10), 
#                 [unit_time_makino_12p5, unit_time_makino_14p5], [unit_length_makino_12p5, unit_length_makino_14p5], trc_cutoff=0.01)
# 
# plot_lobe_lengths_diff_cutoff([standard_12p5[0], standard_14p5[0]], [r'$10^{12.5}M_\odot$', r'$10^{14.5}M_\odot$'], [1,1], range(1,1000,10), 
#                 [unit_time_makino_12p5, unit_time_makino_14p5], [unit_length_makino_12p5, unit_length_makino_14p5], trc_cutoff=0.1)
# 
# plot_lobe_lengths_diff_cutoff([standard_12p5[0], standard_14p5[0]], [r'$10^{12.5}M_\odot$', r'$10^{14.5}M_\odot$'], [1,1], range(1,1000,10), 
#                 [unit_time_makino_12p5, unit_time_makino_14p5], [unit_length_makino_12p5, unit_length_makino_14p5], trc_cutoff=0.001)

# # lobe lengths for individual outbursts
# plot_lobe_lengths_multiple_tracers([standard_14p5[3]], [r'$10^{14.5}M_\odot$'], [4], [10 * u.Myr], [40 * u.Myr], 10,
#                                    [unit_time_makino_14p5], [unit_length_makino_14p5])

# lobe volumes

# # plot lobe volumes for m12.5 and m14.5 n=1
# plot_lobe_volumes([standard_12p5[0], standard_14p5[0]], [r'$10^{12.5}M_\odot$', r'$10^{14.5}M_\odot$'], [1,1], range(1,1000,10), 
#                 [unit_time_makino_12p5, unit_time_makino_14p5], [unit_length_makino_12p5, unit_length_makino_14p5],
#                  width=0.5)

# # plot lobe volumes for all m14.5 runs
# plot_lobe_volumes(standard_14p5, ['$n=1$', '$n=2$', '$n=3$', '$n=4$'], [1,2,3,4], range(1,1000,10), 
#                 [unit_time_14p5]*4, [unit_length_14p5]*4)
# 
# # plot lobe volumes for all m12.5 runs
# plot_lobe_volumes(standard_12p5, ['$n=1$', '$n=2$', '$n=3$', '$n=4$'], [1,2,3,4], range(1,1000,10), 
#                 [unit_time_12p5]*4, [unit_length_12p5]*4)

# lobe widths

# # plot lobe widths for m12.5 and m14.5 n=1
# plot_lobe_widths_slow([standard_12p5[0], standard_14p5[0]], [r'$10^{12.5}M_\odot$', r'$10^{14.5}M_\odot$'], [1,1], range(1,1000,100), 
#                 [unit_time_12p5, unit_time_14p5], [unit_length_12p5, unit_length_14p5])

# Morphology comparison
# plot_morphology_comparisons('rho', 
#                             [unit_density_makino_14p5.to(u.kg / (u.m ** 3)).value, 
#                              unit_density_makino_12p5.to(u.kg / (u.m ** 3)).value],
#                             r'Density ($\log_{10}{kg / m^3})$', 'density-plot.mplstyle', timestep=100, log=True)
# 
# plot_morphology_comparisons('vx1', 
#                             [unit_speed_makino_14p5.value, unit_speed_makino_12p5.value],
#                             r'Radial velocity (km / s)', 'radial-velocity-plot.mplstyle', timestep=100, log=False)
# 
# plot_morphology_comparisons('vx2', 
#                             [unit_speed_makino_14p5.value, unit_speed_makino_12p5.value],
#                             r'Azimuthal velocity (km / s)', 'azimuthal-velocity-plot.mplstyle', timestep=100, log=False)
# 
# plot_morphology_comparisons('prs', 
#                             [unit_pressure_makino_14p5.to(u.Pa).value, unit_pressure_makino_12p5.to(u.Pa).value],
#                             r'Pressure ($\log_{10}{Pa}$)', 'pressure-plot.mplstyle', timestep=100, log=True)

# Lobe masses

# # plot lobe masses for m12.5 and m14.5 n=1
# plot_lobe_masses([standard_12p5[0], standard_14p5[0]], [r'$10^{12.5}M_\odot$', r'$10^{14.5}M_\odot$'], [1,1], range(1,1000,10), 
#                  [unit_time_makino_12p5, unit_time_makino_14p5], [unit_length_makino_12p5, unit_length_makino_14p5], 
#                  [unit_density_makino_12p5, unit_density_makino_14p5], width=0.5)

# Four outburst tracer comparisons
#plot_four_outbursts_tracer(standard_14p5[3], unit_length_makino_14p5)
#plot_four_outbursts_tracer(standard_12p5[3], unit_length_makino_12p5)

# radial surface-brightness and pressure
# plot_radial_surface_brightness_pressure(standard_12p5[3], 1000, 4, unit_time_makino_12p5, unit_length_makino_12p5,
#                                         unit_density_makino_12p5, width=1)

# r theta n2 cluster
#plot_r_theta_n2_cluster()

# king profiles

# density thumbnails

plot_density_thumbnail_grid(king_runs_14p5[0], unit_time_king_14p5, unit_length_king_14p5, 
                            unit_density_king_14p5.to(u.kg / (u.m ** 3)).value, (0,300), (0,300),
                            -26, -23)
# 
# plot_density_thumbnail_grid(king_runs_14p5[1], unit_time_king_14p5, unit_length_king_14p5, 
#                             unit_density_king_14p5.to(u.kg / (u.m ** 3)).value, (0,300), (0,300),
#                             -26, -24)
# 
# plot_density_thumbnail_grid(king_runs_12p5[0], unit_time_king_12p5, unit_length_king_12p5, 
#                             unit_density_king_12p5.to(u.kg / (u.m ** 3)).value, (0,300), (0,300),
#                             -26.5, -23.5)
# 
# plot_density_thumbnail_grid(king_runs_12p5[1], unit_time_king_12p5, unit_length_king_12p5, 
#                             unit_density_king_12p5.to(u.kg / (u.m ** 3)).value, (0,300), (0,300),
#                             -26.5, -23.5)
# 
# # pressure thumbnails
# 
# plot_pressure_thumbnail_grid(king_runs_14p5[0], unit_time_king_14p5, unit_length_king_14p5, unit_pressure_king_14p5.to(u.Pa).value,
#                              (0,300), (0,300),
#                              -13, -12)
# 
# plot_pressure_thumbnail_grid(king_runs_14p5[1], unit_time_king_14p5, unit_length_king_14p5, unit_pressure_king_14p5.to(u.Pa).value,
#                              (0,300), (0,300),
#                              -13, -12)
# 
# plot_pressure_thumbnail_grid(king_runs_12p5[0], unit_time_king_12p5, unit_length_king_12p5, unit_pressure_king_12p5.to(u.Pa).value,
#                              (0,300), (0,300),
#                              -14.5, -13)
# 
# plot_pressure_thumbnail_grid(king_runs_12p5[1], unit_time_king_12p5, unit_length_king_12p5, unit_pressure_king_12p5.to(u.Pa).value,
#                              (0,300), (0,300),
#                              -14.5, -13)

# lobe lengths
# plot lobe lengths for m14.5 and m12.5 king
# plot_lobe_lengths(king_all, ['$n=1$, cluster', '$n=4$, cluster', '$n=1$, poor group', '$n=4$, poor group'], [1,4,1,4],
#                   range(1,1000,10), 
#                 [unit_time_king_14p5]*2 + [unit_time_king_12p5]*2, [unit_length_king_14p5]*2 + [unit_length_king_12p5]*2)

# energy components m14.5 and m12.5 king

# plot_energy_components(king_runs_14p5[0], unit_energy_king_14p5, jet_king_14p5_m25, step=10, dl=True, frameon=True)
# plot_energy_components(king_runs_14p5[1], unit_energy_king_14p5, jet_king_14p5_m25, step=10, dl=False)
# 
# plot_energy_components(king_runs_12p5[0], unit_energy_king_12p5, jet_king_12p5_m25, step=10, dl=False)
# plot_energy_components(king_runs_12p5[1], unit_energy_king_12p5, jet_king_12p5_m25, step=10, dl=False)
# 
# # surface brightness m14.5 and m12.5 king
# plot_surface_brightness(king_runs_14p5[0], 200, 1, unit_time_king_14p5, unit_length_king_14p5, 
#                         unit_density_king_14p5, xlim=(-1, 1), ylim=(-1, 1))
# 
# plot_surface_brightness(king_runs_12p5[0], 200, 1, unit_time_king_12p5, unit_length_king_12p5, 
#                         unit_density_king_12p5, xlim=(-1, 1), ylim=(-1, 1))


# In[ ]:



