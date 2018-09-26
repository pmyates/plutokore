from builtins import object
import numpy as _np
import astropy.units as _u
import astropy.constants as _const
import astropy.cosmology as _cosmology
import scipy.integrate as _integrate


class ObservationalKingProfile(object):

    def __init__(self,
                 *,
                 redshift,
                 T,
                 beta,
                 central_density,
                 core_radius,
                 gamma=None,
                 mu=None,
                 omega_b=None,
                 cosmo=None):

        """Creates a King profile for the given halo mass and redshift"""

        self.redshift = redshift
        self.T = T
        self.beta = beta
        self.central_density = central_density
        self.core_radius = core_radius

        if mu is None:
            mu = 0.6
        self.mu = mu

        if cosmo is None:
            cosmo = _cosmology.Planck15
        self.cosmo = cosmo

        if omega_b is None:
            omega_b = self.cosmo.Ob(self.redshift) / self.cosmo.Om(
                self.redshift)
        self.omega_b = omega_b

        if gamma is None:
            gamma = 5.0 / 3.0
        self.gamma = gamma

        # calculate sound speed
        self.sound_speed = self.calculate_sound_speed(self.gamma, self.T,
                                                      self.mu)

    def get_density(self, r):
        return self.calculate_density_at_radius(self.central_density,
                                                self.beta, self.core_radius, r)

    @staticmethod
    def calculate_density_at_radius(central_density, beta, core_radius, r):
        return central_density * _np.power(1 + ((r / core_radius)**2),
                                           -3.0 * beta / 2.0)
        #return central_density * _np.exp(-delta_nfw)*(1+r/r_s) ** (delta_nfw/(r/r_s))

    @staticmethod
    def calculate_sound_speed(gamma, T, mu):
        return _np.sqrt((gamma * T) / (mu * _const.m_p)).to(_u.km / _u.s)
