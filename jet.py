import numpy as _np
import astropy.units as _u
from tabulate import tabulate as _tabulate

class AstroJet:
    """Basic class for calculating relevant astrophyiscal jet quantities, as shown in Krause+ (2012) and Alexander (2006)"""
    
    def __init__(self, opening_angle, ext_mach_number, ext_sound_speed, ext_density, jet_power, gamma):
        """Create a jet with the specified half opening angle,
        external Mach number, external sound speed and external density
        
        Keyword arguments:
        opening_angle -- the half opening angle of the jet, in degrees
        ext_mach_number -- the external Mach number of the jet
        ext_sound_speed -- the external sound speed of the medium
        ext_density -- the external core density of the medium
        jet_power -- the power of the jet (Q)
        gamma -- gamma
        """
        
        self.theta = _np.deg2rad(opening_angle)
        self.M_x = ext_mach_number
        self.c_x = ext_sound_speed
        self.rho_0 = ext_density
        self.Q = jet_power
        self.gamma = gamma
        
        self.calculate_length_scales()
        
    def calculate_length_scales(self):
        self.v_jet = self.get_V_jet()
        self.L_1 = self.calculate_L_1()
        self.L_2 = self.calculate_L_2()
        self.omega = self.get_omega()
        self.L_1a = self.get_L_1_a()
        self.L_1b = self.get_L_1_b()
        self.L_1c = self.get_L_1_c()
        self.time_scaling = (self.L_1 / self.c_x).to(u.Myr)
        self.length_scaling = self.L_1
        
        
    # (semi)private methods
    def calculate_L_1(self):
        return 2*_np.sqrt(2)*_np.sqrt(self.Q/(self.rho_0 * (self.v_jet ** 3))).to(_u.kpc)
    def calculate_L_2(self):
        return _np.sqrt(self.Q/(self.rho_0 * (self.c_x ** 3))).to(_u.kpc)
    def get_omega(self):
        return 2.0 * _np.pi * (1-_np.cos(self.theta)) 
    def get_L_1_b(self):
        return _np.sqrt(1.0/(4.0 * (self.Omega))) * self.L_1 
    def get_L_1_a(self):
        return _np.sqrt((self.gamma / (4 * self.Omega)) * (self.M_x ** 2) * (_np.sin(self.theta) ** 2)) * self.L_1 
    def get_L_1_c(self):
        return _np.sqrt((self.gamma / (4 * self.Omega)) * (self.M_x ** 2)) * self.L_1
    def get_v_jet(self):
        return self.M_x * self.c_x
    
    def calculate_Q():
        return ((1.0/8.0) * self.M_x ** 3 * self.L_1 ** 2 * self.rho_0 * self.c_x ** 3).to(_u.W)
    
    def get_calculated_parameter_table(self):
        jet_calculated_data = [[r'$\Omega$', self.omega], [r'$v_{jet}$', self.v_jet], [r'$L_1$', self.L_1], [r'$L_2$', self.L_2],
                               [r'$L_{1a}$', self.L_1a], [r'$L_{1b}$', self.L_1b], [r'$L_{1c}$', self.L_1c]]
        jet_calculated_headings = ['Calculated Jet Parameter', 'Value']
        return _tabulate(jet_calculated_data, headers=jet_calculated_headings, tablefmt='pipe')
