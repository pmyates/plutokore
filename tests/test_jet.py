#from .context.plutokore import jet
#from .context import plutokore
import pytest
from plutokore import jet

def test_jet_creation_makino(makino_env):
    from astropy import units as u
    theta_deg = 15
    M_x = 25
    Q = 1e37 * u.W
    test_jet = jet.AstroJet(theta_deg, M_x, makino_env.sound_speed, makino_env.central_density, Q, makino_env.gamma)

def test_jet_creation_king(king_env):
    from astropy import units as u
    theta_deg = 15
    M_x = 25
    Q = 1e37 * u.W
    test_jet = jet.AstroJet(theta_deg, M_x, king_env.sound_speed, king_env.central_density, Q, king_env.gamma)

def test_calculate_length_scales(astro_jet):
    astro_jet.calculate_length_scales()

def test_get_q(astro_jet):
    astro_jet.calculate_Q()

def test_get_param_table(astro_jet):
    astro_jet.get_calculated_parameter_table()
