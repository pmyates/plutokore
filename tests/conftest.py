import pytest
from astropy import units as u
from astropy import cosmology
from plutokore import jet

pytest_plugins = 'pytest_datafiles'

@pytest.fixture(params=[12.5, 14.5])
def makino_env(request):
    from plutokore.environments.makino import MakinoProfile

    mass = (10 ** request.param) * u.M_sun
    z = 0
    return MakinoProfile(mass, z, delta_vir=200, cosmo=cosmology.Planck15, concentration_method='klypin-planck-relaxed')

@pytest.fixture(params=[12.5, 14.5])
def king_env(request):
    from plutokore.environments.king import KingProfile

    mass = (10 ** request.param) * u.M_sun
    z = 0
    return KingProfile(mass, z, delta_vir=200, cosmo=cosmology.Planck15, concentration_method='klypin-planck-relaxed')

@pytest.fixture()
def astro_jet(makino_env):
    theta_deg = 15
    M_x = 25
    Q = 1e37 * u.W
    return jet.AstroJet(theta_deg, M_x, makino_env.sound_speed, makino_env.central_density, Q, makino_env.gamma)
