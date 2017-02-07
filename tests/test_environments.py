import pytest


def test_makino_env_creation(makino_env):
    assert makino_env is not None


def test_king_env_creation(king_env):
    assert king_env is not None


def test_default_viriality():
    from plutokore.environments.makino import MakinoProfile
    from plutokore.environments.king import KingProfile
    from astropy import units as u
    from astropy import cosmology

    mass = (10**12.5) * u.M_sun
    z = 0
    mp = MakinoProfile(
        mass,
        z,
        cosmo=cosmology.Planck15,
        concentration_method='klypin-planck-relaxed')
    kp = KingProfile(
        mass,
        z,
        cosmo=cosmology.Planck15,
        concentration_method='klypin-planck-relaxed')

    assert mp.delta_vir == 200
    assert kp.delta_vir == 200


def test_concentration_methods():
    from plutokore.environments.makino import MakinoProfile
    from plutokore.environments.king import KingProfile
    from astropy import units as u
    from astropy import cosmology

    conc_methods = [
        'dolag', 'bullock', 'klypin-planck-all', 'klypin-planck-relaxed',
        'klypin-wmap-all', 'klypin-wmap-relaxed', 'dutton', 'maccio'
    ]

    mass = (10**12.5) * u.M_sun
    z = 0

    for cm in conc_methods:
        mp = MakinoProfile(
            mass, z, cosmo=cosmology.Planck15, concentration_method=cm)
        kp = KingProfile(
            mass, z, cosmo=cosmology.Planck15, concentration_method=cm)

        assert mp.concentration != 0.0
        assert kp.concentration != 0.0


def test_wrong_concentration_method():
    from plutokore.environments.makino import MakinoProfile
    from plutokore.environments.king import KingProfile
    from astropy import units as u
    from astropy import cosmology

    try:
        mass = (10**12.5) * u.M_sun
        z = 0
        mp = MakinoProfile(
            mass, z, cosmo=cosmology.Planck15, concentration_method='invalid')
        assert 0
    except ValueError:
        pass
    try:
        kp = KingProfile(
            mass, z, cosmo=cosmology.Planck15, concentration_method='invalid')
        assert 0
    except ValueError:
        pass


def test_default_cosmology():
    from plutokore.environments.makino import MakinoProfile
    from plutokore.environments.king import KingProfile
    from astropy import units as u
    from astropy import cosmology

    mass = (10**12.5) * u.M_sun
    z = 0
    mp = MakinoProfile(mass, z, concentration_method='klypin-planck-relaxed')
    kp = KingProfile(mass, z, concentration_method='klypin-planck-relaxed')

    assert mp.cosmo is cosmology.Planck15
    assert kp.cosmo is cosmology.Planck15


def test_default_conc_method():
    from plutokore.environments.makino import MakinoProfile
    from plutokore.environments.king import KingProfile
    from astropy import units as u
    from astropy import cosmology

    mass = (10**12.5) * u.M_sun
    z = 0
    mp = MakinoProfile(mass, z)
    kp = KingProfile(mass, z)

    assert mp.concentration != 0.0
    assert kp.concentration != 0.0


def test_get_king_density(king_env):
    from astropy import units as u
    d = king_env.get_density(1 * u.Mpc)
    assert d > 0
