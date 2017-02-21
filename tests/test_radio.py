import pytest
import os
from plutokore import radio

slow = pytest.mark.skipif(
    not pytest.config.getoption('--runslow'),
    reason='need --runslow option to run'
)

pytestmark = slow

PLUTO_FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'data', )


@pytest.mark.datafiles(
    os.path.join(PLUTO_FIXTURE_DIR, 'pluto'), keep_top_dir=True)
def test_get_luminosity_old_nonconvolved(datafiles, makino_env_12p5, jet_12p5):
    from plutokore import jet
    from plutokore import io
    from astropy import units as u
    assert len(datafiles.listdir()) == 1
    path = str(datafiles.listdir()[0])
    data = io.pload(0, w_dir=path)

    z = 0.1
    beam_width = 5 * u.arcsec
    ntracers = 4

    uv = jet.get_unit_values(makino_env_12p5, jet_12p5)
    (l, f) = radio.get_luminosity_old(data, uv.density, uv.length,
                                           uv.time, z, beam_width, ntracers)


@pytest.mark.datafiles(
    os.path.join(PLUTO_FIXTURE_DIR, 'pluto'), keep_top_dir=True)
def test_get_luminosity_old_convolved(datafiles, makino_env_12p5, jet_12p5):
    from plutokore import jet
    from plutokore import io
    from astropy import units as u
    assert len(datafiles.listdir()) == 1
    path = str(datafiles.listdir()[0])
    data = io.pload(0, w_dir=path)

    z = 0.1
    beam_width = 5 * u.arcsec
    ntracers = 4

    uv = jet.get_unit_values(makino_env_12p5, jet_12p5)
    (l, f) = radio.get_luminosity_old(
        data,
        uv.density,
        uv.length,
        uv.time,
        z,
        beam_width,
        ntracers,
        convolve_flux=True)


@pytest.mark.datafiles(
    os.path.join(PLUTO_FIXTURE_DIR, 'pluto'), keep_top_dir=True)
def test_get_surface_brightness(datafiles, makino_env_12p5, jet_12p5):
    from plutokore import jet
    from plutokore import io
    from astropy import units as u
    assert len(datafiles.listdir()) == 1
    path = str(datafiles.listdir()[0])
    data = io.pload(0, w_dir=path)

    z = 0.1
    beam_width = 5 * u.arcsec
    ntracers = 4

    uv = jet.get_unit_values(makino_env_12p5, jet_12p5)
    l = radio.get_luminosity(data, uv, z, beam_width)
    f = radio.get_flux_density(l, z)
    fc = radio.get_convolved_flux_density(f, z, beam_width)
    sb = radio.get_surface_brightness(f, data, uv, z, beam_width)
