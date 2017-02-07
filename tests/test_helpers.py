import pytest
import os
from plutokore import helpers

PLUTO_FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'data', )


def test_calculate_unit_values(astro_jet, makino_env):
    helpers.get_unit_values(makino_env, astro_jet)


def test_suppress_stdout():
    with helpers.suppress_stdout():
        print('This does not show')


def test_print_md():
    helpers.printmd('test')


def test_figsize_equal_ratio():
    fs = helpers.figsize(1, 1)
    assert fs[0] == fs[1]  # equal ratio


def test_figsize_no_ratio():
    fs = helpers.figsize(1)
    assert fs[0] != fs[1]  # equal ratio
    assert fs[0] > 0
    assert fs[1] > 0


def test_figsize_diff_ratio():
    r = 0.85
    fs = helpers.figsize(1, ratio=r)
    assert fs[0] * r == fs[1]


def test_new_fig():
    import matplotlib
    matplotlib.use('Agg')
    f, a = helpers.newfig(1, 1)
    assert f is not None
    assert a is not None


def test_save_fig():
    import matplotlib
    matplotlib.use('Agg')
    f, a = helpers.newfig(1, 1)
    assert f is not None
    assert a is not None


@pytest.mark.datafiles(
    os.path.join(PLUTO_FIXTURE_DIR, 'pluto'), keep_top_dir=True)
def test_get_last_timestep(datafiles):
    assert len(datafiles.listdir()) == 1
    path = str(datafiles.listdir()[0])
    lt = helpers.get_last_timestep(path)
    assert lt == 1003
