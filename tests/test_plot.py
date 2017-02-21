import pytest
import os
from plutokore import plot

PLUTO_FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'data', )


@pytest.mark.datafiles(
    os.path.join(PLUTO_FIXTURE_DIR, 'pluto'), keep_top_dir=True)
def test_get_data_direct(datafiles):
    from plutokore import io
    assert len(datafiles.listdir()) == 1
    path = str(datafiles.listdir()[0])
    data = io.pload(0, w_dir=path)
    plot.get_pluto_data_direct(data, 'rho', True, path, 0)
    plot.get_pluto_data_direct(data, 'rho', False, path, 0)
    plot.get_pluto_data_direct_no_log(data, 'rho', True, path, 0)

def test_figsize_equal_ratio():
    fs = plot.figsize(1, 1)
    assert fs[0] == fs[1]  # equal ratio


def test_figsize_no_ratio():
    fs = plot.figsize(1)
    assert fs[0] != fs[1]  # equal ratio
    assert fs[0] > 0
    assert fs[1] > 0


def test_figsize_diff_ratio():
    r = 0.85
    fs = plot.figsize(1, ratio=r)
    assert fs[0] * r == fs[1]


def test_new_fig():
    import matplotlib
    matplotlib.use('Agg')
    f, a = plot.newfig(1, 1)
    assert f is not None
    assert a is not None


def test_save_fig():
    import matplotlib
    matplotlib.use('Agg')
    f, a = plot.newfig(1, 1)
    assert f is not None
    assert a is not None


