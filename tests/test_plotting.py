import pytest
import os
from plutokore import plotting

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
    plotting.get_pluto_data_direct(data, 'rho', True, path, 0)
    plotting.get_pluto_data_direct(data, 'rho', False, path, 0)
    plotting.get_pluto_data_direct_no_log(data, 'rho', True, path, 0)
