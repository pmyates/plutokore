import pytest
import os
from plutokore import io

PLUTO_FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'data',
)

@pytest.mark.datafiles(
    os.path.join(PLUTO_FIXTURE_DIR, 'pluto'),
    keep_top_dir = True
)
def test_load_data_file(datafiles):
    assert len(datafiles.listdir()) == 1
    path = datafiles.listdir()[0]
    data = io.pload(0, w_dir=path)

@pytest.mark.datafiles(
    os.path.join(PLUTO_FIXTURE_DIR, 'pluto'),
    keep_top_dir = True
)
def test_load_pluto_times(datafiles, tmpdir):
    assert len(datafiles.listdir()) == 1
    path = datafiles.listdir()[0]
    info = io.nlast_info(w_dir=path)
    assert info['nlast'] == 1003
    assert info['time'] - 24.0 < 0.01
