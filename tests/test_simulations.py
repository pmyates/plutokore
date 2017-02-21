import pytest
import os
from plutokore import simulations

PLUTO_FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'data', )

@pytest.mark.datafiles(
    os.path.join(PLUTO_FIXTURE_DIR, 'pluto'), keep_top_dir=True)
def test_get_last_timestep(datafiles):
    assert len(datafiles.listdir()) == 1
    path = str(datafiles.listdir()[0])
    lt = simulations.get_last_timestep(path)
    assert lt == 1003
