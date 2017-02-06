import pytest
from plutokore import helpers

def test_calculate_unit_values(astro_jet, makino_env):
    helpers.get_unit_values(makino_env, astro_jet)
