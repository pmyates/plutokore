import pytest
import os
from plutokore import configuration

PLUTO_FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'data', )


@pytest.mark.datafiles(
    os.path.join(PLUTO_FIXTURE_DIR, 'makino-config-files'), keep_top_dir=True)
def test_validate_yaml_makino(datafiles):
    assert len(datafiles.listdir()) == 1
    path = str(datafiles.listdir()[0])

    yaml_file = os.path.join(path, 'config.yaml')
    ini_file = os.path.join(path, 'pluto.ini')
    definition_file = os.path.join(path, 'definitions.h')

    configuration.validate_yaml(yaml_file, ini_file, definition_file)

@pytest.mark.datafiles(
    os.path.join(PLUTO_FIXTURE_DIR, 'king-config-files'), keep_top_dir=True)
def test_validate_yaml_king(datafiles):
    assert len(datafiles.listdir()) == 1
    path = str(datafiles.listdir()[0])

    yaml_file = os.path.join(path, 'config.yaml')
    ini_file = os.path.join(path, 'pluto.ini')
    definition_file = os.path.join(path, 'definitions.h')

    configuration.validate_yaml(yaml_file, ini_file, definition_file)

@pytest.mark.datafiles(
    os.path.join(PLUTO_FIXTURE_DIR, 'makino-config-files'), keep_top_dir=True)
def test_create_yaml_template_makino(datafiles):
    assert len(datafiles.listdir()) == 1
    path = str(datafiles.listdir()[0])

    yaml_file = os.path.join(path, 'template.yaml')
    ini_file = os.path.join(path, 'pluto.ini')
    definition_file = os.path.join(path, 'definitions.h')

    configuration.create_sim_yaml_file_template(yaml_file, ini_file, definition_file)
    configuration.validate_yaml_keys(yaml_file)

@pytest.mark.datafiles(
    os.path.join(PLUTO_FIXTURE_DIR, 'king-config-files'), keep_top_dir=True)
def test_create_yaml_template_king(datafiles):
    assert len(datafiles.listdir()) == 1
    path = str(datafiles.listdir()[0])

    yaml_file = os.path.join(path, 'template.yaml')
    ini_file = os.path.join(path, 'pluto.ini')
    definition_file = os.path.join(path, 'definitions.h')

    configuration.create_sim_yaml_file_template(yaml_file, ini_file, definition_file)
