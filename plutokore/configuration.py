def read_sim_yaml_file(yaml_file):
    """Return a list of all lines in the given yaml file"""
    import yaml
    with open(yaml_file) as f:
        return yaml.load(f)

def create_sim_yaml_file_template(yaml_file, ini_file, definition_file):
    """Create a template simulation yaml file using the given ini and definition files as a base"""
    import yaml

    default = 'CHANGEME'

    # load data files
    ini_data = read_ini_file(ini_file)
    definition_data = read_definition_file(definition_file)

    yaml_data = {
        'environment-properties': {
            'profile': default,
            'concentration-profile': default,
            'cosmology': default,
            'halo-mass-exponent': default,
            'redshift': default
        },
        'jet-properties': {
            'mach-number': ini_data['Parameters'].getfloat('mach'),
            'power-exponent': default,
            'opening-angle-degrees': ini_data['Parameters'].getfloat('theta'),
        },
        'simulation-properties': {
            'name': default,
            'total-time-myrs': default,
            'jet-active-time-myrs': default,
            'geometry': definition_data['GEOMETRY'],
            'dimensions': int(definition_data['DIMENSIONS']),
            'tracer-count': int(definition_data['NTRACER'])
        },
        'intermittent-properties': {
            'outburst-count': int(ini_data['Parameters'].getfloat('jet_episodes'))
        }
    }

    with open(yaml_file, 'x') as f:
        yaml.dump(yaml_data, stream=f, default_flow_style=False, indent=4)

def validate_yaml(yaml_file, ini_file, definition_file):
    """Validate a yaml file with ini and definition files"""
    validate_yaml_keys(yaml_file)
    validate_yaml_with_ini(yaml_file, ini_file)
    validate_yaml_with_definitions(yaml_file, definition_file)
    validate_unit_values(yaml_file, definition_file)
    validate_times(yaml_file, ini_file)
    validate_environment(yaml_file, ini_file)

def load_environment(env_properties):
    """Create an environment from the given environment properties"""
    from astropy import cosmology
    from astropy import units as u
    from plutokore import environments

    cosmo_options = {'planck': cosmology.Planck15, 'wmap': cosmology.WMAP9}
    profile_options = {'makino': environments.makino.MakinoProfile,
                       'king': environments.king.KingProfile}
    conc_profile_options = {'klypin-relaxed': 'klypin-planck-relaxed',
                            'klypin-all': 'klypin-planck-all'}

    cosmo = cosmo_options[env_properties['cosmology']]
    profile_creator = profile_options[env_properties['profile']]
    conc_profile = conc_profile_options[env_properties['concentration-profile']]

    mass = (10 ** env_properties['halo-mass-exponent']) * u.M_sun
    redshift = env_properties['redshift']

    env = profile_creator(
        mass,
        redshift,
        delta_vir=200,
        cosmo=cosmo,
        concentration_method=conc_profile)
    return env

def load_jet(jet_properties, env):
    """Create a jet from the given jet properties and environment"""
    from astropy import units as u
    from . import jet

    mach = jet_properties['mach-number']
    power = (10 ** jet_properties['power-exponent']) * u.W
    opening_angle = jet_properties['opening-angle-degrees']

    j = jet.AstroJet(
        opening_angle,
        mach,
        env.sound_speed,
        env.central_density,
        power,
        env.gamma)
    return j

def get_unit_values(sim_yaml):
    """Return the unit values relating to the given simulation yaml"""
    import yaml
    from astropy import cosmology
    from astropy import units as u
    from . import environments
    from . import jet
    from . import helpers

    data = read_sim_yaml_file(sim_yaml)

    env_properties = data['environment-properties']
    jet_properties = data['jet-properties']
    simulation_properties = data['simulation-properties']
    intermittent_properties = data['intermittent-properties']

    env = load_environment(env_properties)
    j = load_jet(jet_properties, env)
    uv = helpers.get_unit_values(env, j)
    return uv

def read_definition_file(definition_file):
    """Loads the given definition file"""
    define_character = '#'

    with open(definition_file) as f:
        lines = f.readlines()

    # find only lines beginning with '#'
    filtered = [x for x in lines if x.startswith(define_character)]

    # create and populate settings dictionary
    settings_dict = {}

    for line in filtered:
        sp = line.split()[1:] # First entry is '#define'
        settings_dict[sp[0]] = sp[1]
    return settings_dict

def read_ini_file(ini_file):
    """Loads the given ini file"""
    import configparser

    parameter_key = 'Parameters'

    # setup config parser with whitespace as a delimiter
    parser = configparser.ConfigParser(delimiters=(' '))

    # read the config
    parser.read(ini_file)

    # load the Mach number from the file
    return parser

def validate_yaml_with_ini(yaml_file, ini_file):
    """Validates the given yaml file against the given ini file"""

    jp = 'jet-properties'
    sp = 'simulation-properties'
    param = 'Parameters'

    # load the files
    yaml_data = read_sim_yaml_file(yaml_file)
    ini_data = read_ini_file(ini_file)

    # check mach number matches
    assert yaml_data[jp]['mach-number'] == ini_data[param].getfloat('mach'), 'Mach number in pluto.ini does not match Mach number in yaml'

    # check opening angle matches
    assert yaml_data[jp]['opening-angle-degrees'] == ini_data[param].getfloat('theta'), 'Opening angle in pluto.ini does not match angle in yaml'
    
    # calculate duty cycle and check that it matches
    yaml_duty_cycle = yaml_data[sp]['jet-active-time-myrs'] / yaml_data[sp]['total-time-myrs']
    ini_duty_cycle = ini_data[param].getfloat('jet_active_time') / ini_data[param].getfloat('simulation_time')
    assert abs(yaml_duty_cycle - ini_duty_cycle) < 0.1, 'Duty cycle (active / total time) in pluto.ini does not match duty cycle in yaml'

    # check outburst count matches
    assert yaml_data['intermittent-properties']['outburst-count'] == ini_data[param].getfloat('jet_episodes'), 'Outburst count in pluto.ini does not match outburst count in yaml'

def validate_yaml_with_definitions(yaml_file, definition_file):
    """Validates the given yaml file against the given definition file"""

    sp = 'simulation-properties'
    geom = 'GEOMETRY'
    dim = 'DIMENSIONS'
    ntrc = 'NTRACER'

    # load the files
    yaml_data = read_sim_yaml_file(yaml_file)
    definitions_data = read_definition_file(definition_file)

    # check geometry matches
    assert yaml_data[sp]['geometry'].lower() == definitions_data[geom].lower(), 'Geometry in definitions.h does not match geometry in yaml'

    # check dimensions matches
    assert yaml_data[sp]['dimensions'] == int(definitions_data[dim]), 'Dimensions in definitions.h does not match dimensions in yaml'

    # check tracer count matches
    assert yaml_data[sp]['tracer-count'] == int(definitions_data[ntrc]), 'Tracer count in definitions.h does not match tracer count in yaml'

def validate_yaml_keys(yaml_file):
    """Validates the keys in the given yaml file"""

    base_keys = ('environment-properties', 'jet-properties', 'simulation-properties')
    e_keys = ('profile', 'concentration-profile', 'cosmology', 'halo-mass-exponent', 'redshift')
    j_keys = ('mach-number', 'power-exponent', 'opening-angle-degrees')
    s_keys = ('name', 'total-time-myrs', 'jet-active-time-myrs', 'geometry', 'dimensions', 'tracer-count')
    i_keys = ('outburst-count',)
    extra_keys = ('intermittent-properties')

    data = read_sim_yaml_file(yaml_file)

    for k in base_keys:
        assert k in data
    for k in e_keys:
        assert k in data['environment-properties']
    for k in j_keys:
        assert k in data['jet-properties']
    for k in s_keys:
        assert k in data['simulation-properties']
    if 'intermittent-properties' in data:
        for k in i_keys:
            assert k in data['intermittent-properties']

def validate_unit_values(yaml_file, definition_file):
    """Validate the unit files obtained from the given yaml file with those in the definition file"""
    from astropy import units as u

    # load unit values
    uv = get_unit_values(yaml_file)

    # load definitions
    definition_data = read_definition_file(definition_file)

    # check unit density - it is in g / cm^3
    def_unit_density = float(definition_data['UNIT_DENSITY']) * u.g / (u.cm ** 3)
    assert abs((def_unit_density - uv.density.to(u.g / u.cm ** 3)) / def_unit_density) < 0.1, 'Unit density in definitions file does not match that calculated by yaml'

    # check unit length - it is in cm
    tmp = definition_data['UNIT_LENGTH'].split('*')
    if len(tmp) == 1: # if there is no '*' then just take the value as it is
        def_unit_length = float(tmp[0]) * u.cm
    else:             # otherwise the value is assumed to be given in parsecs
        def_unit_length = float(tmp[0]) * u.pc
    assert abs((def_unit_length.to(u.cm) - uv.length.to(u.cm)) / def_unit_length.to(u.cm)) < 0.1, 'Unit length in definitions file does not match that calculated by yaml'

    # check unit velocity - it is in cm / s
    def_unit_velocity = float(definition_data['UNIT_VELOCITY']) * u.cm / u.s
    assert abs((def_unit_velocity - uv.speed.to(u.cm / u.s)) / def_unit_velocity) < 0.1, 'Unit velocity in definitions file does not match that calculated by yaml'

def validate_times(yaml_file, ini_file):
    """Validate the times obtained from the yaml file with those given in the ini file"""
    from astropy import units as u
    sp = 'simulation-properties'
    param = 'Parameters'

    # load yaml
    yaml_data = read_sim_yaml_file(yaml_file)
    # load unit values
    uv = get_unit_values(yaml_file)
    # load ini values
    ini_data = read_ini_file(ini_file)

    total_time_yaml = yaml_data[sp]['total-time-myrs'] * u.Myr
    total_time_sim = uv.time * ini_data[param].getfloat('simulation_time')
    assert abs((total_time_sim - total_time_yaml) / total_time_sim) < 0.1, 'Total simulation time in ini file does not match that calculated by yaml'

    active_time_yaml = yaml_data[sp]['jet-active-time-myrs'] * u.Myr
    active_time_sim = uv.time * ini_data[param].getfloat('jet_active_time')
    assert abs((active_time_sim - active_time_yaml) / active_time_sim) < 0.1, 'Jet active time in ini file does not match that calculated by yaml'

def validate_environment(yaml_file, ini_file):
    """Validate the environment obtained from the yaml file with that given in the ini file"""
    from astropy import units as u
    param = 'Parameters'

    # load yaml data
    yaml_data = read_sim_yaml_file(yaml_file)
    # load unit values
    uv = get_unit_values(yaml_file)
    # load ini values
    ini_data = read_ini_file(ini_file)
    # load environment
    env_data = load_environment(yaml_data['environment-properties'])
    # load jet
    jet_data = load_jet(yaml_data['jet-properties'], env_data)

    # which environment are we looking at?
    env = yaml_data['environment-properties']['profile']

    # check rho_0
    rho_0_sim = ini_data[param].getfloat('rho_0') * uv.density
    rho_0_calculated = env_data.central_density
    assert relative_error(rho_0_sim, rho_0_calculated) < 0.1, 'Calculated rho_0 does not match rho_0 given in pluto.ini'

    # check delta_nfw - only if this is an NFW profile!
    if env == 'makino':
        assert relative_error(env_data.nfw_parameter, ini_data[param].getfloat('delta_nfw')) < 0.1, 'Calculated delta_NFW does not match delta_NFW given in pluto.ini'

    # check r_scaling - if this is nfw
    if env == 'makino':
        assert relative_error(env_data.scale_radius, ini_data[param].getfloat('r_scaling') * uv.length) < 0.1, 'Calculated scale/core radius does not match that given in pluto.ini'

    # check r_core - if this is king
    if env == 'king':
        assert relative_error(env_data.core_radius, ini_data[param].getfloat('r_scaling') * uv.length) < 0.1, 'Calculated scale/core radius does not match that given in pluto.ini'

def setup_path(): #pragma: no cover
    """Add the plutokore development folder to the path so it can be imported"""
    import sys
    sys.path.append('/home/patrick/honours/plutokore')

def relative_error(a,b):
    """Return the relative error between a and b"""
    return abs(a - b)/min(abs(a), abs(b))
