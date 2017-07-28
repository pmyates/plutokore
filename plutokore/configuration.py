class SimulationConfiguration:

    # yaml version
    latest_yaml_version = 2

    def __init__(self, yaml_file, ini_file, definition_file):

        # Assign file locations
        self.yaml_file = yaml_file
        self.ini_file = ini_file,
        self.definition_file = definition_file

        # Create empty error list
        self.errors = []

        # Set tolerance value
        self.tolerance = 1e-3

    def check_values(self, success, expected, actual, location, message):
        """
        Checks the expected value against the actual value.
        Returns False if they don't match, True if they do.
        If they do not match, an error is added to the error list.
        """
        if not (success):
            self.errors.append({
                'message': message,
                'location': location,
                'expected_value': expected,
                'actual_value': actual,
            })
            return False
        return True

    def validate(self):
        if not self.validate_yaml_keys(): return
        if not self.validate_yaml_with_ini(): return
        if not self.validate_yaml_with_definitions(): return
        if not self.validate_unit_values(): return
        if not self.validate_times(): return
        if not self.validate_environment(): return

    def load_environment(self, env_properties):
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

    def load_jet(self, jet_properties, env):
        """Create a jet from the given jet properties and environment"""
        from astropy import units as u
        from . import jet

        mach = jet_properties['mach-number']
        power = (10 ** jet_properties['power-exponent']) * u.W
        if self.yaml_version == 1:
            opening_angle = jet_properties['opening-angle-degrees']
        elif self.yaml_version >= 2:
            opening_angle = jet_properties['opening-angle-degrees-one']

        j = jet.AstroJet(
            opening_angle,
            mach,
            env.sound_speed,
            env.central_density,
            power,
            env.gamma)
        return j

    def get_unit_values(self):
        """Return the unit values relating to the given simulation yaml"""
        import yaml
        from astropy import cosmology
        from astropy import units as u
        from . import environments
        from . import jet

        data = read_sim_yaml_file(self.yaml_file)

        env_properties = data['environment-properties']
        jet_properties = data['jet-properties']
        simulation_properties = data['simulation-properties']
        intermittent_properties = data['intermittent-properties']

        env = self.load_environment(env_properties)
        j = self.load_jet(jet_properties, env)
        uv = jet.get_unit_values(env, j)
        return uv

    def validate_yaml_with_ini(self):
        """Validates the given yaml file against the given ini file"""

        jp = 'jet-properties'
        sp = 'simulation-properties'
        param = 'Parameters'

        # load the files
        yaml_data = read_sim_yaml_file(self.yaml_file)
        ini_data = read_ini_file(self.ini_file)

        # check mach number matches
        self.check_values(yaml_data[jp]['mach-number'] == ini_data[param].getfloat('mach'), yaml_data[jp]['mach-number'], ini_data[param].getfloat('mach'), 'pluto.ini', 'Mach number in pluto.ini does not match Mach number in yaml')

        # check opening angle matches
        if self.yaml_version == 1:
            self.check_values(yaml_data[jp]['opening-angle-degrees'] == ini_data[param].getfloat('theta'), yaml_data[jp]['opening-angle-degrees'], ini_data[param].getfloat('theta'), 'pluto.ini', 'Opening angle in pluto.ini does not match angle in yaml')
        elif self.yaml_version >= 2:
            # check first jet opening angle
            self.check_values(yaml_data[jp]['opening-angle-degrees-one'] == ini_data[param].getfloat('theta_one'), yaml_data[jp]['opening-angle-degrees-one'], ini_data[param].getfloat('theta_one'), 'pluto.ini', 'Opening angle in pluto.ini does not match angle in yaml')
            # check second jet opening angle
            self.check_values(yaml_data[jp]['opening-angle-degrees-two'] == ini_data[param].getfloat('theta_two'), yaml_data[jp]['opening-angle-degrees-two'], ini_data[param].getfloat('theta_two'), 'pluto.ini', 'Second jet opening angle in pluto.ini does not match angle in yaml')

        # calculate duty cycle and check that it matches
        yaml_duty_cycle = float(yaml_data[sp]['jet-active-time-myrs']) / float(yaml_data[sp]['total-time-myrs'])
        ini_duty_cycle = float(ini_data[param].getfloat('jet_active_time')) / float(ini_data[param].getfloat('simulation_time'))
        self.check_values(relative_error(yaml_duty_cycle, ini_duty_cycle) <= self.tolerance, yaml_duty_cycle, ini_duty_cycle, 'pluto.ini', 'Duty cycle (active / total time) in pluto.ini does not match duty cycle in yaml')

        # check outburst count matches
        self.check_values(yaml_data['intermittent-properties']['outburst-count'] == ini_data[param].getfloat('jet_episodes'), yaml_data['intermittent-properties']['outburst-count'], ini_data[param].getfloat('jet_episodes'), 'pluto.ini', 'Outburst count in pluto.ini does not match outburst count in yaml')

        return True

    def validate_yaml_with_definitions(self):
        """Validates the given yaml file against the given definition file"""

        sp = 'simulation-properties'
        geom = 'GEOMETRY'
        dim = 'DIMENSIONS'
        ntrc = 'NTRACER'
        uj = 'UNIQUE_JETS'
        dp = 'DENSITY_PROFILE'

        # load the files
        yaml_data = read_sim_yaml_file(self.yaml_file)
        definitions_data = read_definition_file(self.definition_file)

        # check geometry matches
        self.check_values(yaml_data[sp]['geometry'].lower() == definitions_data[geom].lower(), yaml_data[sp]['geometry'].lower(), definitions_data[geom].lower(), 'definitions.h', 'Geometry in definitions.h does not match geometry in yaml')

        # check dimensions matches
        self.check_values(yaml_data[sp]['dimensions'] == int(definitions_data[dim]), yaml_data[sp]['dimensions'], int(definitions_data[dim]), 'definitions.h', 'Dimensions in definitions.h does not match dimensions in yaml')

        # check tracer count matches
        self.check_values(yaml_data[sp]['tracer-count'] == int(definitions_data[ntrc]), yaml_data[sp]['tracer-count'], int(definitions_data[ntrc]), 'definitions.h', 'Tracer count in definitions.h does not match tracer count in yaml')

        # check unique jet matches
        if self.yaml_version >=2:
            self.check_values(yaml_data[sp]['unique-jets'] == bool(definitions_data[uj].lower()), yaml_data[sp]['unique-jets'], bool(definitions_data[uj].lower()), 'definitions.h', 'Unique jets setting in definitions.h does not match unique jets setting in yaml')

        # check environment matches
        if self.yaml_version >=2:
            self.check_values(yaml_data['environment-properties']['profile'].lower() == definitions_data[dp].lower(), yaml_data['environment-properties']['profile'].lower(), definitions_data[dp].lower(), 'definitions.h', 'Profile specified in definitions.h does not match profile specified in yaml')

        return True


    def validate_yaml_keys(self):
        """Validates the keys in the given yaml file"""

        data = read_sim_yaml_file(self.yaml_file)

        # check we can process this yaml file (version is <= to latest_yaml_version)
        if 'yaml-version' in data:
            self.yaml_version = data['yaml-version']
        else:
            self.yaml_version = 1

        if not self.check_values(self.yaml_version <= self.latest_yaml_version, '<= {}'.format(self.latest_yaml_version), self.yaml_version, 'config.yaml', 'Yaml file is too new to be handled by this code'): return False

        if self.yaml_version == 1:
            base_keys = ('environment-properties', 'jet-properties', 'simulation-properties')
        elif self.yaml_version >= 2:
            base_keys = ('yaml-version', 'environment-properties', 'jet-properties', 'simulation-properties')

        e_keys = ('profile', 'concentration-profile', 'cosmology', 'halo-mass-exponent', 'redshift')

        if self.yaml_version == 1:
            j_keys = ('mach-number', 'power-exponent', 'opening-angle-degrees')
        elif self.yaml_version >= 2:
            j_keys = ('mach-number', 'power-exponent', 'opening-angle-degrees-one', 'opening-angle-degrees-two')

        if self.yaml_version == 1:
            s_keys = ('name', 'total-time-myrs', 'jet-active-time-myrs', 'geometry', 'dimensions', 'tracer-count')
        elif self.yaml_version >= 2:
            s_keys = ('name', 'total-time-myrs', 'jet-active-time-myrs', 'geometry', 'dimensions', 'tracer-count', 'unique-jets')
        i_keys = ('outburst-count',)
        extra_keys = ('intermittent-properties')

        for k in base_keys:
            if not self.check_values(k in data, k, data, 'config.yaml', 'Key does not exist in yaml file'): return False
        for k in e_keys:
            if not self.check_values(k in data['environment-properties'], k, data['environment-properties'], 'config.yaml', 'Key does not exist in yaml file'): return False
        for k in j_keys:
            if not self.check_values(k in data['jet-properties'], k, data['jet-properties'], 'config.yaml', 'Key does not exist in yaml file'): return False
        for k in s_keys:
            if not self.check_values(k in data['simulation-properties'], k, data['simulation-properties'], 'config.yaml', 'Key does not exist in yaml file'): return False
        if 'intermittent-properties' in data:
            for k in i_keys:
                if not self.check_values(k in data['intermittent-properties'], k, data['intermittent-properties'], 'config.yaml', 'Key does not exist in yaml file'): return False

        return True

    def validate_unit_values(self):
        """Validate the unit files obtained from the given yaml file with those in the definition file"""
        from astropy import units as u

        # load unit values
        uv = self.get_unit_values()

        # load definitions
        definition_data = read_definition_file(self.definition_file)

        # check unit density - it is in g / cm^3
        def_unit_density = float(definition_data['UNIT_DENSITY']) * u.g / (u.cm ** 3)
        self.check_values(relative_error(def_unit_density, uv.density.to(u.g / u.cm ** 3)) <= self.tolerance, uv.density.to(u.g / u.cm ** 3), def_unit_density, 'defintions.h', 'Unit density in definitions file does not match that calculated by yaml')

        # check unit length - it is in cm
        tmp = definition_data['UNIT_LENGTH'].split('*')

        # if there is no '*' then just take the value as it is
        if len(tmp) == 1: # pragma: no cover
            def_unit_length = float(tmp[0]) * u.cm
        else:             # otherwise the value is assumed to be given in parsecs
            def_unit_length = float(tmp[0]) * u.pc

        self.check_values(relative_error(uv.length.to(u.kpc), def_unit_length.to(u.kpc)) <= self.tolerance, uv.length.to(u.kpc), def_unit_length.to(u.kpc), 'definitions.h', 'Unit length in definitions file does not match that calculated by yaml')

        # check unit velocity - it is in cm / s
        def_unit_velocity = float(definition_data['UNIT_VELOCITY']) * u.cm / u.s
        self.check_values(relative_error(uv.speed.to(u.cm / u.s), def_unit_velocity) <= self.tolerance, uv.speed.to(u.km / u.s), def_unit_velocity.to(u.km / u.s), 'defintions.h', 'Unit velocity in definitions file does not match that calculated by yaml')

        return True

    def validate_times(self):
        """Validate the times obtained from the yaml file with those given in the ini file"""
        from astropy import units as u
        sp = 'simulation-properties'
        param = 'Parameters'

        # load yaml
        yaml_data = read_sim_yaml_file(self.yaml_file)
        # load unit values
        uv = self.get_unit_values()
        # load ini values
        ini_data = read_ini_file(self.ini_file)

        total_time_yaml = yaml_data[sp]['total-time-myrs'] * u.Myr
        total_time_sim = uv.time * ini_data[param].getfloat('simulation_time')
        self.check_values(relative_error(total_time_yaml, total_time_sim) <= self.tolerance, total_time_yaml, total_time_sim, 'pluto.ini', 'Total simulation time in ini file does not match that calculated by yaml')

        active_time_yaml = yaml_data[sp]['jet-active-time-myrs'] * u.Myr
        active_time_sim = uv.time * ini_data[param].getfloat('jet_active_time')
        self.check_values(relative_error(active_time_yaml, active_time_sim) <= self.tolerance, active_time_yaml, active_time_sim, 'pluto.ini', 'Jet active time in ini file does not match that calculated by yaml.\nSet to {} to match yaml.'.format(active_time_yaml / uv.time))

        return True

    def validate_environment(self):
        """Validate the environment obtained from the yaml file with that given in the ini file"""
        from astropy import units as u
        param = 'Parameters'

        # load yaml data
        yaml_data = read_sim_yaml_file(self.yaml_file)
        # load unit values
        uv = self.get_unit_values()
        # load ini values
        ini_data = read_ini_file(self.ini_file)
        # load environment
        env_data = self.load_environment(yaml_data['environment-properties'])
        # load jet
        jet_data = self.load_jet(yaml_data['jet-properties'], env_data)

        # which environment are we looking at?
        env = yaml_data['environment-properties']['profile']

        # check rho_0
        rho_0_sim = ini_data[param].getfloat('rho_0') * uv.density
        rho_0_calculated = env_data.central_density
        self.check_values(relative_error(rho_0_calculated, rho_0_sim) <= self.tolerance, rho_0_calculated, rho_0_sim, 'pluto.ini', 'Calculated rho_0 does not match rho_0 given in pluto.ini')

        # check delta_nfw - only if this is an NFW profile!
        if env == 'makino':
            self.check_values(relative_error(env_data.nfw_parameter.value, ini_data[param].getfloat('delta_nfw')) <= self.tolerance, env_data.nfw_parameter.value, ini_data[param].getfloat('delta_nfw'), 'pluto.ini', 'Calculated delta_NFW does not match delta_NFW given in pluto.ini')

        # check r_scaling - if this is nfw
        if env == 'makino':
            self.check_values(relative_error(env_data.scale_radius, ini_data[param].getfloat('r_scaling') * uv.length) <= self.tolerance, env_data.scale_radius, ini_data[param].getfloat('r_scaling') * uv.length, 'pluto.ini', 'Calculated scale/core radius does not match that given in pluto.ini')

        # check r_core - if this is king
        if env == 'king':
            self.check_values(relative_error(env_data.core_radius, ini_data[param].getfloat('r_scaling') * uv.length) <= self.tolerance, env_data.core_radius, ini_data[param].getfloat('r_scaling') * uv.length, 'pluto.ini', 'Calculated scale/core radius does not match that given in pluto.ini')

        return True

def create_sim_yaml_file_template(yaml_file, ini_file, definition_file):
    """Create a template simulation yaml file using the given ini and definition files as a base"""
    import yaml
    import os

    default = 'CHANGEME'

    # load data files
    ini_data = read_ini_file(ini_file)
    definition_data = read_definition_file(definition_file)

    # get yaml version
    if 'DENSITY_PROFILE' in definition_data:
        version = SimulationConfiguration.latest_yaml_version
    else:
        version = 1

    ep = {
        'profile': default,
        'concentration-profile': default,
        'cosmology': default,
        'halo-mass-exponent': default,
        'redshift': default
    }

    jp = {
        'mach-number': ini_data['Parameters'].getfloat('mach'),
        'power-exponent': default,
    }
    if version == 1:
        jp['opening-angle-degrees'] = ini_data['Parameters'].getfloat('theta')
    elif version >=2:
        jp['opening-angle-degrees-one'] = ini_data['Parameters'].getfloat('theta_one'),
        jp['opening-angle-degrees-two'] = ini_data['Parameters'].getfloat('theta_two'),

    sp = {
        'name': default,
        'total-time-myrs': default,
        'jet-active-time-myrs': default,
        'geometry': definition_data['GEOMETRY'],
        'dimensions': int(definition_data['DIMENSIONS']),
        'tracer-count': int(definition_data['NTRACER']),
    }
    if version >= 2:
        sp['unique-jets'] = bool(definition_data['UNIQUE_JETS'])

    ip = {
        'outburst-count': int(ini_data['Parameters'].getfloat('jet_episodes'))
    }

    yaml_data = {
        'yaml-version': version,
        'environment-properties': ep,
        'jet-properties': jp,
        'simulation-properties': sp,
        'intermittent-properties': ip,
    }

    if os.path.isfile(yaml_file):
        raise ValueError('A yaml file already exists at the given path')
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_data, stream=f, default_flow_style=False, indent=4)

def validate_yaml(yaml_file, ini_file, definition_file):
    """Validate a yaml file with ini and definition files"""
    config = SimulationConfiguration(yaml_file, ini_file, definition_file)
    config.validate()
    return config.errors

def read_sim_yaml_file(yaml_file):
    """Return a list of all lines in the given yaml file"""
    import yaml
    with open(yaml_file) as f:
        return yaml.load(f)

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

def setup_path(): #pragma: no cover
    """Add the plutokore development folder to the path so it can be imported"""
    import sys
    sys.path.append('/home/patrick/honours/plutokore')

def relative_error(a,b):
    """Return the relative error between a and b"""
    return abs(a - b)/min(abs(a), abs(b))
