#!/bin/env python3

import sys
import os
try:
    import plutokore as pk
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import plutokore as pk
import argparse
import pprint

def check_directory(directory):
    print('Checking simulation directory {}'.format(directory))
    paths = {
        'yaml': os.path.join(directory, 'config.yaml'),
        'ini': os.path.join(directory, 'pluto.ini'),
        'definitions': os.path.join(directory, 'definitions.h')
    }
    errors = pk.configuration.validate_yaml(paths['yaml'], paths['ini'], paths['definitions'])

    pprint.pprint(errors, width=100) if len(errors) > 0 else print('Configuration valid')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Simulation directory')
    args = parser.parse_args()

    check_directory(args.directory)

if __name__ == '__main__':
    main()
