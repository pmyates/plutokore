#!/bin/env python3

import os
import sys
import numpy as np
import argparse
from numba import jit
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.table import QTable
import pathlib
import h5py
import glob

def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_directory', help='Data directory', type=str)
    parser.add_argument('name', help='Name of data files (e.g. radio for radio.XXXX.hdf5)', type=str)
    args = parser.parse_args()

    combined_file_name = f'{args.name}.hdf5'

    done_once = False


    _, number, _ = os.path.basename(sorted(glob.glob(f'{os.path.join(args.data_directory, args.name)}.*.hdf5'))[-1]).split('.')

    output = []
    total_luminosity = []

    with h5py.File(os.path.join(args.data_directory, combined_file_name), 'w') as f:
        for dfile in sorted(glob.glob(f'{args.data_directory}/{args.name}.*.hdf5')):
            with h5py.File(dfile) as g:
                if not done_once:
                    for name, value in list(g.values())[0].attrs.items():
                        if 'output' in name or 'total_luminosity' in name: continue
                        f.attrs[name] = value
                    done_once = True
                name, value = list(g.items())[0]
                if not name in f:
                    f.copy(value, f, without_attrs = True)
                    output.append(int(name))
                    total_luminosity.append(value.attrs['total_luminosity'])
        f.create_dataset('luminosity', data=np.transpose(np.vstack((output, total_luminosity))))
    return

if __name__ == '__main__':
    main()
