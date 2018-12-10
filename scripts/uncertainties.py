#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
if os.path.exists(os.path.expanduser('~/plutokore')):
    sys.path.append(os.path.expanduser('~/plutokore'))
else:
    sys.path.append(os.path.expanduser('~/uni/plutokore'))
import plutokore as pk
import numpy as np
import argparse
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from pathlib import Path
import h5py
import code
import scipy
from scipy import stats
import matplotlib.pyplot as plt

processed_path = Path('~/uni/sim-data/').expanduser()

paths = {
    'm12.5' : {
        # Makino simulations
        'M25-n1': processed_path / 'm12.5-M25-n1/',
        'M25-n2': processed_path / 'm12.5-M25-n2/',
        'M25-n3': processed_path / 'm12.5-M25-n3/',
        'M25-n4': processed_path / 'm12.5-M25-n4/',

        # King simulations
        'M25-n1-beta': processed_path / 'm12.5-M25-n1-beta/',
        'M25-n4-beta': processed_path / 'm12.5-M25-n4-beta/',

        # High res simulations
        'M25-n1-high-res': processed_path / 'm12.5-M25/',

        # High mach simulations
        'M75-n1': processed_path / 'm12.5-M75/',
        'M200-n1': processed_path / 'm12.5-M200/',
    },
    'm14.5' : {
        # Makino simulations
        'M25-n1': processed_path / 'm14.5-M25-n1/',
        'M25-n2': processed_path / 'm14.5-M25-n2/',
        'M25-n3': processed_path / 'm14.5-M25-n3/',
        'M25-n4': processed_path / 'm14.5-M25-n4/',

        # King simulations
        'M25-n1-beta': processed_path / 'm14.5-M25-n1-beta/',
        'M25-n4-beta': processed_path / 'm14.5-M25-n4-beta/',

        # High res simulations
        'M25-n1-high-res': processed_path / 'm14.5-M25/',

        # High mach simulations
        'M75-n1': processed_path / 'm14.5-M75/',
        'M200-n1': processed_path / 'm14.5-M200/',
    }
}


def load_radio_info(radio_dir):
    x1 = None
    x2 = None
    sb = None
    outputs = None
    lum = None
    with h5py.File(radio_dir / 'radio.hdf5', mode='r') as f:
        i = 0
        lum = np.array(f['luminosity'])
        for n,v in f.items():
            if 'luminosity' in n: continue
            if x1 is None:
                x1 = np.array(v['X1'])
                x2 = np.array(v['X2'])
            if sb is None:
                c = len(f.items()) - 1
                sb = np.zeros((c, x1.shape[0], x2.shape[0]))
            if outputs is None:
                outputs = np.zeros((c))
            sb[i,:] = np.array(v['sb'])
            outputs[i] = int(n)
            i += 1
    return {'x1': x1, 'x2': x2, 'sb': sb, 'outputs': outputs, 'lum': lum}

def load_sim_info(sim_dir):
    uv, env, jet = pk.configuration.load_simulation_info(sim_dir / 'config.yaml')
    return {'uv': uv, 'env': env, 'jet': jet}

def load_dynamics_info(sim_dir):
    with h5py.File(sim_dir / 'dynamics.hdf5', mode='r') as f:
        length = np.array(f['length'])
        time = np.array(f['time'])
        volume = np.array(f['volume'])
    return {'l': length, 'v': volume, 't': time}

def calculate_length_uncertainty():
    for env, sims in paths.items():
        x = np.arange(0, 40, 0.01)
        m25_data = load_dynamics_info(sims['M25-n1'])
        m75_data = load_dynamics_info(sims['M75-n1'])
        m200_data = load_dynamics_info(sims['M200-n1'])
        y25 = scipy.interpolate.interp1d(m25_data['t'], m25_data['l'], kind='cubic', bounds_error=False)(x)
        y75 = scipy.interpolate.interp1d(m75_data['t'], m75_data['l'], kind='cubic', bounds_error=False)(x)
        t = scipy.interpolate.UnivariateSpline(m200_data['t'][::2], m200_data['l'][::2], k=3, ext=0)
        y200 = t(x)

        plt.plot(x, y25)
        plt.plot(m25_data['t'], m25_data['l'])
        plt.plot(x, y75)
        plt.plot(m75_data['t'], m75_data['l'])
        plt.plot(x, y200)
        plt.plot(m200_data['t'], m200_data['l'])
        #plt.axes().set_aspect('equal')
        #plt.show()

        m25_isnan = ~np.isnan(y25)
        m75_isnan = ~np.isnan(y75)
        m200_isnan = ~np.isnan(y200)

        combined_2575 = np.logical_and(m25_isnan, m75_isnan)
        combined_25200 = np.logical_and(m25_isnan, m200_isnan)

        combined_2575[0] = False
        combined_25200[0] = False

        mean_2575 = np.nanmean(np.abs(y25[combined_2575]-y75[combined_2575])/y25[combined_2575])
        mean_25200 = np.nanmean(np.abs(y25[combined_25200]-y200[combined_25200])/y25[combined_25200])

        print(f'Length mean ({env})')
        print('M25,M75: {0}'.format(mean_2575))
        print('M25,M200: {0}'.format(mean_25200))

        # print(f'Length NRSMD ({env})')
        # print('M25,M75: {0}'.format((np.sqrt(np.sum((y25[m75_isnan] - y75[m75_isnan]) ** 2) / y25[m75_isnan].shape[0])) / (np.max(y25) - np.min(y25))))

def calculate_volume_uncertainty():
    for env, sims in paths.items():
        x = np.arange(0, 40, 0.01)
        m25_data = load_dynamics_info(sims['M25-n1'])
        m75_data = load_dynamics_info(sims['M75-n1'])
        m200_data = load_dynamics_info(sims['M200-n1'])
        y25 = scipy.interpolate.interp1d(m25_data['t'], m25_data['v'], kind='cubic', bounds_error=False)(x)
        y75 = scipy.interpolate.interp1d(m75_data['t'], m75_data['v'], kind='cubic', bounds_error=False)(x)
        #t = scipy.interpolate.UnivariateSpline(m200_data['t'][::2], m200_data['v'][::2], k=3, ext=0)
        t = scipy.interpolate.interp1d(m200_data['t'][::2], m200_data['v'][::2], kind='cubic', bounds_error=False)
        y200 = t(x)

        m25_isnan = ~np.isnan(y25)
        m75_isnan = ~np.isnan(y75)
        m200_isnan = ~np.isnan(y200)

        combined_2575 = np.logical_and(m25_isnan, m75_isnan)
        combined_25200 = np.logical_and(m25_isnan, m200_isnan)

        combined_2575[0] = False
        combined_25200[0] = False

        mean_2575 = np.nanmean(np.abs(y25[combined_2575]-y75[combined_2575])/y25[combined_2575])
        mean_25200 = np.nanmean(np.abs(y25[combined_25200]-y200[combined_25200])/y25[combined_25200])

        print(f'Volume mean ({env})')
        print('M25,M75: {0}'.format(mean_2575))
        print('M25,M200: {0}'.format(mean_25200))

def calculate_luminosity_uncertainty():
    for env, sims in paths.items():
        x = np.arange(0, 40, 0.01)
        m25_data = load_radio_info(sims['M25-n1'])
        m25_time = load_dynamics_info(sims['M25-n1'])['t']

        m75_data = load_radio_info(sims['M75-n1'])
        m75_time = load_dynamics_info(sims['M75-n1'])['t']

        m200_data = load_radio_info(sims['M200-n1'])
        m200_time = load_dynamics_info(sims['M200-n1'])['t']

        y25 = scipy.interpolate.interp1d(m25_time[m25_data['lum'][:,0].astype(int)], m25_data['lum'][:,1], kind='cubic', bounds_error=False)(x)
        y75 = scipy.interpolate.interp1d(m75_time, m75_data['lum'][:,1], kind='cubic', bounds_error=False)(x)
        #t = scipy.interpolate.UnivariateSpline(m200_data['t'][::2], m200_data['v'][::2], k=3, ext=0)
        t = scipy.interpolate.interp1d(m200_time[m200_data['lum'][:,0].astype(int)][::2], m200_data['lum'][::2,1], kind='cubic', bounds_error=False)
        y200 = t(x)

        m25_isnan = ~np.isnan(y25)
        m75_isnan = ~np.isnan(y75)
        m200_isnan = ~np.isnan(y200)

        combined_2575 = np.logical_and(m25_isnan, m75_isnan)
        combined_25200 = np.logical_and(m25_isnan, m200_isnan)

        combined_2575[0] = False
        combined_25200[0] = False

        mean_2575 = np.nanmean(np.abs(y25[combined_2575]-y75[combined_2575])/y25[combined_2575])
        mean_25200 = np.nanmean(np.abs(y25[combined_25200]-y200[combined_25200])/y25[combined_25200])

        print(f'Luminosity mean ({env})')
        print('M25,M75: {0}'.format(mean_2575))
        print('M25,M200: {0}'.format(mean_25200))

def main():
    calculate_length_uncertainty()
    calculate_volume_uncertainty()
    calculate_luminosity_uncertainty()

if __name__ == '__main__':
    main()
