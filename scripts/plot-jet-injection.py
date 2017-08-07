#!/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plot
import argparse

def plot_injection(t1, t2, iw, r, c, ir):
    l = ir+1
    x = np.linspace(-l, l, num=r)
    z = np.linspace(-l, l, num=r)
    v = np.zeros((x.shape[0], z.shape[0]))

    for i in range(v.shape[0]):
        for k in range(v.shape[1]):
            r = np.sqrt(x[i]**2 + z[k]**2)
            if z[k] > 0:
                off = iw / np.tan(t1)
                ih = ir * np.cos(t1)
            else:
                off = iw / np.tan(np.pi-t2)
                ih = ir * np.cos(t2)
            th = np.arccos((z[k] + off) / np.sqrt(x[i]**2 + (z[k] + off)**2))

            h = np.abs(r * np.cos(th))

            if (r <= ir and c == 'cap') or (h <= ih and c == 'nocap'):
                if z[k] > 0 and th < t1:
                    v[i,k] = 1
                elif z[k] < 0 and th > np.pi - t2:
                    v[i,k] = -1

    fig,ax = plot.subplots(1, 1, subplot_kw={'aspect': 'equal'})
    ax.pcolormesh(x, z, v.T)
    plot.show()
    return

def main():
    parser = argparse.ArgumentParser(
        description = 'Plot jet injection region'
    )
    parser.add_argument('theta1', type=float, help='First jet opening angle (degrees)')
    parser.add_argument('theta2', type=float, help='Second jet opening angle (degrees)')
    parser.add_argument('initial_width', type=float, help='Initial width of jet injection cone')
    parser.add_argument('-r', '--resolution', help='Grid cell count  (default: %(default)s)', type=int, default=100)
    parser.add_argument('-c', '--conetype', help='Cone type (with cap or without)', choices=['cap', 'nocap'], default='cap')
    parser.add_argument('-i', '--injectionradius', help='Injection radius (default: %(default)s)', type=float, default=1.0)
    args = parser.parse_args()

    plot_injection(np.deg2rad(args.theta1), np.deg2rad(args.theta2), args.initial_width, args.resolution, args.conetype, args.injectionradius)

if __name__ == "__main__":
    main()
