#!/usr/bin/env python3

import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sim_dir', help='Simulation directory')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('-s', '--skip', help='Skip pattern for outputs', default=1, type=int)
    args = parser.parse_args()

    print(args)

if __name__ == '__main__':
    main()
