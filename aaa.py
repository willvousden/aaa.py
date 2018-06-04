#!/usr/bin/env python3
'''
Plot the elevation profile from CSVs produced by GPS Visualizer.

'''

from __future__ import print_function, division

import argparse
import itertools
import matplotlib as mpl
import matplotlib.pyplot as pp
import numpy as np
import sys

pp.style.use('ggplot')

_KM_FACTOR = 1000
_EARTH_RADIUS_KM = 6367
_COLOURS = itertools.cycle(mpl.rcParams['axes.prop_cycle'].by_key()['color'])


def haversine(lon1, lat1, lon2, lat2):
    '''
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    '''
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return _EARTH_RADIUS_KM * c


def accumulate(lat, long):
    distance = np.cumsum(haversine(long[:-1], lat[:-1],
                                   long[1:], lat[1:]))
    return np.concatenate(([0], distance))


def smooth(x, y, size):
    s = size
    l = size // 2
    y = np.cumsum(y)
    return (x[l:-l],
            (y[s:] - y[:-s]) / size)


def main(argv):
    parser = argparse.ArgumentParser(prog=argv[0],
                                     description=__doc__)
    parser.add_argument('-i', '--input',
                        help='The track file name.')
    parser.add_argument('-o', '--output',
                        help='The figure file name.')
    parser.add_argument('--splits',
                        metavar='DISTANCE',
                        help='Calculate climb for splits of this distance '
                             '(in km).',
                        required=False)
    args = parser.parse_args()

    # Load AAA climb rating data.
    aaa_dist, aaa_climb = np.loadtxt('aaa.csv').T

    f, axes = pp.subplots(3, 1, sharex=True)
    a1, a2, a3 = axes
    #  data = np.loadtxt(args.input, usecols=[1, 2, 3], skiprows=1)
    data = np.loadtxt(args.input, usecols=[0, 1, 2], skiprows=1)

    dist = accumulate(data[:, 0], data[:, 1])
    elev = data[:, 2]

    dist_ = np.arange(dist[0], dist[-1], 1 / _KM_FACTOR)
    elev = np.interp(dist_, dist, elev)
    dist = dist_

    climb = np.cumsum(np.maximum(0, np.diff(elev)))

    a1.plot(*smooth(dist, elev, 20))
    a1.set_ylabel('Elevation (m)')
    a1.set_xlim(dist[0], dist[-1])

    a2.plot(dist[1:], climb)
    a2.set_ylabel('Total climb (m)')

    aaa_distances = np.arange(50, 700, 100)
    for colour, d in zip(_COLOURS, aaa_distances):
        n = d * _KM_FACTOR
        c = np.interp(d, aaa_dist, aaa_climb)
        rolling_sum = climb[n:] - climb[:-n]
        a3.plot(dist[n//2+1:-n//2],
                rolling_sum,
                color=colour,
                label='{} km'.format(d))
        a3.axhline(y=c,
                   dashes=(3, 2),
                   color=colour)
        if np.any(rolling_sum >= c):
            aaa_points = np.round(np.max(rolling_sum) / 250) / 4

    print('AAA points: {}'.format(aaa_points))
    a3.legend(loc='upper right')
    a3.set_ylabel('AAA climb (m)')
    a3.set_xlabel('Distance (km)')

    f.savefig(args.output)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
