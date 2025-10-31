#!/usr/bin/env python3
"""
Plot the elevation profile from CSVs produced by GPS Visualizer.

"""

from __future__ import annotations

import argparse
import gpxpy
import itertools
import matplotlib as mpl
import matplotlib.pyplot as pp
import numpy as np
import sys

from pathlib import Path


_KM_FACTOR = 1000
_EARTH_RADIUS_KM = 6367
_COLOURS = mpl.rcParams["axes.prop_cycle"].by_key()["color"]


def read_csv(
    path: Path, columns: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if columns is None:
        columns = [0, 1, 2]
    data = np.loadtxt(path, usecols=columns, skiprows=1)
    lat, long, elev = data.T
    distance = accumulate(lat, long)
    return distance, elev


def read_gpx(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r") as f:
        gpx = gpxpy.parse(f)

    lat: list[float] = []
    long: list[float] = []
    elev: list[float | None] = []
    track = gpx.tracks[0]
    segment = track.segments[0]
    for point in segment.points:
        lat.append(point.latitude)
        long.append(point.longitude)
        elev.append(point.elevation)

    lat_arr = np.array(lat)
    long_arr = np.array(long)
    elev_arr = np.array([e if e is not None else 0.0 for e in elev])
    distance = accumulate(lat_arr, long_arr)
    return distance, elev_arr


def get_elevation(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix == ".gpx":
        distance, elevation = read_gpx(path)
    elif path.suffix == ".csv":
        distance, elevation = read_csv(path)
    else:
        raise ValueError(f"Unknown input file format: {path}")

    # Re-sample at 1 metre.
    distance, elevation = resample(distance, elevation, 1 / _KM_FACTOR)
    return distance, elevation, get_climb(elevation)


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return _EARTH_RADIUS_KM * c


def accumulate(lat: np.ndarray, long: np.ndarray) -> np.ndarray:
    distance = np.cumsum(haversine(long[:-1], lat[:-1], long[1:], lat[1:]))
    return np.concatenate(([0], distance))


def smooth(x: np.ndarray, y: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    half_size = size // 2
    y_cumsum = np.cumsum(y)
    return (x[half_size:-half_size], (y_cumsum[size:] - y_cumsum[:-size]) / size)


def get_climb(elevation: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(np.maximum(0, np.diff(elevation)))))


def resample(x: np.ndarray, y: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    x_ = np.arange(x[0], x[-1], dx)
    return x_, np.interp(x_, x, y)


def aaa_plot(args: argparse.Namespace) -> mpl.figure.Figure:
    if args.input is None:
        raise ValueError("Input file is required")
    distance, elevation, climb = get_elevation(args.input)

    # Load AAA climb rating presets.
    aaa_csv_path = Path("aaa.csv")
    aaa_dist, aaa_climb = np.loadtxt(aaa_csv_path).T

    f, axes = pp.subplots(3, 1, sharex=True, dpi=300)
    a1, a2, a3 = axes
    a1.plot(*smooth(distance, elevation, 20))
    a1.set_ylabel("Elevation (m)")
    a1.set_xlim(distance[0], distance[-1])

    a2.plot(distance, climb)
    a2.set_ylabel("Total climb (m)")

    aaa_distances = np.arange(50, 700, 100)
    aaa_points = 0.0
    for colour, d in zip(itertools.cycle(_COLOURS), aaa_distances):
        n = d * _KM_FACTOR
        c = np.interp(d, aaa_dist, aaa_climb)
        rolling_sum = climb[n:] - climb[:-n]
        a3.plot(
            distance[n // 2 + 1 : -n // 2],
            rolling_sum,
            color=colour,
            label=f"{d} km",
        )
        a3.axhline(y=c, dashes=(3, 2), color=colour)
        if np.any(rolling_sum >= c):
            aaa_points = np.round(np.max(rolling_sum) / 250) / 4

    print(f"AAA points: {aaa_points}")
    a3.legend(loc="upper right")
    a3.set_ylabel("AAA climb (m)")
    a3.set_xlabel("Distance (km)")

    return f


def splits_plot(args: argparse.Namespace) -> mpl.figure.Figure:
    if args.input is None:
        raise ValueError("Input file is required")
    distance, elevation, climb = get_elevation(Path(args.input))

    f, axes = pp.subplots(3, 1, sharex=True)
    a1, a2, a3 = axes

    a1.plot(*smooth(distance, elevation, 20), color=_COLOURS[0])
    a1.plot(
        *smooth(distance, elevation, 20 * _KM_FACTOR), alpha=0.75, color=_COLOURS[0]
    )
    a1.set_ylabel("Elevation (m)")
    a1.set_xlim(distance[0], distance[-1])

    a2.plot(distance, climb)
    a2.set_ylabel("Total climb (m)")

    bins = np.arange(0, distance[-1], args.length)
    a3.hist(
        distance, bins=bins, weights=np.concatenate(([0], np.diff(climb))) / args.length
    )
    a3.set_ylabel("Climb per km")
    a3.set_xlabel("Distance (km)")

    return f


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=argv[0], description=__doc__)
    parser.add_argument("-i", "--input", type=Path, help="The track file name.")
    subparsers = parser.add_subparsers()

    parser_splits = subparsers.add_parser("splits")
    parser_splits.add_argument(
        "-o", "--output", type=Path, help="The figure file name."
    )
    parser_splits.add_argument(
        "-l",
        "--length",
        type=int,
        metavar="DISTANCE",
        help="Calculate climb for splits of this distance "
        "(in km).  Default: %(default)s",
        required=False,
        default=50,
    )
    parser_splits.set_defaults(target=splits_plot)

    parser_aaa = subparsers.add_parser("aaa")
    parser_aaa.add_argument("-o", "--output", type=Path, help="The figure file name.")
    parser_aaa.set_defaults(target=aaa_plot)

    args = parser.parse_args()
    if not hasattr(args, "target"):
        parser.print_help()
        return 1
    figure = args.target(args)
    if args.output:
        figure.savefig(args.output, dpi=300)
    else:
        pp.show(figure)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
