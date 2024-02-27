from typing import Tuple

import numpy as np

Point = Tuple[float, float]


def nail_point(i: int, num_nails: int) -> Point:
    theta = 2 * np.pi * i / num_nails
    return ((np.cos(theta) + 1) / 2, (np.sin(theta) + 1) / 2)


def dist(a: Point, b: Point):
    ax, ay = a
    bx, by = b

    return np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


def chord_dist(a: Point, b: Point, c: Point) -> float:
    ax, ay = a
    bx, by = b
    cx, cy = c

    if abs(ay-by) < 1e-4:
        return abs(cy - ay)
    if abs(ax-bx) < 1e-4:
        return abs(cx - ax)

    ab_slope = (by - ay) / (bx - ax)
    ab_intercept = ay - ab_slope * ax

    orth_ab_slope = -1.0 / ab_slope
    orth_ab_intercept = cy - orth_ab_slope * cx

    cross_x = (orth_ab_intercept - ab_intercept) / (ab_slope - orth_ab_slope)
    cross_y = ab_slope * cross_x + ab_intercept

    return dist((cx, cy), (cross_x, cross_y))
