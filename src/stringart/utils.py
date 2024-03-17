from typing import Tuple

import numpy as np

Point = Tuple[float, float]

def argmin(d):
    m = min(d.items(), key=(lambda t: t[1]))
    return m[0]

def argmax(d):
    m = max(d.items(), key=(lambda t: t[1]))
    return m[0]

def mse_loss(img, simg):
    return ((img - simg) ** 2).sum()

def normalize(arr: np.ndarray, min: float, max: float) -> np.ndarray:
    return (arr - min) / (max - min)

def connection_idx(m: int, n: int, num_nails: int) -> int:
    return m * (num_nails - 1) + n - (1 if n > m else 0)


def nail_point(i: int, num_nails: int) -> Point:
    theta = -2 * np.pi * i / num_nails
    return ((np.cos(theta) + 1) / 2, (np.sin(theta) + 1) / 2)


def string_chord(m, n, num_nails, nail_radius):

    a = np.array(nail_point(m, num_nails))
    b = np.array(nail_point(n, num_nails))

    if a[0] == b[0]:
        angle = 0 if a[1] < b[1] else np.pi
    elif a[1] == b[1]:
        angle = -np.pi / 2 if a[0] < b[0] else np.pi / 2
    else:
        slope = (b[1] - a[1]) / (b[0] - a[0])
        angle = np.arctan(slope)
        if a[0] < b[0]:
            angle -= np.pi / 2
        else:
            angle += np.pi / 2

    offset = np.array((np.cos(angle), np.sin(angle))) * nail_radius
    return tuple(a + offset), tuple(b + offset)


def dist(a: Point, b: Point):
    ax, ay = a
    bx, by = b

    return np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


def chord_dist(a: Point, b: Point, c: Point) -> float:
    ax, ay = a
    bx, by = b
    cx, cy = c

    if abs(ay - by) < 1e-4:
        return abs(cy - ay)
    if abs(ax - bx) < 1e-4:
        return abs(cx - ax)

    ab_slope = (by - ay) / (bx - ax)
    ab_intercept = ay - ab_slope * ax

    orth_ab_slope = -1.0 / ab_slope
    orth_ab_intercept = cy - orth_ab_slope * cx

    cross_x = (orth_ab_intercept - ab_intercept) / (ab_slope - orth_ab_slope)
    cross_y = ab_slope * cross_x + ab_intercept

    return dist((cx, cy), (cross_x, cross_y))
