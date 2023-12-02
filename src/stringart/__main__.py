import argparse
from typing import Callable, Tuple


import numpy as np

Point = Tuple[float, float]

def chord_dist(a: Point, b: Point, c: Point) -> float:
    
    ax, ay = a
    bx, by = b
    cx, cy = c

    if ay == by:
        return abs(cy - ay)
    if ax == bx:
        return abs(cx - ax)
    
    ab_slope = (by-ay)/(bx-ax)
    ab_intercept = ay - ab_slope * ax

    orth_ab_slope = - 1. / ab_slope
    orth_ab_intercept = cy - orth_ab_slope * cx

    cross_x = (orth_ab_intercept - ab_intercept) / (ab_slope - orth_ab_slope)
    cross_y = ab_slope * cross_x + ab_intercept

    dist = np.sqrt((cx - cross_x)**2 + (cy - cross_y)**2)

    return dist

def compute_strings(image: np.array, influence_func: Callable[[float], float], influence_radius: float, num_nails: int) -> np.array:
    """Compute string art connections for image, using num nails.

    Returns a num_nails x num_nails array of non-negative integers, indicating the
    number of threads that should connect each pair of nails.
    """

    img_H, img_W = image.shape

    nail_xs = np.cos([i/num_nails * 2 * np.pi for i in range(num_nails)])
    nail_ys = np.sin([i/num_nails * 2 * np.pi for i in range(num_nails)])

    def nail_point(i: int):
        return (nail_xs[i], nail_ys[i])

    A = np.array(shape=(num_nails, num_nails))

    def compute_influence():

        for (i, j) in image.values():
            y, x = i/img_H, j/img_W

            mark = 1
            for m in range(num_nails):
                n = mark
                while n < num_nails and chord_dist(nail_point(m), nail_point(n), (x, y)) > influence_radius:
                    n += 1
                if n == num_nails:
                    return
                while n < num_nails and chord_dist(nail_point(m), nail_point(n), (x, y)) <= influence_radius:
                    dist = chord_dist(nail_point(m), nail_point(n), (x, y))
                    A[n,m] += influence_func(dist)
                    n += 1

    compute_influence()





def parse_args():
    parser = argparse.ArgumentParser()


def main():
    pass


if __name__ == "__main__":
    main()