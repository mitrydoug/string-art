import argparse
import os
import sys
from os.path import exists, join

import numpy as np
from tqdm import tqdm

from .utils import chord_dist, dist, connection_idx, nail_point, string_chord

NAIL_FRAC = 0.2

def mask_path(height, width, num_nails, nail_frac, max_dist):
    return join(
        "masks",
        f"mask_h={height}_w={width}_numnails={num_nails}_nailfrac={nail_frac}_maxdist={max_dist}.npy",
    )


def compute_dist_mask(width, height, num_nails, nail_frac, max_dist) -> np.ndarray:

    pixels = width * height
    connections = num_nails * (num_nails - 1)
    # note: circle radius is 1/2
    nail_radius = np.pi / num_nails * nail_frac

    d = dict()

    X = np.zeros((pixels, connections))

    for i, j in tqdm(np.ndindex(height, width), total=pixels):
        y = 1 - i / height
        x = j / width
        if dist((x, y), (0.5, 0.5)) > 0.5:
            continue
        # print(f"i={i}, j={j}")
        pixel_idx = i * width + j

        mark = 1
        for m in range(num_nails):
            # print(f"m={m}")
            n = max(mark, m + 1) % num_nails
            while (
                n != m
                and chord_dist(
                    *string_chord(m, n, num_nails, nail_radius), (x, y)
                )
                > max_dist
            ):
                n = (n+1) % num_nails

            mark = n
            while (
                n != m
                and chord_dist(
                    *string_chord(m, n, num_nails, nail_radius), (x, y)
                )
                <= max_dist
            ):
                conn_idx = connection_idx(m, n, num_nails)
                cdist = chord_dist(
                    *string_chord(m, n, num_nails, nail_radius), (x, y)
                )
                if (pixel_idx, conn_idx) in d:
                    print((pixel_idx, conn_idx))
                    print(d[(pixel_idx, conn_idx)])
                    print((i, j, m, n))
                    sys.exit(0)
                d[(pixel_idx, conn_idx)] = (i, j, m, n)

                X[pixel_idx, conn_idx] += 1 - cdist / max_dist
                n = (n + 1) % num_nails

    return X


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--height", type=int, default=128)
    parser.add_argument("-x", "--width", type=int, default=128)
    parser.add_argument("-n", "--num-nails", type=int, default=100)
    parser.add_argument("-f", "--nail-frac", type=float, default=0.2)
    parser.add_argument("-d", "--max-dist", type=float, default=0.1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    mask_fp = mask_path(args.width, args.height, args.num_nails, args.nail_frac, args.max_dist)
    if exists(mask_fp):
        print(f"Mask exists: {mask_fp}")
        sys.exit(0)
    x = compute_dist_mask(args.width, args.height, args.num_nails, args.nail_frac, args.max_dist)
    os.makedirs("masks", exist_ok=True)
    with open(mask_fp, "wb") as f:
        print(f"Saving mask: {mask_fp}")
        np.save(f, x)


if __name__ == "__main__":
    main()
