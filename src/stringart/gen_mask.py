import argparse
import os
import sys
from os.path import exists, join

import numpy as np
from tqdm import tqdm

from .utils import chord_dist, dist, nail_point


def mask_path(height, width, num_nails, max_dist):
    return join(
        "masks",
        f"mask_h={height}_w={width}_numnails={num_nails}_maxdist={max_dist}.npy",
    )


def _connection_idx(m, n, num_nails):
    # (t-1) + (t-2) + ... + (t-m) + n
    # t * m - m * (m-1) / 2
    return num_nails * m - (m + 1) * m // 2 + n - m - 1


def compute_dist_mask(width, height, num_nails, max_dist) -> np.ndarray:
    pixels = width * height
    connections = (num_nails - 1) * num_nails // 2

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
            n = max(mark, m + 1)
            while (
                n < num_nails
                and chord_dist(
                    nail_point(m, num_nails), nail_point(n, num_nails), (x, y)
                )
                > max_dist
            ):
                n += 1
            if n == num_nails:
                break
            mark = n
            while (
                n < num_nails
                and chord_dist(
                    nail_point(m, num_nails), nail_point(n, num_nails), (x, y)
                )
                <= max_dist
            ):
                conn_idx = _connection_idx(m, n, num_nails)
                cdist = chord_dist(
                    nail_point(m, num_nails), nail_point(n, num_nails), (x, y)
                )
                if (pixel_idx, conn_idx) in d:
                    print((pixel_idx, conn_idx))
                    print(d[(pixel_idx, conn_idx)])
                    print((i, j, m, n))
                    sys.exit(0)
                d[(pixel_idx, conn_idx)] = (i, j, m, n)

                X[pixel_idx, conn_idx] += 1 - cdist / max_dist
                n += 1

    return X


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--height", type=int, default=128)
    parser.add_argument("-x", "--width", type=int, default=128)
    parser.add_argument("-n", "--num-nails", type=int, default=100)
    parser.add_argument("-d", "--max-dist", type=float, default=0.1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    mask_fp = mask_path(args.width, args.height, args.num_nails, args.max_dist)
    if exists(mask_fp):
        print(f"Mask exists: {mask_fp}")
        sys.exit(0)
    x = compute_dist_mask(args.width, args.height, args.num_nails, args.max_dist)
    os.makedirs("masks", exist_ok=True)
    with open(mask_fp, "wb") as f:
        print(f"Saving mask: {mask_fp}")
        np.save(f, x)


if __name__ == "__main__":
    main()
