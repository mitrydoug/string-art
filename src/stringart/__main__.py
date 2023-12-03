import argparse
from typing import Callable, Tuple

import numpy as np
import tempfile
import torch
from PIL import Image
from matplotlib import pyplot as plt

Point = Tuple[float, float]

LEARNING_RATE = 1e-3
MAX_EPOCHS = 50


def load_image(fname):
    img = Image.open(fname).convert("L")
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        img.save(tf.name)
        return plt.imread(tf.name)


def dist(a: Point, b: Point):
    ax, ay = a
    bx, by = b

    return np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


def chord_dist(a: Point, b: Point, c: Point) -> float:
    ax, ay = a
    bx, by = b
    cx, cy = c

    if ay == by:
        return abs(cy - ay)
    if ax == bx:
        return abs(cx - ax)

    ab_slope = (by - ay) / (bx - ax)
    ab_intercept = ay - ab_slope * ax

    orth_ab_slope = -1.0 / ab_slope
    orth_ab_intercept = cy - orth_ab_slope * cx

    cross_x = (orth_ab_intercept - ab_intercept) / (ab_slope - orth_ab_slope)
    cross_y = ab_slope * cross_x + ab_intercept

    return dist((cx, cy), (cross_x, cross_y))


def compute_strings(
    image: np.array,
    influence_func: Callable[[float], float],
    influence_radius: float,
    num_nails: int,
) -> np.array:
    """Compute string art connections for image, using num nails.

    Returns a num_nails x num_nails array of non-negative integers, indicating the
    number of threads that should connect each pair of nails.
    """

    img_H, img_W = image.shape
    print(f"Image shape: {img_H} x {img_W}")
    connections = (num_nails + 1) * num_nails // 2

    nail_xs = (np.cos([i / num_nails * 2 * np.pi for i in range(num_nails)]) + 1) / 2
    nail_ys = (np.sin([i / num_nails * 2 * np.pi for i in range(num_nails)]) + 1) / 2

    def nail_point(i: int):
        return (nail_xs[i], nail_ys[i])

    def connection_idx(m, n):
        # (t-1) + (t-2) + ... + (t-m) + n
        # t * m - m * (m-1) / 2
        # print(f"connection_idx({m}, {n})")
        return num_nails * m - (m + 1) * m // 2 + n - m - 1

    X = np.zeros((img_H * img_W, connections))
    Y = image.reshape(img_H * img_W)

    def compute_influence():
        for i, j in np.ndindex(image.shape):
            y, x = i / img_H, j / img_W
            if dist((x, y), (0.5, 0.5)) > 0.5:
                continue
            print(f"i={i}, j={j}")
            pixel_idx = i * img_W + j

            mark = 1
            for m in range(num_nails):
                # print(f"m={m}")
                n = mark
                while (
                    n < num_nails
                    and chord_dist(nail_point(m), nail_point(n), (x, y))
                    > influence_radius
                ):
                    n += 1
                if n == num_nails:
                    return
                while (
                    n < num_nails
                    and chord_dist(nail_point(m), nail_point(n), (x, y))
                    <= influence_radius
                ):
                    conn_idx = connection_idx(m, n)
                    cdist = chord_dist(nail_point(m), nail_point(n), (x, y))
                    X[pixel_idx, conn_idx] += influence_func(cdist)
                    n += 1

    compute_influence()

    model = torch.nn.Linear(connections, 1, bias=False)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float)
    )
    dataloader = torch.utils.data.DataLoader(dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(MAX_EPOCHS):
        print(f"Starting epoch {epoch}.")
        losses = []
        for batch in dataloader:
            x, y = batch
            y_hat = model(x)
            loss = torch.nn.functional.mse_loss(y, y_hat)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.clamp_(0)

        mean_loss = np.mean(losses)
        print(f"Epoch {epoch} complete. Mean loss {mean_loss}")

    string_weight = model.weight.data
    print(string_weight)
    return string_weight


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--num-nails", type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    image = load_image(args.image)

    max_dist = 0.5

    def linear_falloff(dist):
        return max(0, 1 - dist / max_dist)

    compute_strings(image, linear_falloff, max_dist, args.num_nails)


if __name__ == "__main__":
    main()
