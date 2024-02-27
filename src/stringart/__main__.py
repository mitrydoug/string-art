import argparse
import sys
from os.path import exists
from typing import Callable

import numpy as np
import tempfile
import torch
from PIL import Image
from matplotlib import pyplot as plt

from .gen_mask import mask_path
from .utils import dist

LEARNING_RATE = 1e-4
MAX_EPOCHS = 200


def load_image(fname, height, width):
    img = Image.open(fname).convert("L")
    img = img.resize((height, width))
    with open("grayscale.png", "wb") as f:
        img.save(f)

    return plt.imread("grayscale.png")


def compute_strings_baseline(
    image: np.array,
    influence: np.array,
    num_nails: int,
) -> np.array:
    img_H, img_W = image.shape
    print(image.shape)
    print(influence.shape)

    print(np.count_nonzero(influence))

    non_zero = np.count_nonzero(influence, axis=0)
    print(non_zero.shape)
    print(np.min(non_zero))
    print(np.count_nonzero(non_zero))

    image = image.reshape((img_H * img_W, 1))
    influence = image * influence

    print(influence.shape)

    weights = np.sum(influence, axis=0) / (non_zero + 1e-9)
    print(weights.shape)

    return weights


def compute_strings(
    image: np.array,
    influence: np.array,
    num_nails: int,
) -> np.array:
    """Compute string art connections for image, using num nails.

    Returns a num_nails x num_nails array of non-negative integers, indicating the
    number of threads that should connect each pair of nails.
    """

    img_H, img_W = image.shape
    print(f"Image shape: {img_H} x {img_W}")
    connections = (num_nails - 1) * num_nails // 2

    focus = np.array([1 if dist((j / img_W, 1 - i / img_H), (0.5, 0.5)) < 0.2 else 0.2 for j in range(img_W) for i in range(img_H)])

    T = np.stack((image.reshape(img_H * img_W), focus), axis=1)

    zeroes = np.all(influence == 0, axis=1)
    influence = influence[~zeroes]
    T = T[~zeroes]

    influence_tensor = torch.tensor(influence, dtype=torch.float)
    # nz = np.count_nonzero(influence, 0)
    # adjustment = torch.tensor(1/(nz - nz.min() + 1), dtype=torch.float)

    weights = torch.nn.parameter.Parameter(torch.zeros((connections,)))
    dataset = torch.utils.data.TensorDataset(
        influence_tensor, torch.tensor(T, dtype=torch.float)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
    optimizer = torch.optim.Adam([weights], lr=LEARNING_RATE)

    lambda_1 = 0 # 1e-5

    for epoch in range(MAX_EPOCHS):
        print(f"Starting epoch {epoch}.")
        losses = []
        for batch in dataloader:
            x, t = batch
            y = t[:,0]
            f = t[:,1]
            # print(x)
            # print(y.shape)
            # sys.exit(0)
            pos_weights = torch.nn.functional.softplus(weights, beta=20)
            # print(pos_weights)
            # print(x[0][x[0] > 0])
            y_hat = torch.matmul(x, pos_weights)
            # print(y_hat)
            # sys.exit(0)
            # print((y, y_hat))
            pixel_loss = f * torch.nn.functional.mse_loss(y_hat, y)
            loss = pixel_loss.mean() + lambda_1 * torch.norm(pos_weights, 1)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # with torch.no_grad():
            #     for parameter in model.parameters():
            #         parameter.clamp_(0)

        mean_loss = np.mean(losses)
        print(f"Epoch {epoch} complete. Mean loss {mean_loss}")

    return weights.data.numpy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str)
    parser.add_argument("-y", "--height", type=int)
    parser.add_argument("-x", "--width", type=int)
    parser.add_argument("-n", "--num-nails", type=int)
    parser.add_argument("-d", "--max-dist", default=0.1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    mask_fp = mask_path(args.height, args.width, args.num_nails, args.max_dist)
    if not exists(mask_fp):
        print(f"Mask does not exist: {mask_fn}")
        sys.exit(1)

    with open(mask_fp, "rb") as f:
        mask = np.load(f)

    print(np.max(mask))
    print(np.argmax(mask))

    image = load_image(args.image, args.height, args.width)
    image = 1 - image
    print(image)
    print(image.max())
    print(image.min())

    max_dist = 0.05

    print(f"non-zero: {np.count_nonzero(mask)}")
    nz = (mask > 0).astype(int)
    print(np.sum(nz))
    print(np.max(mask))
    mask = np.maximum(0, 1 - mask / max_dist) * nz
    print(f"non-zero: {np.count_nonzero(mask)}")
    print(np.mean(mask))

    string_weight = compute_strings(image, mask, args.num_nails)
    with open("output.dat", "wb") as f:
        np.save(f, string_weight)
    return string_weight


if __name__ == "__main__":
    main()
