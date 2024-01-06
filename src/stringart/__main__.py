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

LEARNING_RATE = 1e-4
MAX_EPOCHS = 20


def load_image(fname, height, width):
    img = Image.open(fname).convert("L")
    img = img.resize((height, width))
    with open("grayscale.png", "wb") as f:
        img.save(f)

    return plt.imread("grayscale.png")


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

    Y = image.reshape(img_H * img_W)

    zeroes = np.all(influence == 0, axis=1)
    influence = influence[~zeroes]
    Y = Y[~zeroes]

    # influence_tensor = 

    weights = torch.nn.parameter.Parameter(torch.zeros((connections,)))
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(influence, dtype=torch.float), torch.tensor(Y, dtype=torch.float)
    )
    dataloader = torch.utils.data.DataLoader(dataset)
    optimizer = torch.optim.Adam([weights], lr=LEARNING_RATE)

    for epoch in range(MAX_EPOCHS):
        print(f"Starting epoch {epoch}.")
        losses = []
        for batch in dataloader:
            x, y = batch
            pos_weights = torch.nn.functional.softplus(weights, beta=20)
            # print(pos_weights)
            # print(x[0][x[0] > 0])
            y_hat = torch.dot(pos_weights, x[0])
            # print((y, y_hat))
            loss = torch.nn.functional.mse_loss(y_hat, y[0])
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
        print("Mask does not exist: {mask_fn}")
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
