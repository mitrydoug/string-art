import argparse
import json
import random
import sys
from os.path import exists
from typing import Callable

import numpy as np
import tempfile
import torch
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from .gen_mask import mask_path
from .utils import dist, connection_idx

LEARNING_RATE = 1e-4
MAX_EPOCHS = 200


def load_image(fname, height, width):
    img = Image.open(fname).convert("L")
    img = img.resize((height, width))
    with open("grayscale.png", "wb") as f:
        img.save(f)

    return plt.imread("grayscale.png")


def compute_strings_discrete_loop(image: np.array, influence: np.array, num_nails: int):

    def normalize(arr, min, max):
        return (arr - min) / (max - min)

    def loss(img, simg):
        return np.where(img >= 0, (img - simg) ** 2, 0).sum()

    def argmin(d):
        m = min(d.items(), key=(lambda t: t[1]))
        return m[0]
    
    print(influence.shape)

    NUM_STRINGS = 2800

    num_connections = (num_nails - 1) * num_nails
    nonzero = {i: np.nonzero(influence[:, i])[0] for i in range(num_connections)}

    img_H, img_W = image.shape
    image = image.reshape(img_H * img_W)
    image = (image - image.min()) / (image.max() - image.min())

    avail = {i: set(range(num_nails)) - {i} for i in range(num_nails)}
    seq = [0]
    for i in range(NUM_STRINGS - 1):
        if i < NUM_STRINGS - 2:
            next = random.choice(list(avail[seq[-1]]))
        else:  # last string
            next = random.choice(list(avail[seq[-1]] & avail[0]))
        avail[seq[-1]].remove(next)
        avail[next].remove(seq[-1])
        seq.append(next)
    seq.append(0)

    one_idxs = []
    for i in range(len(seq) - 1):
        s = seq[i]
        e = seq[i + 1]
        one_idxs.append(connection_idx(s, e, num_nails))

    weights = np.zeros(num_connections)
    weights[one_idxs] = 1

    improving = True

    strimg = np.matmul(influence, weights)
    smin = strimg.min()
    smax = strimg.max()
    curr_loss = loss(image, normalize(strimg, smin, smax))
    print(f"Starting optimizer. loss={curr_loss}, num_strings={weights.sum()}")

    epoch = 0

    positions = list(range(NUM_STRINGS - 1))

    while improving:

        epoch += 1
        improving = False
        random.shuffle(positions)

        for pos in tqdm(positions):

            # existing triangle
            a = seq[pos]
            b = seq[pos + 1]
            c = seq[pos + 2]

            # remove existing
            ab = connection_idx(a, b, num_nails)
            bc = connection_idx(b, c, num_nails)

            assert weights[ab] == 1
            assert weights[bc] == 1

            weights[ab] = 0
            weights[bc] = 0
            strimg[nonzero[ab]] -= influence[nonzero[ab], ab]
            strimg[nonzero[bc]] -= influence[nonzero[bc], bc]
            smin = strimg.min()
            smax = strimg.max()

            loss_delta = {}
            for middle in range(num_nails):
                if middle == a or middle == c:
                    continue
                am = connection_idx(a, middle, num_nails)
                mc = connection_idx(middle, c, num_nails)
                if weights[am] == 1 or weights[mc] == 1:
                    continue

                nz = np.union1d(nonzero[am], nonzero[mc])
                if nz.size == 0:
                    continue
                irow = image[nz]
                srow = strimg[nz]
                before = loss(irow, normalize(srow, smin, smax))
                srow += influence[np.ix_(nz, [am, mc])].sum(axis=1)
                try:
                    _smax = max(smax, srow.max())
                except:
                    print((a, middle, c))
                    print(nz)
                    raise 
                after = loss(irow, normalize(srow, smin, _smax))
                delta = after - before
                loss_delta[middle] = delta
 
            best_middle = argmin(loss_delta)

            if best_middle != seq[pos+1]:
                improving = True

            seq[pos+1] = best_middle
            am = connection_idx(a, best_middle, num_nails)
            mc = connection_idx(best_middle, c, num_nails)

            assert weights[am] == 0 and weights[mc] == 0

            weights[am] = 1
            weights[mc] = 1
            strimg[nonzero[am]] += influence[nonzero[am], am]
            strimg[nonzero[mc]] += influence[nonzero[mc], mc]
            smin = strimg.min()
            smax = strimg.max()

        curr_loss = loss(image, normalize(strimg, smin, smax))
        print(f"End epoch {epoch}. loss={curr_loss}, num_strings={weights.sum()}")
        with open(f"weights_{epoch}.dat", "wb") as f:
            np.save(f, weights)
        with open(f"seq_{epoch}.json", "w") as f:
            json.dump(seq, f)

    return weights


def compute_strings_discrete(
    image: np.array,
    influence: np.array,
    num_nails: int,
):

    def normalize(arr, min, max):
        return (arr - min) / (max - min)

    def loss(img, simg):
        return np.where(img >= 0, (img - simg) ** 2, 0).sum()

    def argmin(d):
        m = min(d.items(), key=(lambda t: t[1]))
        return m[0]

    MAX_STRINGS = 2800

    num_connections = (num_nails - 1) * num_nails // 2
    nonzero = {i: np.nonzero(influence[:, i])[0] for i in range(num_connections)}
    print(len(nonzero[100]))
    print(nonzero[100])

    # normalize image
    img_H, img_W = image.shape
    image = image.reshape(img_H * img_W)
    imin = image[image >= 0].min()
    image = np.where(image >= 0, (image - imin) / (image.max() - imin), image)

    num_strings = int(0.8 * MAX_STRINGS)
    idxs = set(range(num_connections))
    one_idxs = random.sample(sorted(idxs), num_strings)
    weights = np.zeros(num_connections)
    weights[one_idxs] = 1

    nails = [(m, n) for m in range(num_nails - 1) for n in range(m + 1, num_nails)]
    improving = True

    strimg = np.matmul(influence, weights)
    smin = strimg.min()
    smax = strimg.max()
    curr_loss = loss(image, normalize(strimg, smin, smax))
    print(f"Starting optimizer. loss={curr_loss}, num_strings={weights.sum()}")

    epoch = 0

    while improving:

        epoch += 1
        improving = False
        random.shuffle(nails)
        one_deltas = dict()
        zero_deltas = dict()

        for m, n in tqdm(nails, total=len(nails)):
            idx = connection_idx(m, n, num_nails)
            # 1 if weights[idx] == 0 else -1
            sign = 1 - 2 * weights[idx]

            # compute change in loss
            irow = image[nonzero[idx]]
            srow = strimg[nonzero[idx]]
            before = loss(irow, normalize(srow, smin, smax))
            srow += sign * influence[nonzero[idx], idx]
            _smin = min(smin, srow.min())
            _smax = max(smax, srow.max())
            after = loss(irow, normalize(srow, _smin, _smax))
            delta = after - before

            if sign < 0:
                one_deltas[idx] = delta
            else:
                zero_deltas[idx] = delta

            cand_one = argmin(one_deltas) if one_deltas else None
            cand_zero = argmin(zero_deltas) if zero_deltas else None

            if cand_one and one_deltas[cand_one] < 0:
                # remove string
                weights[cand_one] = 0
                strimg[nonzero[cand_one]] -= influence[nonzero[cand_one], cand_one]
                smin = strimg.min()
                smax = strimg.max()
                num_strings -= 1
                one_deltas.clear()
                zero_deltas.clear()
                improving = True

            elif cand_zero and zero_deltas[cand_zero] < 0 and num_strings < MAX_STRINGS:
                # add string
                weights[cand_zero] = 1
                strimg[nonzero[cand_zero]] += influence[nonzero[cand_zero], cand_zero]
                smin = strimg.min()
                smax = strimg.max()
                num_strings += 1
                one_deltas.clear()
                zero_deltas.clear()
                improving = True

            elif (
                cand_one
                and cand_zero
                and one_deltas[cand_one] + zero_deltas[cand_zero] < 0
            ):

                weights[cand_one] = 0
                weights[cand_zero] = 1

                # compute change in loss
                before = loss(image, normalize(strimg, smin, smax))
                strimg[nonzero[cand_one]] -= influence[nonzero[cand_one], cand_one]
                strimg[nonzero[cand_zero]] += influence[nonzero[cand_zero], cand_zero]
                smin = strimg.min()
                smax = strimg.max()
                after = loss(image, normalize(strimg, smin, smax))

                if after > before:
                    # undo everything
                    weights[cand_one] = 1
                    weights[cand_zero] = 0
                    strimg[nonzero[cand_one]] += influence[nonzero[cand_one], cand_one]
                    strimg[nonzero[cand_zero]] -= influence[
                        nonzero[cand_zero], cand_zero
                    ]
                    smin = strimg.min()
                    smax = strimg.max()

                else:
                    one_deltas.clear()
                    zero_deltas.clear()
                    improving = True

        curr_loss = loss(image, normalize(strimg, smin, smax))
        print(f"End epoch {epoch}. loss={curr_loss}, num_strings={weights.sum()}")
        with open(f"weights_{epoch}.dat", "wb") as f:
            np.save(f, weights)

    return weights


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

    weights = np.sum(influence, axis=0)  # / (non_zero + 1e-9)
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

    image = image.reshape(img_H * img_W)
    image = (image - image.min()) / (image.max() - image.min())

    influence = influence / influence.sum(axis=1).mean()

    focus = np.array(
        [
            1 if dist((j / img_W, 1 - i / img_H), (0.5, 0.5)) < 0.2 else 0.2
            for j in range(img_W)
            for i in range(img_H)
        ]
    )

    T = np.stack((image, focus), axis=1)

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

    lambda_1 = 0  # 1e-5

    for epoch in range(MAX_EPOCHS):
        print(f"Starting epoch {epoch}.")
        losses = []
        for batch in dataloader:
            x, t = batch
            y = t[:, 0]
            f = t[:, 1]
            # print(x)
            # print(y.shape)
            # sys.exit(0)
            # pos_weights = torch.nn.functional.softplus(weights, beta=20)
            sig_weights = torch.nn.functional.sigmoid(weights)
            # print(pos_weights)
            # print(x[0][x[0] > 0])
            y_hat = torch.matmul(x, sig_weights)
            # print(y_hat)
            # sys.exit(0)
            # print((y, y_hat))
            pixel_loss = torch.nn.functional.mse_loss(y_hat, y)
            loss = pixel_loss.mean() + lambda_1 * torch.norm(weights, 1)
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
    parser.add_argument("-f", "--nail-frac", type=float, default=0.2)
    parser.add_argument("-d", "--max-dist", default=0.1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    mask_fp = mask_path(args.height, args.width, args.num_nails, args.nail_frac, args.max_dist)
    if not exists(mask_fp):
        print(f"Mask does not exist: {mask_fp}")
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

    print(f"non-zero: {np.count_nonzero(mask)}")
    print(mask.shape)
    # nz = (mask > 0).astype(int)
    # print(np.sum(nz))
    print(np.max(mask))
    # mask = np.maximum(0, 1 - mask / max_dist) * nz
    print(f"non-zero: {np.count_nonzero(mask)}")
    print(np.mean(mask))

    string_weight = compute_strings_discrete_loop(image, mask, args.num_nails)
    with open("output.dat", "wb") as f:
        np.save(f, string_weight)
    return string_weight


if __name__ == "__main__":
    main()
