import argparse
import itertools
import json
import random
import sys
from collections import defaultdict, Counter
from os.path import exists

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from .gen_mask import mask_path
from .utils import argmax, argmin, connection_idx, mse_loss, normalize 


def load_image(fname, height, width):
    img = Image.open(fname).convert("L")
    img = img.resize((height, width))
    with open("grayscale.png", "wb") as f:
        img.save(f)

    return plt.imread("grayscale.png")


def optimize_path(seq: list[int], num_nails: int) -> list[int]:
    
    graph = defaultdict(list)
    for i in range(len(seq)-1):
        graph[seq[i]].append(seq[i+1])

    loops = []

    while graph:

        start = argmax({m: len(ns) for m, ns in graph.items()})

        loop = [start]

        while loop[-1] in graph:

            m = loop[-1]
            n = argmin({i: abs((i - m) % num_nails - num_nails // 2) for i in graph[m]})
            loop.append(n)
            graph[m].remove(n)
            if len(graph[m]) == 0:
                del graph[m]

        loops.append(loop)

    loops.sort(key=len, reverse=True)

    while len(loops) > 1:
        for i1, i2 in itertools.combinations(range(len(loops)), 2):
            l1 = loops[i1]
            l2 = loops[i2]
            pos1 = [i for i, m in enumerate(l1) if m in set(l2)][0]
            n = l1[pos1]
            pos2 = l2.index(n)
            l3 = l1[:pos1] + l2[pos2:-1] + l2[:pos2] + l1[pos1:]
            loops[i1] = l3
            loops.pop(i2)
            break

    res = loops[0]
    res = res[res.index(0):-1] + res[:res.index(0)+1]

    return res


def compute_strings_discrete_loop(image: np.array, influence: np.array, num_nails: int, num_strings: int, output_dir: str) -> list[int]:

    num_connections = (num_nails - 1) * num_nails
    nonzero = {i: np.nonzero(influence[:, i])[0] for i in range(num_connections)}

    img_H, img_W = image.shape
    image = image.reshape(img_H * img_W)
    image = (image - image.min()) / (image.max() - image.min())

    avail = {i: set(range(num_nails)) - {i} for i in range(num_nails)}
    seq = [0]
    for i in range(num_strings - 1):
        if i < num_strings - 2:
            next = random.choice(list(avail[seq[-1]]))
        else:  # last string
            next = random.choice(list(avail[seq[-1]] & avail[0]))
        avail[seq[-1]].remove(next)
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
    curr_loss = mse_loss(image, normalize(strimg, smin, smax))
    print(f"Starting optimizer. loss={curr_loss}, num_strings={weights.sum()}")

    epoch = 0

    positions = list(range(num_strings - 1))

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
                before = mse_loss(irow, normalize(srow, smin, smax))
                srow += influence[np.ix_(nz, [am, mc])].sum(axis=1)
                try:
                    _smax = max(smax, srow.max())
                except:
                    print((a, middle, c))
                    print(nz)
                    raise 
                after = mse_loss(irow, normalize(srow, smin, _smax))
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

        curr_loss = mse_loss(image, normalize(strimg, smin, smax))
        print(f"End epoch {epoch}. loss={curr_loss}, num_strings={weights.sum()}")
        with open(f"seq_{epoch}.json", "w") as f:
            json.dump(optimize_path(seq, num_nails), f)

    return 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str)
    parser.add_argument("-y", "--height", type=int)
    parser.add_argument("-x", "--width", type=int)
    parser.add_argument("-n", "--num-nails", type=int)
    parser.add_argument("-f", "--nail-frac", type=float, default=0.2)
    parser.add_argument("-d", "--max-dist", default=0.1)
    parser.add_argument("-s", "--num-strings", default=2800)
    parser.add_argument("-o", "--output-dir", default="./")
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

    image = load_image(args.image, args.height, args.width)
    image = 1 - image

    string_weight = compute_strings_discrete_loop(image, mask, args.num_nails, args.num_strings)
    with open("output.dat", "wb") as f:
        np.save(f, string_weight)
    return string_weight


if __name__ == "__main__":
    main()
