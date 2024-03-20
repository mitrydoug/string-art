import argparse
import itertools
import json
import random
import sys
from collections import defaultdict, Counter
from os.path import exists, join

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from .gen_mask import mask_path
from .utils import argmax, argmin, connection_idx, mse_loss, normalize 


def load_image(fname, height, width):
    """Load image as ndarray with specified shape"""
    img = Image.open(fname).convert("L")
    img = img.resize((height, width))
    with open("grayscale.png", "wb") as f:
        img.save(f)

    return plt.imread("grayscale.png")


def optimize_path(seq: list[int], num_nails: int) -> list[int]:
    
    # directed adjacency list representation
    graph = defaultdict(list)
    for i in range(len(seq)-1):
        graph[seq[i]].append(seq[i+1])

    # partition graph edges into collection of cycles
    cycles = []

    while graph:

        # start from node with largest out degree, hoping to maximize
        # cycle length
        start = argmax({m: len(ns) for m, ns in graph.items()})
        cycle = [start]

        while cycle[-1] in graph:
            
            m = cycle[-1]
            # choose next node furthest possible across ring
            n = argmin({i: abs((i - m) % num_nails - num_nails // 2) for i in graph[m]})
            cycle.append(n)
            graph[m].remove(n)
            if len(graph[m]) == 0:
                del graph[m]

        cycles.append(cycle)

    # sort cycles by size, largest first
    cycles.sort(key=len, reverse=True)

    # one by one, merge cycles together, until one cycle remains
    while len(cycles) > 1:
        for i1, i2 in itertools.combinations(range(len(cycles)), 2):
            c1 = cycles[i1]
            c2 = cycles[i2]
            intersection = [i for i, m in enumerate(c1) if m in set(c2)]
            if len(intersection) == 0:
                continue
            # cycles share a node, so they can be merged
            pos1 = intersection[0]
            n = c1[pos1]
            pos2 = c2.index(n)
            # slice cycles together
            l3 = c1[:pos1] + c2[pos2:-1] + c2[:pos2] + c1[pos1:]
            cycles[i1] = l3
            cycles.pop(i2)
            break

    res = cycles[0]
    res = res[res.index(0):-1] + res[:res.index(0)+1]

    return res


def compute_strings_discrete_loop(image: np.array, influence: np.array, num_nails: int, num_strings: int, output_dir: str) -> list[int]:
    """Compute a cycle which optimizes the mean-squared error between string image approximation and reference image.
    
    params:
      image: H x W image array
      influence: (H*W) x num_connections array, each column C represents the influence connection C
        has on each pixel of the string image
      num_nails: number of nails around ring
      num_string: the number of edges the resulting cycle should contain
      output_dir: directory location where checkpoint sequences should be saved
    """

    # i -> j and j -> i treated as unique connections    
    num_connections = (num_nails - 1) * num_nails
    # for each connection, store a mask of image pixels influenced by the connection.
    # used for optimization
    nonzero = {i: np.nonzero(influence[:, i])[0] for i in range(num_connections)}

    # flatten image
    img_H, img_W = image.shape
    image = image.reshape(img_H * img_W)
    # normalize image
    image = (image - image.min()) / (image.max() - image.min())

    # generate a random starting sequence. Don't allow a repeated edge.
    # Note: 12 -> 15 and 15 -> 12 are distinct connections
    avail = {i: set(range(num_nails)) - {i} for i in range(num_nails)}
    cycle = [0]
    for i in range(num_strings - 1):
        if i < num_strings - 2:
            next = random.choice(list(avail[cycle[-1]]))
        else:  # last string
            next = random.choice(list(avail[cycle[-1]] & avail[0]))
        avail[cycle[-1]].remove(next)
        cycle.append(next)
    cycle.append(0)

    # compute weights array
    one_idxs = []
    for i in range(len(cycle) - 1):
        s = cycle[i]
        e = cycle[i + 1]
        one_idxs.append(connection_idx(s, e, num_nails))

    # helps keep track of which connections are included in the current best
    # cycle
    weights = np.zeros(num_connections)
    weights[one_idxs] = 1

    improving = True

    # compute an approximation of the image that would be created by strings
    # following the path of cycle
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
        # traverse positions in random order for each epoch
        random.shuffle(positions)

        for pos in tqdm(positions):

            # triangle of nodes a -> b -> c
            a = cycle[pos]
            b = cycle[pos + 1]
            c = cycle[pos + 2]

            # remove existing strings
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

            # find the best node to replace b as the "middle" node
            # in the original triangle
            middle_losses = {}
            for middle in range(num_nails):
                if middle == a or middle == c:
                    continue
                am = connection_idx(a, middle, num_nails)
                mc = connection_idx(middle, c, num_nails)
                if weights[am] == 1 or weights[mc] == 1:
                    # can't traverse a connection (aka edge) twice
                    # in the cycle 
                    continue

                # all pixels influenced by a->middle or middle->c
                nz = np.union1d(nonzero[am], nonzero[mc])
                if nz.size == 0:
                    continue

                # compute before and after loss
                irow = image[nz]
                srow = strimg[nz]
                srow += influence[np.ix_(nz, [am, mc])].sum(axis=1)
                try:
                    _smax = max(smax, srow.max())
                except:
                    print((a, middle, c))
                    print(nz)
                    raise 
                middle_loss = mse_loss(irow, normalize(srow, smin, _smax))
                middle_losses[middle] = middle_loss
 
            best_middle = argmin(middle_losses)

            if best_middle != cycle[pos+1]:
                # we're making a change
                improving = True

            cycle[pos+1] = best_middle
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
        with open(join(output_dir, f"cycle_{epoch}.json", "w")) as f:
            json.dump(optimize_path(cycle, num_nails), f)

    return optimize_path(cycle)


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
    # we want light pixels to be 0 and dark pixels to be 1
    image = 1 - image

    cycle = compute_strings_discrete_loop(image, mask, args.num_nails, args.num_strings, args.output_dir)
    with open(join(args.output_dir, "cycle_final.json"), "wb") as f:
        json.dump(cycle, f)


if __name__ == "__main__":
    main()
