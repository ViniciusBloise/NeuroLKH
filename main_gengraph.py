import os
import sh
import argparse
import numpy as np
import pandas as pd

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from typing import Mapping, MutableMapping, Sequence, Iterable

from tqdm import tqdm
from intf import config, constant as const
from intf.readerTSP import ReaderTSP

import matplotlib.pyplot as plt

def problem_set(cfg: config.Config = None) -> Iterable[tuple[int, np.array, np.array, str, str, np.array]]:
    print('#### Reading TSP candidate sets through ReaderTSP')
    print('####')

    reader = ReaderTSP(cfg)
    tspdir = os.listdir(cfg.get_dir(const.RES_DIR_TSP))
    tspdir.sort()

    for filename in tspdir:  # tqdm(tspdir):
        problem_name = filename.split('.')[0]
        full_problemname = cfg.get_dir(const.RES_DIR_TSP, problem_name, '.tsp')
        problem = reader.read_instance(full_problemname)
        n_points, positions, dist_matrix, name, optimal_tour = problem

        full_pi = cfg.get_dir(const.RES_DIR_PI, problem_name)
        pis = ReaderTSP.read_pi(filename=full_pi)

        #optimal_tour = reader.get_optimal_solution(name, positions)
        print('### Solution', name, n_points)

        yield n_points, positions, dist_matrix, name, optimal_tour, pis

def normalize_data(data:np.array) -> np.array:
    return data / np.abs(data).max()

def plot_problem(pos:np.array, pis:np.array, n_buckets = 10):
    normal_size = normalize_data(pis)
    size = np.power(1.7,  abs(normal_size) * n_buckets).tolist()
    color = np.array([ 'darkblue' for i in range(normal_size.size)])
    color[np.argwhere(normal_size < 0)] = 'red'
    print(normal_size, color)

    plt.scatter(pos[:, 0], pos[:, 1], s=size, color=color, marker='o')
    plt.show()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for best run, candidate list and stats')
    parser.add_argument('--dataset', default='100', help='dataset path with all problem sets')

    args = parser.parse_args()
    cfg = config.Config()

    if args.dataset:
        cfg.TSP_INST_URL = f'result/{args.dataset}/'

    rows = []
    tspProblemSet = problem_set(cfg=cfg)

    for problem in tspProblemSet: #tdqm
        n_points, positions, dist_matrix, name, optimal_tour, pis = problem
        #print(name, pis)
        plot_problem(positions, pis)
        break

    #print(n_points, positions, dist_matrix, name, optimal_tour)