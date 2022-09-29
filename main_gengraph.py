import os
from tokenize import Name
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
import matplotlib.axes as axes
import matplotlib.figure as figure
from matplotlib.collections import LineCollection

MAX_NUMBER = 1000000

def problem_set(cfg: config.Config = None) -> Iterable[tuple[int, np.array, np.array, str, np.array, np.array]]:
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

        optimal_tour = reader.get_from_LKH_run(name)
        print('Candidate ', name)
        cands = reader.load_candidate(name)
        print('### Solution', name, n_points) #, optimal_tour, cands)

        yield n_points, positions, dist_matrix, name, optimal_tour, pis, cands

def normalize_data(data:np.array) -> np.array:
    return data / np.abs(data).max()

def plot_optimal_tour(pos:np.array, optimal:np.array):
    x = pos[:,0].flatten()
    y = pos[:,1].flatten()

    norm_pos = pos/MAX_NUMBER
    for i in range(len(pos)):
        plt.plot(norm_pos[optimal[i:i+2], 0], norm_pos[optimal[i:i+2], 1], '--m', linewidth=1.5)

    return

def plot_problem(pos:np.array, pis:np.array, name='', n_buckets = 10, fig=None, ax = None):
    norm_pos = pos / MAX_NUMBER
    normal_size = normalize_data(pis)
    size = np.power(1.7,  abs(normal_size) * n_buckets).tolist()
    color = np.array([ 'blue' for i in range(normal_size.size)])
    color[np.argwhere(normal_size < 0)] = 'red'
    #print(normal_size, color)

    ax.scatter(norm_pos[:, 0], norm_pos[:, 1], s=size, color=color, marker='o')
    ax.set_title(f'Nodes: {len(size)}, Sample: {name}')

    return

def plot_subline(x1, x2, y1, y2, lw, ax):
    _SPACING = 1000
    _COLORS = ['darkred', 'orange', 'gold', 'green', 'blue']
    _PIECE = 4

    x = np.linspace(x1, x2, _SPACING)
    b = (y2 - y1)/(x2 - x1) 
    y = lambda j: b * j + y1 - b * x1

    points = np.array([x, y(x)]).T.reshape(-1, 1, 2)

    segments = np.concatenate([points[:_SPACING//_PIECE-1], points[1:_SPACING//_PIECE]], axis=1)
    #print(segments)
    lc = LineCollection(segments, linewidths=lw*4/5,color= _COLORS[5-lw])
  
    ax.add_collection(lc)
    return

def plot_numbers(pos:np.array, fig, ax: axes.Axes):
    norm_pos = pos / MAX_NUMBER
    i = 0
    for point in norm_pos:
        ax.annotate(str(i + 1), (point[0], point[1]))
        i += 1

def plot_candidates(pos:np.array, candidate:np.array, fig, ax: axes.Axes):
    index = 0
    norm_pos = pos / MAX_NUMBER

    for points in candidate:
        x1 = norm_pos[index,0]; y1 = norm_pos[index,1]
        lw = 5
        for endpoint in points:
            x2 = norm_pos[endpoint,0]; y2 = norm_pos[endpoint,1]
      
            #plt.plot( [x1, x2], [y1, y2], color='lightgray', linewidth=1)
            plot_subline(x1, x2, y1, y2, lw = lw, ax = ax)
            lw -= 1
        index += 1
    
    #fig.show()

    return
def plot_save(cfg, name, fig:figure.Figure, ax: axes.Axes):
    file = f'{cfg.TSP_INST_URL}pics/{name}.png'
    os.makedirs(f"{cfg.TSP_INST_URL}pics", exist_ok=True)

    fig.savefig(file)
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

    MAX_PROBLEM = 10
    i = 0
    for problem in tspProblemSet: #tdqm
        n_points, positions, dist_matrix, name, optimal_tour, pis, cand = problem
        #print('### going to plot', name)

        fig, ax = plt.subplots()

        
        plot_candidates(pos=positions, candidate=cand, fig=fig, ax=ax)
        plot_problem(positions, pis, name=name, fig=fig, ax=ax)
        plot_optimal_tour(pos=positions, optimal=optimal_tour)
        #plot_numbers(pos=positions, fig=fig, ax=ax)
        
        plot_save(cfg, name, fig=fig, ax=ax)
        i += 1
        if i > MAX_PROBLEM:
            break

    #plt.show()
    #print(n_points, positions, dist_matrix, name, optimal_tour)