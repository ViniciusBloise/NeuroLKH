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


def generate_bestrun_files(cfg=None):
    print('#### Generate best run files from tsp ###')
    print('####')

    if cfg is None:
        cfg = config.Config()

    instance = cfg.instance
    dirname = cfg.get_dir(const.RES_DIR_BESTRUN)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    lkh_log = cfg.get_dir(const.RES_DIR_LKH_LOG)
    if not os.path.exists(lkh_log):
        os.mkdir(lkh_log)
    tspdir = os.listdir(lkh_log)

    for log_filename in tqdm(tspdir):
        full_logfilename = cfg.get_dir(const.RES_DIR_LKH_LOG, log_filename, '.log')

        bestrun = sh.tail('-n 1', full_logfilename)
        bestrun_file = cfg.get_dir(const.RES_DIR_BESTRUN, log_filename, '.txt')
        # print(bestrun_file)
        with open(bestrun_file, 'w') as f:
            br = str(bestrun.wait())[:-1].strip()
            f.write(br)

    print('# end #')


def read_candidate(cfg: config.Config = None) -> dict[str, dict[str, list[int]]]:
    # returne a dictionary with set of problems
    print('#### reading candidates')
    print('####')

    if cfg is None:
        cfg = config.Config()

    candidatedir = os.listdir(cfg.get_dir(const.RES_DIR_CANDIDATE))

    problem: dict[str, dict[str, list[int]]] = {}

    for cand_filename in tqdm(candidatedir):
        problem_name = cand_filename.split('.')[0]
        problem[problem_name] = None

        full_candfilename = cfg.get_dir(const.RES_DIR_CANDIDATE, cand_filename)
        # print(full_candfilename)
        with open(full_candfilename, 'r') as f:
            cand = f.read()
            # print(cand)
            parts = proc_candidate_file(cand)
        problem[problem_name] = parts
    return problem


def read_bestrun(cfg: config.Config = None):
    print('#### reading bestrun')
    print('####')

    if cfg is None:
        cfg = config.Config()

    candidatedir = os.listdir(cfg.get_dir(const.RES_DIR_BESTRUN))

    problem: dict[str, list[tuple[int,int]]] = {}

    for cand_filename in candidatedir:
        problem_name = cand_filename.split('.')[0]
        problem[problem_name] = None

        full_candfilename = cfg.get_dir(const.RES_DIR_BESTRUN, cand_filename)
        # print(full_candfilename)
        parts = []
        with open(full_candfilename, 'r') as f:
            bestrun = f.read().strip()
            # print(cand)
            a = None
            for node in bestrun.split(' '):
                if a is not None:
                    b = int(node) - 1
                    item = (a,b) if a < b else (b,a)
                    parts.append(item)
                a = int(node) - 1
            
        problem[problem_name] = parts

    return problem


def proc_candidate_file(filetext: str):
    # candidate files are ranging from 1 -> nodes
    # return a dict of nodes and candidates edge according to a priority

    parts: dict[str, list[int]] = {}
    lines = filetext.split('\n')
    n_nodes = int(lines[0])
    for line in lines[1:n_nodes+1]:  # tqdm
        i = 0
        for part in line.split(' '):
            if i == 0:
                node = str(int(part) - 1)
                parts[node] = []
            elif i > 1 and i % 2 == 1:
                parts[node].append(int(part) - 1)
            i += 1
    return parts


def get_edges_from_candidate(candidates, pos=0):
    edges = []
    for key in candidates:
        a = int(key)
        b = candidates[key][pos]
        tuple = (a, b) if(a < b) else (b, a)
        edges.append(tuple)
    return edges


def problem_set2(cfg: config.Config = None):
    print('#### Reading TSP candidate sets')
    print('####')

    tspdir = os.listdir(cfg.get_dir(const.RES_DIR_TSP))

    problem: dict[str, list[tuple(float, float)]] = {}
    for filename in tqdm(tspdir):
        problem_name = filename.split('.')[0]
        full_problemname = cfg.get_dir(const.RES_DIR_TSP, problem_name, '.tsp')
        points: list[tuple(float, float)] = []
        # print(full_candfilename)

        print(full_problemname)
        with open(full_problemname, 'r') as f:
            probls = f.readlines()
            start_idx = [idx for idx, x in enumerate(
                probls) if 'NODE_COORD_SECTION' in x][0]
            end_idx = [idx for idx, x in enumerate(probls) if 'EOF' in x][0]
            print(probls[start_idx])
            for line in probls[start_idx+1:end_idx]:

                items = line.strip().split(' ')[1:]
                point = (float(items[0]), float(items[1]))
                points.append(point)

            # print(cand)
        problem[problem_name] = points
        break
    return problem


def problem_set(cfg: config.Config = None) -> Iterable[tuple[int, np.array, np.array, str, str]]:
    print('#### Reading TSP candidate sets through ReaderTSP')
    print('####')

    reader = ReaderTSP(cfg)
    tspdir = os.listdir(cfg.get_dir(const.RES_DIR_TSP))

    for filename in tspdir:  # tqdm(tspdir):
        problem_name = filename.split('.')[0]
        full_problemname = cfg.get_dir(const.RES_DIR_TSP, problem_name, '.tsp')
        problem = reader.read_instance(full_problemname)
        yield problem


def compute_minimum_spanning_tree(dist_matrix: np.array) -> list[tuple[int, int]]:
    X = np.triu(dist_matrix, 0)
    Tcsr = minimum_spanning_tree(csr_matrix(X))

    edges = []
    for edge in np.argwhere(Tcsr > 0):
        a = edge[0]
        b = edge[1]

        edge_tpl = (a, b) if (a < b) else (b, a)
        edges.append(edge_tpl)
    return edges


def intersect(list1: list[tuple[int, int]], list2: list[tuple[int, int]]):
    return list(set(list1) & set(list2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for best run, candidate list and stats')
    parser.add_argument('--dataset', default='100', help='dataset path with all problem sets')
    parser.add_argument('-bestrun', action='store_const', const=True, help='flag to generate bestrun files')

    args = parser.parse_args()
    cfg = config.Config()

    if args.dataset:
        cfg.TSP_INST_URL = f'result/{args.dataset}/'

    if args.bestrun:
        generate_bestrun_files(cfg=cfg)

    candidates = read_candidate(cfg=cfg)

    bestrun = read_bestrun(cfg=cfg)

    # print(first)
    tspProblemSet = problem_set(cfg=cfg)
    once = True
    print()
    rows = []
    for problem in tqdm(tspProblemSet):
        n_points, positions, dist_matrix, name, optimal_tour = problem
        if once:
            mst = compute_minimum_spanning_tree(dist_matrix)

            if name not in candidates:
                continue
            first = candidates[name]
            edges = get_edges_from_candidate(first)
            # print(len(edges))
            # print(len(mst))
            intx = intersect(edges, mst)
            intBest = intersect(intx, bestrun[name])
            # print(len(intx))

            row = [name, n_points, len(edges), len(mst), len(bestrun), 0, len(intx), len(intBest)]
        rows.append(row)
        #print(n_points, name)
    print(row)

    print('### creating stats file')
    df = pd.DataFrame(rows, columns=['name', 'n#points', 'n#candidates', 'n#mst', 'n#best', 'nothing','n#intx_cand_mst', 'n#intx_cand_mst_best'])

    dirstats = cfg.get_dir(const.RES_DIR_STATS)
    if not os.path.exists(dirstats):
        os.mkdir(dirstats)

    statsfile = dirstats + f'stats{cfg.instance}.txt'
    df.to_csv(statsfile)

    print('### end generation')
    
    #oneProblem = tspProblemSet
    # oneProblem
    #print(name, n_points)
    #keys = list(tspProblemSet.keys())
    # print(len(keys))s
