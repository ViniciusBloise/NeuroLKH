import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from intf import config, constant as const
from intf.readerTSP import ReaderTSP

from intf.graph_builder import GraphBuilder
from intf.lkh_helper import solve_LKH, write_instance, eval_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for best run, candidate list and stats')
    parser.add_argument('--dataset', default='100', help='dataset path with all problem sets')
    parser.add_argument('--model_path', type=str, default='pretrained/neurolkh.pt', help='')
    parser.add_argument('--n_samples', type=int, default=1000, help='')
    parser.add_argument('--batch_size', type=int, default=100, help='')
    parser.add_argument('--lkh_trials', type=int, default=1000, help='')

    args = parser.parse_args()

    dataset = args.dataset
    method = "NeuroLKH"
    print(f'### Start generating features for dataset {args.dataset}')
    eval_dataset(dataset, method, args, rerun=True, max_trials=1000)