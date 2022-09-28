import os
from subprocess import check_call
from multiprocessing import Pool
import time
import tempfile
import sh
import tqdm
from tqdm import trange

import numpy as np
import pickle
import torch
from torch.autograd import Variable

from net.sgcn_model import SparseGCNModel


_base_dir = ''
_DATASET_MAX_SCALE = 1000000


def write_instance(instance, instance_name, instance_filename):
    with open(_base_dir + instance_filename, "w") as f:
        f.write("NAME : " + instance_name + "\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : TSP\n")
        f.write("DIMENSION : " + str(len(instance)) + "\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        s = _DATASET_MAX_SCALE
        for i in range(len(instance)):
            f.write(" " + str(i + 1) + " " + str(instance[i][0] * s)[
                    :10] + " " + str(instance[i][1] * s)[:10] + "\n")
        f.write("EOF\n")


def write_para(dataset_name, instance_name, instance_filename, method, para_filename, max_trials=1000, seed=1234):
    with open(para_filename, "w") as f:
        f.write("PROBLEM_FILE = " + instance_filename + "\n")
        f.write("MAX_TRIALS = " + str(max_trials) + "\n")
        f.write("MOVE_TYPE = 5\nPATCHING_C = 3\nPATCHING_A = 2\nRUNS = 1\n")
        f.write("SEED = " + str(seed) + "\n")
        if method == "NeuroLKH":
            f.write("SUBGRADIENT = NO\n")
            f.write("CANDIDATE_FILE = result/" + dataset_name +
                    "/candidate/" + instance_name + ".txt\n")
            f.write("Pi_FILE = result/" + dataset_name +
                    "/pi/" + instance_name + ".txt\n")
        elif method == "FeatGenerate":
            f.write("GerenatingFeature\n")
            f.write("Feat_FILE = result/" + dataset_name +
                    "/feat/" + instance_name + ".txt\n")
        else:
            assert method == "LKH"

# edge: 1 x n_nodes x 20
def read_feat(feat_filename:str) -> tuple[np.array, np.array, np.array, float]:
    edge_index = []
    edge_feat = []
    inverse_edge_index = []
    with open(feat_filename, "r") as f:
        lines = f.readlines()
        for line in lines[:-1]:
            line = line.strip().split()
            for i in range(20):
                edge_index.append(int(line[i * 3]))
                edge_feat.append(int(line[i * 3 + 1]) / _DATASET_MAX_SCALE)
                inverse_edge_index.append(int(line[i * 3 + 2]))
    edge_index = np.array(edge_index).reshape(1, -1, 20)
    edge_feat = np.array(edge_feat).reshape(1, -1, 20)
    inverse_edge_index = np.array(inverse_edge_index).reshape(1, -1, 20)
    runtime = float(lines[-1].strip())
    return edge_index, edge_feat, inverse_edge_index, runtime


def write_candidate_pi(dataset_name, instance_name, candidate, pi, beta):
    n_node = candidate.shape[0]
    with open(f"result/{dataset_name}/candidate/{instance_name}.txt", "w") as f:
        f.write(str(n_node) + "\n")
        for j in range(n_node):
            line = str(j + 1) + " 0 5"
            for _ in range(5):
                line += " " + \
                    str(int(candidate[j, _]) + 1) + " " + str(_ * 100)
            f.write(line + "\n")
        f.write("-1\nEOF\n")

    with open(f"result/{dataset_name}/pi/{instance_name}.txt", "w") as f:
        f.write(str(n_node) + "\n")
        for j in range(n_node):
            line = str(j + 1) + " " + str(int(pi[j]))
            f.write(line + "\n")
        f.write("-1\nEOF\n")
    
    with open(f"result/{dataset_name}/beta/{instance_name}.txt", "w") as f:
        f.write(str(n_node) + "\n")
        for j in range(n_node):
            # line = str(j + 1) + " " 
            # for k in range(20):
            #     line += " " + str(int(beta[j,k]))
            line = str(j + 1) + " " + str(int(pi[j]))
            f.write(line + "\n")
        f.write("-1\nEOF\n")

def method_wrapper(args):
    if args[0] == "LKH":
        return solve_LKH(*args[1:])
    elif args[0] == "NeuroLKH":
        return solve_NeuroLKH(*args[1:])
    elif args[0] == "FeatGen":
        return generate_feat(*args[1:])


def solve_LKH(dataset_name, instance, instance_name, rerun=False, max_trials=1000):
    para_filename = f"result/{dataset_name}/LKH_para/{instance_name}.para"
    log_filename = f"result/{dataset_name}/LKH_log/{instance_name}.log"
    instance_filename = f"result/{dataset_name}/tsp/{instance_name}.tsp"
    if rerun or not os.path.isfile(log_filename):
        #write_instance(instance, instance_name, instance_filename)
        write_para(dataset_name, instance_name, instance_filename,
                   "LKH", para_filename, max_trials=max_trials)
        with open(log_filename, "w") as f:
            check_call(["./LKH", para_filename], stdout=f)
        
        #Write bestrun file
        bestrun_file = f'result/{dataset_name}/bestrun/{instance_name}.txt'
        bestrun = sh.tail('-n 1', log_filename)
        with open(bestrun_file, 'w') as f:
            f.write(str(bestrun.wait()))
    return read_results(log_filename, max_trials)


def generate_feat(dataset_name, instance, instance_name)-> tuple[np.array, np.array, np.array, float]:
    para_filename = f"result/{dataset_name}/featgen_para/{instance_name}.para"
    instance_filename = f"result/{dataset_name}/tsp/{instance_name}.tsp"
    feat_filename = f"result/{dataset_name}/feat/{instance_name}.txt"

    write_instance(instance, instance_name, instance_filename)
    write_para(dataset_name, instance_name, instance_filename,
               "FeatGenerate", para_filename)
    with tempfile.TemporaryFile() as f:
        check_call(["./LKH", para_filename], stdout=f)
    return read_feat(feat_filename)

# dataset_node_feat: n_samples x n_nodes x 2
# dataset_edge_index: n_samples x n_nodes x n_neighbours
# dataset_edge_feat: n_samples x n_nodes x n_neighbours
# inverse_edge_index: n_samples x n_nodes x n_neighbours
def infer_SGN(net, dataset_node_feat, dataset_edge_index, dataset_edge_feat, dataset_inverse_edge_index,
              batch_size=100):
    candidate = []
    pi = []
    beta = []
    for i in trange(dataset_edge_index.shape[0] // batch_size):
        node_feat = dataset_node_feat[i * batch_size:(i + 1) * batch_size]
        edge_index = dataset_edge_index[i * batch_size:(i + 1) * batch_size]
        edge_feat = dataset_edge_feat[i * batch_size:(i + 1) * batch_size]
        inverse_edge_index = dataset_inverse_edge_index[i * batch_size:(i + 1) * batch_size]

        node_feat = Variable(torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False)
        edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(
            batch_size, -1, 1)
        edge_index = Variable(torch.FloatTensor(edge_index).type(torch.cuda.FloatTensor), requires_grad=False).view(
            batch_size, -1)
        inverse_edge_index = Variable(torch.FloatTensor(inverse_edge_index).type(torch.cuda.FloatTensor), requires_grad=False).view(
            batch_size, -1)
        y_edges, _, y_nodes = net.forward(
            node_feat, edge_feat, edge_index, inverse_edge_index, None, None, 20)
        pi.append(y_nodes.cpu().numpy())
        y_edges = y_edges.detach().cpu().numpy() #n_batch x (n_nodes x n_neighbours) x 2

        y_edges = y_edges[:, :, 1].reshape(
            batch_size, dataset_node_feat.shape[1], 20) #n_batch x n_nodes x n_neighbours(20)
        print('y_edges(2): ', y_edges.shape)
        one_beta = np.copy(y_edges)

        y_edges = np.argsort(-y_edges, -1)
        edge_index = edge_index.cpu().numpy().reshape(-1, y_edges.shape[1], 20)
        candidate_index = edge_index[
            np.arange(batch_size).reshape(-1, 1, 1), np.arange(y_edges.shape[1]).reshape(1, -1, 1), y_edges]
        print('candidate_index:', candidate_index.shape)
        candidate.append(candidate_index[:, :, :5])
        beta_index = one_beta[ np.arange(batch_size).reshape(-1, 1, 1), np.arange(y_edges.shape[1]).reshape(1, -1, 1), y_edges]
        print('beta_index: ', beta_index.shape)
        beta.append(beta_index[:, :, :5])

    candidate = np.concatenate(candidate, 0)
    pi = np.concatenate(pi, 0)
    beta = np.concatenate(beta, 0)
    
    candidate_Pi = np.concatenate(
        [candidate.reshape(dataset_edge_index.shape[0], -1), _DATASET_MAX_SCALE * pi.reshape(dataset_edge_index.shape[0], -1)], -1)
    return candidate_Pi, beta.reshape(dataset_edge_index.shape[0], -1)


def solve_NeuroLKH(dataset_name, instance, instance_name, candidate, pi, beta, rerun=False, max_trials=1000):
    para_filename = f"result/{dataset_name}/NeuroLKH_para/{instance_name}.para"
    log_filename = f"result/{dataset_name}/NeuroLKH_log/{instance_name}.log"
    log_run_filename = f"result/{dataset_name}/lhkhelper_log/{instance_name}.log"
    instance_filename = f"result/{dataset_name}/tsp/{instance_name}.tsp"
    if rerun or not os.path.isfile(log_filename):
        # write_instance(instance, instance_name, instance_filename)
        write_para(dataset_name, instance_name, instance_filename,
                   "NeuroLKH", para_filename, max_trials=max_trials)
        write_candidate_pi(dataset_name, instance_name, candidate, pi, beta)
        with open(log_filename, "w") as f:
            check_call(["./LKH", para_filename], stdout=f)
        #with open(log_run_filename) as f:
        #    f.write("solve_NeuroLKH: instance {instance},  ")

    return read_results(log_filename, max_trials)


def read_results(log_filename, max_trials):
    objs = []
    runtimes = []
    with open(log_filename, "r") as f:
        lines = f.readlines()
        for line in lines:  # read the obj and runtime for each trial
            if line[:6] == "-Trial":
                line = line.strip().split(" ")
                assert len(objs) + 1 == int(line[-3])
                objs.append(int(line[-2]))
                runtimes.append(float(line[-1]))
        final_obj = int(lines[-6].split(",")[0].split(" ")[-1])
        if len(objs) == 0:  # solved by subgradient optimization
            ascent_runtime = float(lines[66].split(" ")[-2])
            return [final_obj] * max_trials, [ascent_runtime] * max_trials
        else:
            assert objs[-1] == final_obj
            return objs, runtimes


def eval_dataset(dataset_filename, method, args, rerun=True, max_trials=1000):
    dataset_name = dataset_filename.strip(".pkl").split("/")[-1]
    os.makedirs(f"result/{dataset_name}/{method}_para", exist_ok=True)
    os.makedirs(f"result/{dataset_name}/{method}_log", exist_ok=True)
    os.makedirs(f"result/{dataset_name}/tsp", exist_ok=True)

    with open(dataset_filename, "rb") as f:
        dataset = pickle.load(f)[:args.n_samples]

    if method == "NeuroLKH":
        os.makedirs(f"result/{dataset_name}/featgen_para", exist_ok=True)
        os.makedirs(f"result/{dataset_name}/feat", exist_ok=True)
        with Pool(os.cpu_count()) as pool:
            feats = list(tqdm.tqdm(
                pool.imap(method_wrapper, [
                          ("FeatGen", dataset_name, dataset[i], str(i)) for i in range(len(dataset))]),
                total=len(dataset)))
        feats = list(zip(*feats))
        edge_index, edge_feat, inverse_edge_index, feat_runtime = feats
        feat_runtime = np.sum(feat_runtime)
        edge_index = np.concatenate(edge_index)
        edge_feat = np.concatenate(edge_feat)
        inverse_edge_index = np.concatenate(inverse_edge_index)
        net = SparseGCNModel()
        net.cuda()
        saved = torch.load(args.model_path)
        net.load_state_dict(saved["model"])
        sgn_start_time = time.time()
        with torch.no_grad():
            candidate_Pi, beta = infer_SGN(net, np.array(dataset), edge_index, edge_feat, inverse_edge_index,
                                     batch_size=args.batch_size)
        sgn_runtime = time.time() - sgn_start_time
        # with open("candidate_Pi/" + dataset_name + ".pkl", "rb") as f:
        #     candidate_Pi = pickle.load(f)
        n_node = len(dataset[0])
        candidate = candidate_Pi[:, :n_node * 5].reshape(-1, n_node, 5)
        pi = candidate_Pi[:, n_node * 5:].reshape(-1, n_node)

        os.makedirs(f"result/{dataset_name}/candidate", exist_ok=True)
        os.makedirs(f"result/{dataset_name}/pi", exist_ok=True)
        os.makedirs(f"result/{dataset_name}/beta", exist_ok=True)

        with Pool(os.cpu_count()) as pool:
            results = list(tqdm.tqdm(pool.imap(method_wrapper, [
                ("NeuroLKH", dataset_name, dataset[i], str(i), candidate[i], pi[i], beta[i], rerun, max_trials) for i in
                range(len(dataset))]), total=len(dataset)))
    else:
        assert method == "LKH"
        feat_runtime = 0
        sgn_runtime = 0
        with Pool(os.cpu_count()) as pool:
            results = list(tqdm.tqdm(pool.imap(method_wrapper,
                                               [("LKH", dataset_name, dataset[i], str(i), rerun, max_trials) for i in
                                                range(len(dataset))]), total=len(dataset)))
    results = np.array(results)
    dataset_objs = results[:, 0, :].mean(0)
    dataset_runtimes = results[:, 1, :].sum(0)
    return dataset_objs, dataset_runtimes, feat_runtime, sgn_runtime
