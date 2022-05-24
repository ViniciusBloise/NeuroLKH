import numpy as np
import torch
from torch.autograd import Variable

from .SRC_swig.LKH import featureGenerate
from .SRC_swig.LKH import lkh_main as LKH


def method_wrapper(args):
    if args[0] == "LKH":
        return solve_LKH(*args[1:])
    elif args[0] == "NeuroLKH":
        return solve_NeuroLKH(*args[1:])
    elif args[0] == "FeatGen":
        return generate_feat(*args[1:])


def solve_LKH(data, n_nodes, max_trials=1000):
    invec = data.copy()
    seed = 1234
    result = LKH(0, max_trials, seed, n_nodes, invec)
    return invec


def generate_feat(data, n_nodes):
    n_edges = 20
    data = np.array(data)
    invec = np.concatenate([data.reshape(-1) * 1000000, np.zeros([n_nodes * (3 * n_edges - 2)])], -1)
    feat_runtime = featureGenerate(1234, invec)
    edge_index = invec[:n_nodes * n_edges].reshape(1, -1, 20)
    edge_feat = invec[n_nodes * n_edges:n_nodes * n_edges * 2].reshape(1, -1, 20)
    inverse_edge_index = invec[n_nodes * n_edges * 2:n_nodes * n_edges * 3].reshape(1, -1, 20)
    return edge_index, edge_feat / 100000000, inverse_edge_index, feat_runtime / 1000000


def infer_SGN(net, dataset_node_feat, dataset_edge_index, dataset_edge_feat, dataset_inverse_edge_index, batch_size=100):
    candidate = []
    pi = []
    for i in range(dataset_edge_index.shape[0] // batch_size):
        node_feat = dataset_node_feat[i * batch_size:(i + 1) * batch_size]
        edge_index = dataset_edge_index[i * batch_size:(i + 1) * batch_size]
        edge_feat = dataset_edge_feat[i * batch_size:(i + 1) * batch_size]
        inverse_edge_index = dataset_inverse_edge_index[i * batch_size:(i + 1) * batch_size]
        node_feat = Variable(torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False)
        edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1, 1)
        edge_index = Variable(torch.FloatTensor(edge_index).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1)
        inverse_edge_index = Variable(torch.FloatTensor(inverse_edge_index).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1)
        y_edges, _, y_nodes = net.forward(node_feat, edge_feat, edge_index, inverse_edge_index, None, None, 20)
        pi.append(y_nodes.cpu().numpy())
        y_edges = y_edges.detach().cpu().numpy()
        y_edges = y_edges[:, :, 1].reshape(batch_size, dataset_node_feat.shape[1], 20)
        y_edges = np.argsort(-y_edges, -1)
        edge_index = edge_index.cpu().numpy().reshape(-1, y_edges.shape[1], 20)
        candidate_index = edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(y_edges.shape[1]).reshape(1, -1, 1), y_edges]
        candidate.append(candidate_index[:, :, :5])
    candidate = np.concatenate(candidate, 0)
    pi = np.concatenate(pi, 0)
    candidate_Pi = np.concatenate([candidate.reshape(dataset_edge_index.shape[0], -1), 1000000 * pi.reshape(dataset_edge_index.shape[0], -1)], -1)
    return candidate_Pi


def solve_NeuroLKH(data, n_nodes, max_trials=1000):
    invec = data.copy()
    seed = 1234
    result = LKH(1, max_trials, seed, n_nodes, invec)
    return invec
