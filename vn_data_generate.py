import numpy as np
import pickle

n_samples = 1000
list_nodes = [30, 50, 100, 200]

file_dir = "test"
for n_nodes in list_nodes:
    x = np.random.uniform(size=[n_samples, n_nodes, 2])
    l_x = x.tolist()
    #print(x.shape, len(l_x), len(l_x[0]), len(l_x[0][0]))

    filename = f"{file_dir}/vn_{n_nodes}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(l_x, f)

print("End of data generating")

