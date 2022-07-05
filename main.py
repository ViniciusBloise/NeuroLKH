from intf.readerTSP import ReaderTSP
from intf.config import Config
import networkx as nx
import os

config = Config(tspInstUsed=os.listdir(r'./result/1000/tsp'))
reader = ReaderTSP(config)

print(config.TSP_INST_USED)

iterator = iter(reader.instances_generator())
instance = next(iterator)
_points, positions, distance_matrix, name, optimal_tour = instance
print(positions)