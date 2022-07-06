from intf.readerTSP import ReaderTSP
from intf.plotterTSP import PlotterTSP
from intf.config import Config
from intf.graph_builder import GraphBuilder
from intf.lkh_helper import solve_NeuroLKH
import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np


config = Config(tspInstUsed=os.listdir(r'./result/1000/tsp'))
reader = ReaderTSP(config)

print(config.TSP_INST_USED)

iterator = iter(reader.instances_generator())
instance = next(iterator)
_points, positions, distance_matrix, name, optimal_tour = instance
print(positions)

#Call the plotter
plotter = PlotterTSP(None)
G = GraphBuilder(pos=positions, dist_matrix=distance_matrix)
G.add_minimum_spanning_tree()
#plt.show()

solve_NeuroLKH(dataset_name='1000', instance=None, instance_name=name, candidate=np.empty(0), pi=np.empty(0), rerun=True)

input('Hit <ENTER> to end.')