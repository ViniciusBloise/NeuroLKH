from intf.readerTSP import ReaderTSP
from intf.plotterTSP import PlotterTSP
from intf.config import Config
from intf.graph_builder import GraphBuilder
from intf.lkh_helper import solve_LKH, write_instance
import intf.constant as const
import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np


dataset_name = '1000'
config = Config(tspInstUsed=os.listdir(f'./result/{dataset_name}/tsp'))
reader = ReaderTSP(config)

iterator = iter(reader.instances_generator())
instance = next(iterator)
_points, positions, distance_matrix, name, optimal_tour = instance
print(positions)

#Call the plotter
plotter = PlotterTSP(None)
G = GraphBuilder(pos=positions, dist_matrix=distance_matrix)
G.add_minimum_spanning_tree()
#plt.show()

solve_LKH(dataset_name=dataset_name, instance=positions, instance_name=name, rerun=True)
print(config.get_dir(const.RES_DIR_BESTRUN, name, '.log'))
input('Hit <ENTER> to end.')