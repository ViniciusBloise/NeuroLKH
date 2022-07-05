from . import config as cfg
import numpy as np
from os.path import join, exists


class ReaderTSP:

    def __init__(self, config=cfg.Config()):
        self.path = config.TSP_INST_URL
        self.instances = config.TSP_INST_USED

        self.distance_formula_dict = {
            'EUC_2D': self.distance_euc,
            'ATT': self.distance_att,
            'GEO': self.distance_geo
        }

    def instances_generator(self):
        for file in self.instances:
            yield self.read_instance(join(f'{self.path}tsp/', file))

    def read_instance(self, filename):
        # read raw data
        with open(filename) as file_object:
            data = file_object.read()
        lines = data.splitlines()

        # get current instance information
        name = lines[0].split(':')[1].strip()
        n_points = np.int(lines[3].split(':')[1].strip())
        distance = lines[4].split(':')[1].strip()
        distance_formula = self.distance_formula_dict[distance]

        # read all data points for the current instance
        positions = np.zeros((n_points, 2))
        for i in range(n_points):
            line_i = lines[6 + i].strip().split(' ')
            positions[i, 0] = float(line_i[1])
            positions[i, 1] = float(line_i[2])

        distance_matrix = ReaderTSP.create_dist_matrix(
            n_points, positions, distance_formula)
        #optimal_tour = self.get_optimal_solution(name, positions)
        optimal_tour = None

        return n_points, positions, distance_matrix, name, optimal_tour

    def get_optimal_solution(self, name, positions):
        filename = f'{self.path}optimal/{name}.npy'
        if exists(filename):
            optimal_tour = ReaderTSP.load_optimal_solution(filename)
        else:
            optimal_tour = ReaderTSP.compute_optimal_solution(positions)
        return optimal_tour

    @staticmethod
    def distance_euc(zi, zj):
        delta_x = zi[0] - zj[0]
        delta_y = zi[1] - zj[1]
        return round(np.sqrt(delta_x ** 2 + delta_y ** 2), 0)

    @staticmethod
    def distance_att(zi, zj):
        delta_x = zi[0] - zj[0]
        delta_y = zi[1] - zj[1]
        rij = np.sqrt((delta_x ** 2 + delta_y ** 2) / 10.0)
        tij = float(rij)
        if tij < rij:
            dij = tij + 1
        else:
            dij = tij
        return dij

    @staticmethod
    def distance_geo(zi, zj):
        RRR = 6378.388
        lat_i, lon_i = ReaderTSP.get_lat_long(zi)
        lat_j, lon_j = ReaderTSP.get_lat_long(zj)
        q1 = np.cos(lon_i - lon_j)
        q2 = np.cos(lat_i - lat_j)
        q3 = np.cos(lat_i + lat_j)
        return float(RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)

    @staticmethod
    def create_dist_matrix(nPoints, positions, distance_formula):
        distance_matrix = np.zeros((nPoints, nPoints))
        for i in range(nPoints):
            for j in range(i, nPoints):
                distance_matrix[i, j] = distance_formula(positions[i], positions[j])
        distance_matrix += distance_matrix.T
        return distance_matrix

    @staticmethod
    def load_optimal_solution(filename):
        return np.load(filename)
