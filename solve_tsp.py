from concorde.tsp import TSPSolver
from concorde.tests.data_utils import get_dataset_path

def solve_concorde(x):
    f = 10000000
    result = []

    solver = TSPSolver.from_data(x[:, 0] * f, x[:, 1] * f, norm='EUC_2D')
    solution = solver.solve()
    q = solution.tour
    q = [int(p) for p in q]
    result.append(q)
    return result