from ast import Str
from intf import config, constant as const
from tqdm import tqdm
import os
import sh


def generate_bestrun_files(cfg = None):
    print('#### Generate best run files from tsp ###')
    print('####')

    if cfg is None:
        cfg = config.Config()

    instance = '100'    
    dirname = cfg.get_dir(const.RES_DIR_BESTRUN)
    tspdir = os.listdir(cfg.get_dir(const.RES_DIR_LKH_LOG))

    for log_filename in tqdm(tspdir):
        full_logfilename = cfg.get_dir(const.RES_DIR_LKH_LOG, log_filename, '.log')

        bestrun = sh.tail('-n 1', full_logfilename)        
        bestrun_file = cfg.get_dir(const.RES_DIR_BESTRUN, log_filename, '.txt')
        #print(bestrun_file)
        with open(bestrun_file, 'w') as f:
            br = str(bestrun.wait())[:-1].strip()
            f.write(br)

    print('# end #')

def read_candidate(cfg: config.Config  = None):
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
        #print(full_candfilename)
        with open(full_candfilename, 'r') as f:
            cand = f.read()
            #print(cand)
            parts = proc_candidate_file(cand)
        problem[problem_name] = parts
    return problem

def proc_candidate_file(filetext: str):
    #candidate files are ranging from 1 -> nodes
    #return a dict of nodes and candidates edge according to a priority

    parts: dict[str, list[int]] = {}
    lines = filetext.split('\n')
    n_nodes = int(lines[0])
    for line in lines[1:n_nodes+1]: #tqdm
        i = 0
        for part in line.split(' '):
            if i == 0:
                node = str(int(part) - 1)
                parts[node]=[]
            elif i > 1 and i % 2 == 1: 
                parts[node].append( int(part) - 1)
            i += 1
    return parts

def get_edges_from_candidate(candidates, pos=0):
    edges = []
    for key in tqdm(candidates):
        a = int(key)
        b = candidates[key][pos]
        tuple = (a,b) if(a<b) else (b,a)
        edges.append( tuple)
    return edges

def problem_set(cfg: config.Config = None):
    print('#### Reading TSP candidate sets')
    print('####')

    tspdir = os.listdir(cfg.get_dir(const.RES_DIR_TSP))

    problem: dict[str, list[tuple(float, float)]] = {}
    for filename in tqdm(tspdir):
        problem_name = filename.split('.')[0]
        full_problemname = cfg.get_dir(const.RES_DIR_TSP, problem_name, '.tsp')
        points: list[tuple(float, float)] = []
        #print(full_candfilename)
        
        print(full_problemname)
        with open(full_problemname, 'r') as f:
            probls = f.readlines()
            start_idx = [idx for idx, x in enumerate(probls) if 'NODE_COORD_SECTION' in x][0]
            end_idx = [idx for idx, x in enumerate(probls) if 'EOF' in x][0]
            print(probls[start_idx])
            for line in probls[start_idx+1:end_idx]:
       
                items = line.strip().split(' ')[1:]
                point = (float(items[0]), float(items[1]))
                points.append(point)

            #print(cand)
        problem[problem_name] = points
        break
    return problem


if __name__ == '__main__':
    #generate_bestrun_files()
    cfg = config.Config()

    candidates = read_candidate(cfg=cfg)
    #edges = get_edges_from_candidate(candidates)
    #print(edges)
    print(len(candidates))

    tspProblemSet = problem_set(cfg=cfg)
    print(tspProblemSet)
