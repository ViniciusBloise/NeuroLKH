import yaml
from . import constant as const
import os


class Config:

    def __init__(self, tspInstUsed=None):
        with open(const.CONFIG_YAML) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        self.config = config

        self.TSP_INST_URL = config[const.TSP_INST_URL]
        self.TSP_INST_USED = tspInstUsed
        #self.TSP_INST_USED = config[const.TSP_INST_USED]

    def get_dir(self, dirtype, filename=None, ext='.txt') -> str:
        if filename == None:
            return self.TSP_INST_URL + dirtype + f'/'
        else:
            return self.TSP_INST_URL + dirtype + f'/{os.path.basename(filename).split(".")[0]}{ext}'