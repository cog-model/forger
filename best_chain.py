from hierarchical_tasks_extraction import extract_chain
from hierarchical_tasks_extraction.extract_chain import TrajectoryInformation

import json
import os

import tensorflow as tf
import yaml

import pipeline
from utils.tf_util import config_gpu

def find_best():
    a = extract_chain.generate_final_chain(data_dir='demonstrations/')
    print(a)
    #chain = TrajectoryInformation('demonstrations/MineRLObtainIronPickaxe-v0/v3_rigid_mustard_greens_monster-11_878-4825')
    #final_chain = chain.to_old_chain_format(items=chain.chain, return_time_indexes=False)
    #print(final_chain)
    return


if __name__ == '__main__':
    find_best()