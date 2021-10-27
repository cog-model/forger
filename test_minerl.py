import argparse
import json
import os

import wandb
import yaml
import tensorflow as tf

from utils.tf_util import config_gpu
from hierarchical_task_policy.ItemAgent import ItemAgent


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config_gpu()


def run_test(file_name):
    wandb.init(anonymous='allow', project="test_minerl", group='22_10_100_episodes')


    with open(file_name, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    agent_config = config['agent']
    wrapper_config = config['wrappers']
    test_config = config['test_agent']
    agent_config["wandb"] = wandb

    with open(config['chain_path'], "r") as f:
        chain = json.load(f)

    item_agent = ItemAgent(chain)

    item_agent.run(agent_config, wrapper_config, "MineRLObtainDiamondDense-v0", **test_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test entry runner')
    parser.add_argument('--config', type=str, action="store", help='yaml file with params',
                        required=True)
    params = parser.parse_args()
    with tf.device('/gpu'):
        run_test(params.config)
