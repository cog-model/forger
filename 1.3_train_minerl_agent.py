import argparse
import json
import os

import tensorflow as tf
import yaml

import pipeline
from utils.tf_util import config_gpu
from hierarchical_tasks_extraction.extract_chain import TrajectoryInformation
from hierarchical_task_policy.ItemAgent import ItemAgent
import wandb


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config_gpu()


def run_train(file_name):
    print('==========================step1: prepare wandb and config==========================')

    wandb.init(anonymous='allow', project="MineRL_pickaxe", group='PC_1.3_26_Oct')

    with open(file_name, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    if not os.path.isdir('train'):
        os.mkdir('train')

    print('==========================step2: load chain==========================')

    with open(config['chain_path'], "r") as f:
        chain = json.load(f)

    print('\nwhat is chain:',chain)

    print('==========================step3: init an agent with chain==========================')
    item_agent = ItemAgent(chain)

    agent_config = config['agent']
    buffer_config = config['buffer']
    wrapper_config = config['wrappers']
    agent_config['wandb'] = wandb
    train_config = config['train_agent']

    print('==========================step4: load pretrain agent==========================')
    item_agent.load_agents(agent_config, buffer_config, wrapper_config,
                             train_config['env_name'], train_config['pretrain'])

    if 'train' in train_config:
        item_agent.train(agent_config, buffer_config, wrapper_config, train_config['env_name'], **train_config['train'])




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train entry runner')
    parser.add_argument('--config', type=str, action="store", help='yaml file with params',
                        required=True)
    params = parser.parse_args()
    with tf.device('/gpu'):
        run_train(params.config)
