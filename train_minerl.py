import argparse
import json
import os

import tensorflow as tf
import yaml

import pipeline
from utils.tf_util import config_gpu
from hierarchical_tasks_extraction.extract_chain import TrajectoryInformation #
from hierarchical_task_policy.ItemAgent import ItemAgent
import wandb

#from Ori_hierachical_tasks_extraction.extract_chain import TrajectoryInformation



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config_gpu()


def run_train(file_name):
    print('==========================step1: prepare wandb and config==========================')

    wandb.init(anonymous='allow', project="ForgER", group='train_entry')

    chain = TrajectoryInformation('demonstrations/MineRLObtainIronPickaxe-v0/v3_rigid_mustard_greens_monster-11_878-4825')
    final_chain = chain.to_old_chain_format(items=chain.chain, return_time_indexes=False)

    print('finished final_chain')

    with open(file_name, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    if not os.path.isdir('train'):
        os.mkdir('train')
    with open('train/chain.json', "w") as f:
        json.dump(final_chain, f)

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

    print('==========================step4: pretrain agent==========================')
    if 'pretrain' in train_config:
        item_agent.pre_train(agent_config, buffer_config, wrapper_config,
                             train_config['env_name'], train_config['pretrain'])
        print('pre-train finished.......')
    if 'log_agent_pipeline' in config:
        pipeline.run_pipeline(file_name)
    if 'train' in train_config:
        item_agent.train(agent_config, buffer_config, wrapper_config, train_config['env_name'], **train_config['train'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train entry runner')
    parser.add_argument('--config', type=str, action="store", help='yaml file with params',
                        required=True)
    params = parser.parse_args()
    with tf.device('/gpu'):
        run_train(params.config)
