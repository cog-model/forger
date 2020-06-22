import json

import yaml
import tensorflow as tf
from hierarchical_task_policy.ItemAgent import ItemAgent
import wandb
import argparse
from pipeline import run_pipeline


def run_train(file_name):
    wandb.init(sync_tensorboard=True, anonymous='allow', project="FSRB", group='hierarchical_tasks_extraction aug')

    with open('train/chain.json', "r") as f:
        chain = json.load(f)
    item_agent = ItemAgent(chain)
    with open(file_name, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    global_config = config['global']
    global_config['wandb'] = wandb
    train_config = config['train_agent']
    item_agent.pre_train(global_config, train_config['pretrain'])
    item_agent.train(global_config, train_config['train'], train_config['train']['env_name'],
                     episodes=train_config['train']['episodes'],
                     agents_to_train=tuple(train_config['train']['items']['list']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hie augmentation experiment runner')
    parser.add_argument('-file_name', type=str, action="store", help='yaml file with params',
                        required=True)
    parser.add_argument('-run_pipeline', action="store_true", help='either run pipeline before experiment or not')
    params = parser.parse_args().__dict__

    with tf.device('/gpu'):
        if params['run_pipeline']:
            run_pipeline(params['file_name'])
        run_train(params['file_name'])
