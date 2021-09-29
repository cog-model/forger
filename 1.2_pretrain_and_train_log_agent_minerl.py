import argparse
import json
import os

import tensorflow as tf
import yaml

#import pipeline
import pipeline
from utils.tf_util import config_gpu
from hierarchical_tasks_extraction.extract_chain import TrajectoryInformation #
from hierarchical_task_policy.ItemAgent import ItemAgent
import wandb


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config_gpu()


def run_train(file_name):
    print('==========================step1: prepare wandb and config==========================')

    wandb.init(anonymous='allow', project="1.2_log_agent", group='pretrain_and_train_log_agent')

    with open(file_name, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    if not os.path.isdir('train'):
        os.mkdir('train')

    print('==========================step2: load chain==========================')

    with open(config['chain_path'], "r") as f:
        chain = json.load(f)

    print('\nwhat is chain:',chain)


    print('==========================step3: pretrain and train log_agent though pipeline==========================')

    if 'log_agent_pipeline' in config:
        pipeline.run_pipeline(file_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train entry runner')
    parser.add_argument('--config', type=str, action="store", help='yaml file with params',
                        required=True)
    params = parser.parse_args()
    with tf.device('/gpu'):
        run_train(params.config)
