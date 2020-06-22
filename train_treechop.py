import minerl
import gym
from ForgER.agent import *
import tensorflow as tf

from ForgER.replay_buff import AggregatedBuff
from pipeline import wrap_env
import yaml
import argparse
import wandb
from utils.data_loaders import TreechopLoader
from utils.tf_util import config_gpu
from utils.discretization import get_dtype_dict
from ForgER.model import get_network_builder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config_gpu()


def run_treechop(config):
    wandb.init(config=tf.compat.v1.flags.FLAGS, sync_tensorboard=True, anonymous='allow', project="tf2refact",
               group='pipeline')

    with open(config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    agent_config = config['agent']
    # pipeline_config = config['log_agent_pipeline']
    wrappers_config = config['wrappers']
    wandb.config.update(agent_config)
    # wandb.config.update(pipeline_config)
    agent_config['wandb'] = wandb

    data_ = minerl.data.make("MineRLTreechop-v0", data_dir='demonstrations')
    data_ = TreechopLoader(data_, threshold=config['pretrain']['min_demo_reward'],
                           **wrappers_config)
    make_model = get_network_builder('minerl_dqfd')
    env = wrap_env(gym.make("MineRLTreechop-v0"), **wrappers_config)
    env_dict, dtype_dict = get_dtype_dict(env)

    replay_buffer = AggregatedBuff(env_dict=env_dict, **config['buffer'])
    agent = Agent(agent_config, replay_buffer, make_model, env.observation_space,
                  env.action_space, dtype_dict)
    agent.add_demo(data_, )
    agent.pre_train(config['pretrain']['steps'])
    summary_writer = tf.summary.create_file_writer('train/')
    with summary_writer.as_default():
        scores_, _ = agent.train(env, name="model.ckpt", episodes=config['episodes'])
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train treechop')
    parser.add_argument('--config', type=str, action="store", help='yaml file with params',
                        required=True)
    params = parser.parse_args()
    with tf.device('/gpu'):
        run_treechop(params.config)
