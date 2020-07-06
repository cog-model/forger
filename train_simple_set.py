import gym
from chainerrl.wrappers.atari_wrappers import EpisodicLifeEnv, FireResetEnv, \
    ClipRewardEnv, WarpFrame, \
    MaxAndSkipEnv

from ForgER.agent import *
import tensorflow as tf

from ForgER.replay_buff import AggregatedBuff
from utils.env_wrappers import FrameSkip, FrameStack, get_discretizer
import yaml
import argparse
from utils.data_loaders import NPZLoader
from utils.tf_util import config_gpu
from ForgER.model import get_network_builder
from utils.discretization import get_dtype_dict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config_gpu()


def make_deepmind(env_id, frame_skip=4, frame_stack=4):
    """Configure environment for DeepMind-style Atari.
    """
    env = gym.make(env_id)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, frame_stack)
    return env


def make_env(env_id, frame_skip=4, frame_stack=2, atari=False, seed=None, discretizer=None):
    if atari:
        return make_deepmind(env_id, frame_skip, frame_stack)
    environment = gym.make(env_id)
    if frame_skip > 1:
        environment = FrameSkip(environment, frame_skip)
    if frame_stack > 1:
        environment = FrameStack(environment, frame_stack)
    if discretizer:
        environment = get_discretizer(discretizer)(environment)
    if seed is not None:
        environment.seed(seed)
    return environment


def run_pipeline(file_name):
    with open(file_name, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    env = make_env(**config['env'])
    env_dict, dtype_dict = get_dtype_dict(env)
    replay_buffer = AggregatedBuff(env_dict=env_dict, **config['buffer'])
    make_model = get_network_builder(config['model_name'])
    agent = Agent(config['agent'], replay_buffer, make_model, env.observation_space, env.action_space, dtype_dict)
    if 'loader' in config:
        data_ = NPZLoader(**config['loader'])
        agent.add_demo(data_,)
    summary_writer = tf.summary.create_file_writer(config['tb_dir'])
    with summary_writer.as_default():
        agent.pre_train(**config['pretrain'])
        scores_, _ = agent.train(env, **config['train'])
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ForgER runner')
    parser.add_argument('--config', type=str, action="store", help='yaml file with params',
                        required=True)
    params = parser.parse_args()
    with tf.device('/gpu'):
        run_pipeline(params.config)
