import gym
from chainerrl.wrappers.atari_wrappers import EpisodicLifeEnv, FireResetEnv, \
    ClipRewardEnv, WarpFrame, \
    MaxAndSkipEnv, NoopResetEnv
from ForgER.agent import *
import tensorflow as tf

from utils.env_wrappers import FrameSkip, FrameStack, SaveDS
import yaml
import argparse
from utils.tf_util import config_gpu
from ForgER.model import get_network_builder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config_gpu()


def make_deepmind(env_id, frame_skip=4, frame_stack=4):
    """Configure environment for DeepMind-style Atari.
    """
    env = gym.make(env_id)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, frame_stack)
    return env


def make_env(env_id, frame_skip=4, frame_stack=2, atari=False):
    if atari:
        return make_deepmind(env_id, frame_skip, frame_stack)
    environment = gym.make(env_id)
    if frame_skip > 1:
        environment = FrameSkip(environment, frame_skip)
    if frame_stack > 1:
        environment = FrameStack(environment, frame_stack)
    return environment


def save_ds(file_name):
    with open(file_name, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    if 'num_env' in config['env']:
        config['env'].pop('num_env')
    if 'seed' in config['env']:
        config['env'].pop('seed')
    env = make_env(**config['env'])
    env = SaveDS(env)
    make_model = get_network_builder(config['model_name'])
    agent = Agent(config['agent'], None, make_model, env.observation_space, env.action_space)
    agent.test(env, **config['test'])
    env.save_ds('train/{}'.format(config['env']['env_id']))
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ds_saver')
    parser.add_argument('-file_name', type=str, action="store", help='yaml file with params',
                        required=True)
    params = parser.parse_args().__dict__
    with tf.device('/gpu'):
        save_ds(**params)