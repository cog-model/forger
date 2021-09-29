from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import os
from builtins import *
from future import standard_library
from utils.env_wrappers import FrameSkip, FrameStack, ObtainPoVWrapper
from ForgER.replay_buff import AggregatedBuff

standard_library.install_aliases()

from ForgER.agent import Agent


class ScaleGradHook(object):
    name = 'ScaleGrad'
    call_for_each_param = True
    timing = 'pre'

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, rule, param):
        if getattr(param, 'scale_param', False):
            param.grad *= self.scale


class NoOpAgent:
    def __init__(self):
        pass

    def run(self, env):
        done, score, state = False, 0, env.reset()
        step = 0
        while done is False:
            action = {}
            next_state, reward, done, _ = env.step(action)
            score += reward
            step += 1
        return score

    def train(self, env):
        return self.run(env)

    def save_agent(self):
        pass

    def load_agent(self):
        pass


class RfDAgent:
    def __init__(self, item_dir, agent_config, buffer_config, env_dict,
                 epsilon=0.1, eps_decay=0.99, final_epsilon=0.01, **kwargs):
        self.item_dir = item_dir
        if buffer_config:
            replay_buff = AggregatedBuff(env_dict=env_dict, **buffer_config)
        else:
            replay_buff = AggregatedBuff(32, env_dict=env_dict)
        self.replay_buff = replay_buff
        self.agent = Agent(agent_config, replay_buff, **kwargs)
        self.counter = 0
        self.epsilon = epsilon
        self.epsilon_decay = eps_decay
        self.final_epsilon = final_epsilon

    def make_env(self, env, test=None):
        env = ObtainPoVWrapper(env)
        env = FrameSkip(env)
        env = FrameStack(env, 2)
        return env

    def run(self, env):
        return self.agent.test(env, None)

    def train(self, env):
        reward, self.counter = self.agent.train_episode(env, current_step=self.counter,
                                                        epsilon=self.epsilon)
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
        return reward, self.counter

    def save_agent(self):
        self.agent.save(self.item_dir + '/model.ckpt')

    def load_agent(self):
        print('loading agent now...')
        if os.path.exists(self.item_dir):
            self.agent.load(self.item_dir + '/model.ckpt')

        else:
            print('WARNING: No weights in {}'.format(self.item_dir))
