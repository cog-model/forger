import gym
from gym.spaces import Dict, Box

from utils.discretization import ExpertActionPreprocessing


class FakeEnv(gym.Env):

    def __init__(self, data):
        self.index = 0
        self.states, self.actions, self.rewards, self.next_states, self.dones = data
        self.viewer = None
        self.action_preprocessor = ExpertActionPreprocessing()
        self.observation_space = Dict(pov=Box(0, 256, (64, 64, 3)))
        super().__init__()

    def preprocess_action(self, action):
        action = self.action_preprocessor.to_joint_action(action)
        action = self.action_preprocessor.to_compressed_action(action)
        action = self.action_preprocessor.to_discrete_action(action)
        return action

    def step(self, action):
        if self.are_all_frames_used():
            raise KeyError("No available data for sampling")
        # action = self.preprocess_action(self.actions[self.index])

        info = {'expert_action': self.actions[self.index]}
        result = self.next_states[self.index], self.rewards[self.index], self.dones[self.index], info
        self.index += 1
        return result

    def reset(self):
        if self.are_all_frames_used():
            raise KeyError("No available data for sampling")
        return self.states[self.index]

    def _get_image(self):
        return self.next_states[min(len(self.states) - 1, self.index)]['pov']

    def are_all_frames_used(self):
        return self.index >= len(self.states)

    def render(self, mode="human"):
        img = self._get_image()
        if mode == "rgb_array":
            return img
        elif mode == "human":
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
