import os
from collections import deque

import cv2
import numpy as np
import gym

from utils.config_validation import WrappersCfg

mapping = dict()


class LazyFrames:

    def __init__(self, frames, stack_axis=2):
        self.stack_axis = stack_axis
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=self.stack_axis)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class DiscreteBase(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_dict = {}
        self.action_space = gym.spaces.Discrete(len(self.action_dict))

    def step(self, action):
        s, r, done, info = self.env.step(self.action_dict[action])
        return s, r, done, info

    def sample_action(self):
        return self.action_space.sample()


class DiscreteWrapper(DiscreteBase):
    def __init__(self, env, always_attack=True, angle=5):
        super().__init__(env)
        self.action_dict = {
            0: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 1, 'jump': 0, 'left': 0, 'right': 0,
                'sneak': 0, 'sprint': 0},
            1: {'attack': always_attack, 'back': 0, 'camera': [0, angle], 'forward': 0, 'jump': 0, 'left': 0,
                'right': 0, 'sneak': 0, 'sprint': 0},
            2: {'attack': 1, 'back': 0, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0,
                'sprint': 0},
            3: {'attack': always_attack, 'back': 0, 'camera': [angle, 0], 'forward': 0, 'jump': 0, 'left': 0,
                'right': 0, 'sneak': 0, 'sprint': 0},
            4: {'attack': always_attack, 'back': 0, 'camera': [-angle, 0], 'forward': 0, 'jump': 0, 'left': 0,
                'right': 0, 'sneak': 0, 'sprint': 0},
            5: {'attack': always_attack, 'back': 0, 'camera': [0, -angle], 'forward': 0, 'jump': 0, 'left': 0,
                'right': 0, 'sneak': 0, 'sprint': 0},
            6: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 1, 'jump': 1, 'left': 0, 'right': 0,
                'sneak': 0, 'sprint': 0},
            7: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 1, 'right': 0,
                'sneak': 0, 'sprint': 0},
            8: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 1,
                'sneak': 0, 'sprint': 0},
            9: {'attack': always_attack, 'back': 1, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 0,
                'sneak': 0, 'sprint': 0}}
        self.action_space = gym.spaces.Discrete(len(self.action_dict))


class ActionNoiseWrapper(gym.Wrapper):

    def step(self, action: dict):
        return self.env.step(self.apply_noise(action))

    @staticmethod
    def apply_noise(action):
        if 'camera' in action:
            x, y = list(np.random.normal(0, 0.02, 2))
            action['camera'] = np.add([x, y], action['camera'])
        return action


def wrap_env(env, settings: WrappersCfg):
    env = ActionNoiseWrapper(env)
    env = ObtainPoVWrapper(env)
    env = FrameSkip(env, settings.frame_skip)
    env = FrameStack(env, settings.frame_stack)
    env = DiscreteWrapper(env)
    return env


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)

        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        infos = []
        info = {}
        obs = None
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            infos.append(info)
            total_reward += reward
            if done:
                break
        if 'expert_action' in infos[0]:
            info['expert_action'] = self.env.preprocess_action([info_['expert_action'] for info_ in infos])
        return obs, total_reward, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, channel_order='hwc', use_tuple=False):

        gym.Wrapper.__init__(self, env)
        self.k = k
        self.observations = deque([], maxlen=k)
        self.stack_axis = {'hwc': 2, 'chw': 0}[channel_order]
        self.use_tuple = use_tuple

        if self.use_tuple:
            pov_space = env.observation_space[0]
            inv_space = env.observation_space[1]
        else:
            inv_space = None
            pov_space = env.observation_space

        low_pov = np.repeat(pov_space.low, k, axis=self.stack_axis)
        high_pov = np.repeat(pov_space.high, k, axis=self.stack_axis)
        pov_space = gym.spaces.Box(low=low_pov, high=high_pov, dtype=pov_space.dtype)

        if self.use_tuple:
            low_inv = np.repeat(inv_space.low, k, axis=0)
            high_inv = np.repeat(inv_space.high, k, axis=0)
            inv_space = gym.spaces.Box(low=low_inv, high=high_inv, dtype=inv_space.dtype)
            self.observation_space = gym.spaces.Tuple(
                (pov_space, inv_space))
        else:
            self.observation_space = pov_space

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.observations.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.observations.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.observations) == self.k
        if self.use_tuple:
            frames = [x[0] for x in self.observations]
            inventory = [x[1] for x in self.observations]
            return (LazyFrames(list(frames), stack_axis=self.stack_axis),
                    LazyFrames(list(inventory), stack_axis=0))
        else:
            return LazyFrames(list(self.observations), stack_axis=self.stack_axis)


class ObtainPoVWrapper(gym.ObservationWrapper):
    """Obtain 'pov' value (current game display) of the original observation."""

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = self.env.observation_space.spaces['pov']

    def observation(self, observation):
        return observation['pov']


class DiscreteBase(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_dict = {}
        self.action_space = gym.spaces.Discrete(len(self.action_dict))

    def step(self, action):
        s, r, done, info = self.env.step(self.action_dict[action])
        return s, r, done, info

    def sample_action(self):
        return self.action_space.sample()


class SaveVideoWrapper(gym.Wrapper):
    current_episode = 0

    def __init__(self, env, path='train/', resize=4, reward_threshold=0):
        """
        :param env: wrapped environment
        :param path: path to save videos
        :param resize: resize factor
        """
        super().__init__(env)
        self.path = path
        self.recording = []
        self.rewards = [0]
        self.resize = resize
        self.reward_threshold = reward_threshold
        self.previous_reward = 0

    def step(self, action):
        """
        make a step in environment
        :param action: agent's action
        :return: observation, reward, done, info
        """
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        self.recording.append(self.bgr_to_rgb(observation['pov']))
        return observation, reward, done, info

    def get_reward(self):
        return sum(map(int, self.rewards))

    def reset(self, **kwargs):
        """
        reset environment and save game video if its not empty
        :param kwargs:
        :return: current observation
        """

        reward = self.get_reward()
        if self.current_episode > 0 and reward >= self.reward_threshold:
            name = str(self.current_episode).zfill(4) + "r" + str(reward).zfill(4) + ".mp4"
            full_path = os.path.join(self.path, name)
            upscaled_video = [self.upscale_image(image, self.resize) for image in self.recording]
            self.save_video(full_path, video=upscaled_video)
        self.current_episode += 1
        self.rewards = [0]
        self.recording = []
        self.env.seed(self.current_episode)
        observation = self.env.reset(**kwargs)
        self.recording.append(self.bgr_to_rgb(observation['pov']))
        return observation

    @staticmethod
    def upscale_image(image, resize):
        """
        increase image size (for better video quality)
        :param image: original image
        :param resize:
        :return:
        """
        size_x, size_y, size_z = image.shape
        return cv2.resize(image, dsize=(size_x * resize, size_y * resize))

    @staticmethod
    def save_video(filename, video):
        """
        saves video from list of np.array images
        :param filename: filename or path to file
        :param video: [image, ..., image]
        :return:
        """
        size_x, size_y, size_z = video[0].shape
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 60.0, (size_y, size_x))
        for image in video:
            out.write(image)
        out.release()
        cv2.destroyAllWindows()

    @staticmethod
    def bgr_to_rgb(image):
        """
        converts BGR image to RGB
        :param image: bgr image
        :return: rgb image
        """
        return image[..., ::-1]
