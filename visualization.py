import cv2
import sys
from ForgER.model import get_network_builder
import os
import gym
from ForgER.agent import Agent
from collections import defaultdict
import numpy as np
from utils.tf_util import saliency_map
import tensorflow as tf


def saliency_img(weights_path, obs_shape, action_number, image_list, rows, columns, name=None):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = defaultdict(lambda: defaultdict(lambda: None))

    make_model = get_network_builder("minerl_dqfd")
    obs_space = np.ones(obs_shape)
    ac_space = gym.spaces.discrete.Discrete(action_number)
    agent = Agent(config['agent'], None, make_model, obs_space, ac_space, None)
    if weights_path is not None:
        agent.load(weights_path)

    def normalize_and_add(arr, obs):
        rsl = obs[:, :, -3:]
        rsl[:, :, -1] = np.clip(obs[:,:,-1] + arr, 0, 255)
        return rsl

    stack = []
    for x in image_list:
        if x.shape[:2] != obs_space.shape[:2]:
            resized = cv2.resize(x, dsize=obs_space.shape[-2::-1])[..., ::-1]
        else:
            resized = x[..., ::-1]
        if x.shape[-1] != obs_space.shape[-1]:
            assert (obs_space.shape[-1] % x.shape[-1]) == 0
            resized = np.repeat(resized, obs_space.shape[-1] // x.shape[-1], -1)
        ph = tf.convert_to_tensor(resized[None].astype('float32'))
        rsl = saliency_map(agent.online_model, ph).numpy()[0]
        rsl -=rsl.min()
        rsl = rsl / rsl.max()
        rsl *= 255
        rsl = cv2.resize(rsl.astype(np.uint8), dsize=x.shape[-2::-1]).astype('int32')
        print(rsl.shape)
        print(x.shape)
        rsl = normalize_and_add(rsl, x)
        stack.append(rsl.astype(np.uint8))
    stacks = [np.hstack(tuple(stack[i:i + columns]))
              for i in range(0, rows * columns, columns)]
    stacks = np.vstack(tuple(stacks))
    if name is None:
        cv2.imwrite('train/stack.png', stacks)
    else:
        cv2.imwrite('train/' + str(name) + '_stack.png', stacks)


if __name__ == '__main__':
    import os
    for folder in os.listdir('weights'):
        name = folder
        model_name = os.listdir(os.path.join('weights', folder))[0][:-6]
        images = os.listdir('images')
        step = 0
        for left in range(0, len(images), 4):
            right = min(left + 4, len(images))
            to_stack = [cv2.imread(os.path.join('images', im)) for im in images[left:right]]
            saliency_img('weights/{}/{}'.format(folder, model_name), (64, 64, 6), 10,
                         to_stack, 1, len(to_stack), '{}_'.format(step) + name)
            step += 1

    # images = [cv2.imread('minecraft.jpg') for _ in range(1)]
    # saliency_img('0_model.ckpt', (64, 64, 6), 10, images, 1, 1)
