import os
from collections import defaultdict
import numpy as np
import minerl
from typing import List

from hierarchical_tasks_extraction.Item import Item, Action
from hierarchical_tasks_extraction.utils import TrajectoryDataPipeline


class ChainInfo:
    def __init__(self, chain, reward, env_name, trajectory_name, id_, length, time_indexes):
        self.chain = chain
        self.reward = reward
        self.env = env_name
        self.trajectory_name = trajectory_name
        self.id = id_
        self.length = length
        self.time_indexes = time_indexes

    def __str__(self):
        return str(self.reward) + "\n" + str(self.chain)


class TrajectoryInformation:
    def __init__(self, path_to_trajectory, trajectory=None):
        self.path_to_trajectory = path_to_trajectory
        self.trajectory_name = os.path.basename(path_to_trajectory)
        if trajectory is None:
            trajectory = TrajectoryDataPipeline.load_data_no_pov(self.path_to_trajectory)
        state, action, reward, next_state, done = trajectory
        self.chain = self.extract_subtasks(trajectory)
        self.reward = int(sum(reward))
        self.length = len(reward)

    def __str__(self):
        return self.path_to_trajectory + '\n' + str(self.chain)

    @classmethod
    def extract_from_dict(cls, dictionary, left, right):
        result = dict()
        for key, value in dictionary.items():
            if isinstance(value, dict):
                result[key] = cls.extract_from_dict(value, left, right)
            else:
                result[key] = value[left:right]
        return result

    def slice_trajectory_by_item(self, trajectory):

        if trajectory is None:
            trajectory = TrajectoryDataPipeline.load_data(self.path_to_trajectory)
        state, action, reward, next_state, done = trajectory
        if self.length != len(reward):
            print(self.length, len(reward))
            raise NameError("Please, double check trajectory")
        result = defaultdict(list)
        for item in self.chain:
            # skip short ones
            if item.end - item.begin < 4:
                continue
            sliced_state = self.extract_from_dict(state, item.begin, item.end)
            sliced_action = self.extract_from_dict(action, item.begin, item.end)
            sliced_reward = reward[item.begin:item.end]
            sliced_next_state = self.extract_from_dict(next_state, item.begin, item.end)
            sliced_done = done[item.begin:item.end]
            result[item.name].append([sliced_state, sliced_action, sliced_reward, sliced_next_state, sliced_done])
        return result

    @staticmethod
    def to_old_chain_format(items: List[Item], return_time_indexes: bool):
        result = []
        used_actions = defaultdict(int)
        for item in items:
            for action in item.actions:
                full_action = f"{action.name}{action.value}"
                result.append(f"{action.name}:{used_actions[full_action]}:{action.value}")
                used_actions[full_action] += 1
            result.append(f"{item.name}:{item.value}")
        time_indexes = [(f"{item.name}+{item.value}", item.begin, item.end) for item in items]
        if return_time_indexes:
            return result, time_indexes
        return result

    @classmethod
    def compute_item_order(cls, trajectory, return_time_indexes=False, ):
        return cls.to_old_chain_format(cls.extract_subtasks(trajectory), return_time_indexes=return_time_indexes)

    @classmethod
    def extract_subtasks(cls, trajectory,
                         excluded_actions=("attack", "back", "camera",
                                           "forward", "jump", "left",
                                           "right", "sneak", "sprint"),
                         item_appear_limit=4) -> List[Item]:
        """
        computes item and actions order in time order
        :param trajectory:
        :param excluded_actions: by default all POV actions is excluded
        :param item_appear_limit: filter item vertexes appeared more then item_appear_limit times
        :return:
        """
        states, actions, rewards, next_states, _ = trajectory
        items = states['inventory'].keys()

        # add auxiliary items to deal with crafting actions
        empty_item = Item(name='empty', value=0, begin=-1, end=0)
        result: List[Item] = [empty_item]

        for index in range(len(rewards)):

            for action in actions:
                if action not in excluded_actions:
                    a = Action(name=action, value=actions[action][index])
                    last_item = result[-1]
                    if not a.is_noop():
                        if last_item.get_last_action() != a:
                            last_item.add_action(a)
            for item in items:
                if next_states['inventory'][item][index] > states['inventory'][item][index]:
                    i = Item(item, next_states['inventory'][item][index], begin=result[-1].end, end=index)
                    last_item = result[-1]
                    if i.name == last_item.name:
                        # update the required number of items
                        last_item.value = i.value
                        last_item.end = index
                    else:
                        # add new item in chain
                        result.append(i)

        result.append(empty_item)
        for item, next_item in zip(reversed(result[:-1]), reversed(result[1:])):
            item.actions, next_item.actions = next_item.actions, item.actions

        # trying to remove bugs with putting and getting items on the crafting table and furnace
        to_remove = set()
        for index, item in enumerate(result):
            if item.begin == item.end:
                to_remove.add(index)
                if index - 1 >= 0:
                    to_remove.add(index - 1)
            if sum([1 for _ in result[:index + 1] if _.name == item.name]) >= item_appear_limit:
                to_remove.add(index)

        for index in reversed(sorted(list(to_remove))):
            if result[index].actions:
                # saving useful actions of wrong items
                result[index + 1].actions = (*result[index].actions, *result[index + 1].actions)
            result.pop(index)

        # remove empty items
        result = [item for item in result if item != empty_item]

        return result


def all_chains_info(envs, data_dir):
    chains = []

    def get_reward(trajectory_):
        return int(sum(trajectory_[2]))

    for env_name in envs:
        data = minerl.data.make(env_name, data_dir=data_dir)

        for index, trajectory_name in enumerate(sorted(data.get_trajectory_names())):
            print(trajectory_name)
            trajectory = TrajectoryDataPipeline.load_data_no_pov(
                os.path.join(data_dir, env_name, trajectory_name))
            # trajectory = load_data_without_pov(
            #     os.path.join(data_dir, env_name, trajectory_name))

            chain, time_indexes = TrajectoryInformation.compute_item_order(trajectory, return_time_indexes=True)
            chains.append(ChainInfo(chain=chain, reward=get_reward(trajectory), env_name=env_name,
                                    trajectory_name=trajectory_name, id_=index, length=len(trajectory[2]),
                                    time_indexes=time_indexes))

    return chains


def generate_best_chains(envs=("MineRLObtainIronPickaxeDense-v0",), data_dir="../data/"):
    """
    generates final chain
    it may sampled randomly, but be careful short chains give poor results
    :param envs: number of envs
    :param data_dir:
    :return:
    """
    chains = all_chains_info(envs=envs, data_dir=data_dir)
    filtered = [c for c in chains if c.reward == max([_.reward for _ in chains])]
    filtered = [c for c in sorted(filtered, key=lambda x: x.length)][:60]
    filtered = [c for c in sorted(filtered, key=lambda x: len(x.chain)) if 25 < len(c.chain) <= 31]
    filtered_chains = []
    for chain in filtered:
        filtered_chains.append(chain.chain)
    return filtered_chains


def generate_final_chain(envs=("MineRLObtainIronPickaxeDense-v0",), data_dir="../data/"):
    """
    generates final chain
    it may sampled randomly, but be careful short chains give poor results
    :param envs: number of envs
    :param data_dir:
    :return:
    """
    return generate_best_chains(envs=envs, data_dir=data_dir)[-1]
