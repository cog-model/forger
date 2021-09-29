import collections
import os

import cv2
import gym
import numpy as np
from minerl.data import DataPipeline


class TrajectoryDataPipeline:
    """
    number of tools to load trajectory
    """

    @staticmethod
    def get_trajectory_names(data_dir):
        # noinspection PyProtectedMember
        result = [os.path.basename(x) for x in DataPipeline._get_all_valid_recordings(data_dir)]
        return sorted(result)

    @staticmethod
    def map_to_dict(handler_list: list, target_space: gym.spaces.space, ignore_keys=()):
        def _map_to_dict(i: int, src: list, key: str, gym_space: gym.spaces.space, dst: dict):

            if isinstance(gym_space, gym.spaces.Dict):
                dont_count = False
                inner_dict = collections.OrderedDict()
                for idx, (k, s) in enumerate(gym_space.spaces.items()):
                    if key in ['equipped_items', 'mainhand']:
                        dont_count = True
                        i = _map_to_dict(i, src, k, s, inner_dict)
                    else:
                        _map_to_dict(idx, src[i].T, k, s, inner_dict)
                dst[key] = inner_dict
                if dont_count:
                    return i
                else:
                    return i + 1
            else:
                dst[key] = src[i]
                return i + 1

        result = collections.OrderedDict()
        index = 0
        for key, space in target_space.spaces.items():
            if key in ignore_keys:
                continue
            index = _map_to_dict(index, handler_list, key, space, result)
        return result

    @staticmethod
    def load_video_frames(video_path, suffix_size):
        cap = cv2.VideoCapture(video_path)
        ret, frame_num = True, 0
        while ret:
            ret, _ = DataPipeline.read_frame(cap)
            if ret:
                frame_num += 1

        num_states = suffix_size
        frames = []
        max_frame_num = frame_num
        frame_num = 0
        # Advance video capture past first i-frame to start of experiment
        cap = cv2.VideoCapture(video_path)
        for _ in range(max_frame_num - num_states):
            ret, _ = DataPipeline.read_frame(cap)
            frame_num += 1
            if not ret:
                return None

        while ret and frame_num < max_frame_num:
            ret, frame = DataPipeline.read_frame(cap)
            frames.append(frame)
            frame_num += 1
        return frames

    # noinspection PyProtectedMember
    @classmethod
    def load_data(cls, file_dir, ignore_keys=()):

        from copy import deepcopy
        def rename(old_dict, old_name, new_name):
            new_dict = {}
            for key, value in zip(old_dict.keys(), old_dict.values()):
                new_key = key if key != old_name else new_name
                new_dict[new_key] = old_dict[key]
            return new_dict

        numpy_path = str(os.path.join(file_dir, 'rendered.npz'))
        video_path = str(os.path.join(file_dir, 'recording.mp4'))

        state = np.load(numpy_path, allow_pickle=True)
        action_dict = dict([(key, state[key]) for key in state if key.startswith('action')])

        # Modify the form of action_dict to action_dict_v2
        action_dict_v2 = ([(key, state[key]) for key in state if key.startswith('action')])
        action_dict_v2 = sorted(action_dict_v2)
        action_dict_v2 = collections.OrderedDict(action_dict_v2)

        action_dict_v2 = rename(action_dict_v2, 'action$attack', 'action_attack')
        action_dict_v2 = rename(action_dict_v2, 'action$back', 'action_back')
        action_dict_v2 = rename(action_dict_v2, 'action$camera', 'action_camera')
        action_dict_v2 = rename(action_dict_v2, 'action$craft', 'action_craft')
        action_dict_v2 = rename(action_dict_v2, 'action$equip', 'action_equip')
        action_dict_v2 = rename(action_dict_v2, 'action$forward', 'action_forward')
        action_dict_v2 = rename(action_dict_v2, 'action$jump', 'action_jump')
        action_dict_v2 = rename(action_dict_v2, 'action$left', 'action_left')
        action_dict_v2 = rename(action_dict_v2, 'action$right', 'action_right')
        action_dict_v2 = rename(action_dict_v2, 'action$nearbyCraft', 'action_nearbyCraft')
        action_dict_v2 = rename(action_dict_v2, 'action$nearbySmelt', 'action_nearbySmelt')
        action_dict_v2 = rename(action_dict_v2, 'action$sprint', 'action_sprint')
        action_dict_v2 = rename(action_dict_v2, 'action$place', 'action_place')
        action_dict_v2 = rename(action_dict_v2, 'action$sneak', 'action_sneak')
        action_dict_v2 = collections.OrderedDict(action_dict_v2)

        actions = list(action_dict_v2.keys())

        action_data = [None for _ in actions]

        for i, key in enumerate(actions):
            action_data[i] = np.asanyarray(action_dict_v2[key])

        # action_data = correct form except some action is none, but it is because of the dataset

        current_observation_data = [None for _ in actions]
        next_observation_data = [None for _ in actions]

        reward_vec = state['reward']
        reward_data = np.asanyarray(reward_vec, dtype=np.float32)
        done_data = [False for _ in range(len(reward_data))]
        done_data[-1] = True
        info_dict = collections.OrderedDict([(key, state[key]) for key in state if key.startswith('observation')])

        info_dict_mainhand = collections.OrderedDict(
            [(key, state[key]) for key in state if key.startswith('observation$equipped_items.mainhand.')])
        info_dict_mainhand_v1 = deepcopy(info_dict_mainhand)
        info_dict_mainhand_v1 = rename(info_dict_mainhand_v1, 'observation$equipped_items.mainhand.damage',
                                       'observation_damage')
        info_dict_mainhand_v1 = rename(info_dict_mainhand_v1, 'observation$equipped_items.mainhand.maxDamage',
                                       'observation_maxDamage')
        info_dict_mainhand_v1 = rename(info_dict_mainhand_v1, 'observation$equipped_items.mainhand.type',
                                       'observation_type')

        info_dict_inv = collections.OrderedDict(
            [(key, state[key]) for key in state if key.startswith('observation$inventory')])
        info_dict_inv_v1 = deepcopy(info_dict_inv)
        info_dict_mainhand_v1.update({'observation_inventory': np.array(list(info_dict_inv_v1.values()))})
        info_dict_mainhand_v1['observation_inventory'] = info_dict_mainhand_v1['observation_inventory'].T

        info_dict_mainhand_v1 = collections.OrderedDict(info_dict_mainhand_v1)

        # info_dict_mainhand_v1 is the correct form of info_dict, inv is also correct[Varified]

        observables = list(info_dict_mainhand_v1.keys()).copy()
        observables.append('pov')

        if 'pov' not in ignore_keys:
            frames = cls.load_video_frames(video_path=video_path, suffix_size=len(reward_vec) + 1)
        else:
            frames = None

        for i, key in enumerate(observables):
            if key in ignore_keys:
                continue
            if key == 'pov':
                current_observation_data[i] = np.asanyarray(frames[:-1])
                next_observation_data[i] = np.asanyarray(frames[1:])
            else:
                current_observation_data[i] = np.asanyarray(info_dict_mainhand_v1[key][:-1])
                next_observation_data[i] = np.asanyarray(info_dict_mainhand_v1[key][1:])

        gym_spec = gym.envs.registration.spec('MineRLObtainIronPickaxeDense-v0')
        observation_dict = cls.map_to_dict(current_observation_data, gym_spec._kwargs['observation_space'])
        next_observation_dict = cls.map_to_dict(next_observation_data, gym_spec._kwargs['observation_space'])
        action_dict = cls.map_to_dict(action_data, gym_spec._kwargs['action_space'])

        # return action_dict_v2,observation_dict,next_observation_dict,action_dict,current_observation_data,next_observation_data

        #New test function
        def convert_obs_dict(observation_dict):

            def rename(old_dict, old_name, new_name):
                new_dict = {}
                for key, value in zip(old_dict.keys(), old_dict.values()):
                    new_key = key if key != old_name else new_name
                    new_dict[new_key] = old_dict[key]
                return new_dict

            from copy import deepcopy
            sss = deepcopy(observation_dict)

            temp = collections.OrderedDict({'damage': dict(sss)['equipped_items.mainhand.damage'],
                                            'maxDamage': dict(sss)['equipped_items.mainhand.maxDamage'],
                                            'type': dict(sss)['equipped_items.mainhand.type']})

            sss['equipped_items.mainhand.damage'] = collections.OrderedDict({'mainhand': temp})
            sss = rename(sss, 'equipped_items.mainhand.damage', 'equipped_items')

            del (sss['equipped_items.mainhand.maxDamage'])
            del (sss['equipped_items.mainhand.type'])

            sss = collections.OrderedDict(sss)

            return sss

        observation_dict = convert_obs_dict(observation_dict)

        next_observation_dict = convert_obs_dict(next_observation_dict)

        return [observation_dict, action_dict, reward_data, next_observation_dict, done_data]

    @classmethod
    def load_data_no_pov(cls, file_dir):
        return cls.load_data(file_dir, ignore_keys=('pov',))


class VisTools:
    """
    number of methods to draw chains with pyGraphviz
    """

    @staticmethod
    def get_all_vertexes_from_edges(edges):
        """
        determines all vertex of a graph
        :param edges: list of edges
        :return: list of vertexes
        """
        vertexes = []
        for left, right in edges:
            if left not in vertexes:
                vertexes.append(left)
            if right not in vertexes:
                vertexes.append(right)
        return vertexes

    @staticmethod
    def get_colored_vertexes(chain):
        """
        determines the color for each vertex
        item + its actions have the same color
        :param chain:
        :return: {vertex: color}
        """
        vertexes = VisTools.get_all_vertexes_from_edges(chain)
        result = {}
        colors = ["#ffe6cc", "#ccffe6"]
        current_color = 0

        for vertex in vertexes:
            result[vertex] = colors[current_color]

            bool_ = True
            for action in ["equip", "craft", "nearbyCraft", "nearbySmelt", 'place']:
                if action + ":" in vertex:
                    bool_ = False
            if bool_:
                current_color = (current_color + 1) % len(colors)

        return result

    @staticmethod
    def replace_with_name(name):
        """
        replace all names with human readable variants
        :param name: crafting action or item (item string will be skipped)
        :return: human readable name
        """
        if len(name.split(":")) == 3:
            name, order, digit = name.split(":")
            name = name + ":" + order
            translate = {"place": ["none", "dirt", "stone", "cobblestone", "crafting_table", "furnace", "torch"],
                         "nearbySmelt": ["none", "iron_ingot", "coal"],
                         "nearbyCraft": ["none", "wooden_axe", "wooden_pickaxe", "stone_axe", "stone_pickaxe",
                                         "iron_axe",
                                         "iron_pickaxe", "furnace"],
                         "equip": ["none", "air", "wooden_axe", "wooden_pickaxe", "stone_axe", "stone_pickaxe",
                                   "iron_axe",
                                   "iron_pickaxe"],
                         "craft": ["none", "torch", "stick", "planks", "crafting_table"],
                         }
            name_without_digits = name
            while name_without_digits not in translate:
                name_without_digits = name_without_digits[:-1]
            return name + " -> " + translate[name_without_digits][int(digit)]
        else:
            return name

    @staticmethod
    def draw_graph(file_name, graph, format_="svg", vertex_colors=None):
        """
        drawing png graph from the list of edges
        :param vertex_colors:
        :param format_: resulted file format
        :param file_name: file_name
        :param graph: graph file with format: (left_edge, right_edge) or (left_edge, right_edge, label)
        :return: None
        """
        import pygraphviz as pgv
        g_out = pgv.AGraph(strict=False, directed=True)
        for i in graph:
            g_out.add_edge(i[0], i[1], color='black')
            edge = g_out.get_edge(i[0], i[1])
            if len(i) > 2:
                edge.attr['label'] = i[2]
        g_out.node_attr['style'] = 'filled'
        if vertex_colors:
            for vertex, color in vertex_colors.items():
                g_out.get_node(vertex).attr['fillcolor'] = color
        g_out.layout(prog='dot')
        g_out.draw(path="{file_name}.{format_}".format(**locals()))

    @staticmethod
    def save_chain_in_graph(chain_to_save, name="out", format_="png"):
        """
        saving image of a graph using draw_graph method
        :param chain_to_save:
        :param name: filename
        :param format_: file type e.g. ".png" or ".svg"
        :return:
        """
        graph = []

        for c_index, item in enumerate(chain_to_save):
            if c_index:
                graph.append(
                    [str(c_index) + '\n' + VisTools.replace_with_name((chain_to_save[c_index - 1])),
                     str(c_index + 1) + '\n' + VisTools.replace_with_name(item)])
        VisTools.draw_graph(name, graph=graph, format_=format_,
                            vertex_colors=VisTools.get_colored_vertexes(graph))
