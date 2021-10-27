import sys
import gym
from collections import deque, defaultdict
import os
import tqdm
import numpy as np

from utils.env_wrappers import ObtainPoVWrapper, FrameSkip, FrameStack, SaveVideoWrapper, TreechopDiscretWrapper
from hierarchical_tasks_extraction.extract_chain import TrajectoryInformation
from hierarchical_task_policy.HieAgent import RfDAgent
from hierarchical_tasks_extraction.utils import TrajectoryDataPipeline
from ForgER.model import get_network_builder
from utils.discretization import get_dtype_dict
from utils.data_loaders import TreechopLoader


class LoopCraftingAgent:
    """
    Agent that acts according to the chain
    """

    def __init__(self, crafting_actions):
        """
        :param crafting_actions: list of crafting actions list({},...)
        """
        self.crafting_actions = crafting_actions
        self.current_action_index = 0

    def get_crafting_action(self):
        """
        :return: action to be taken
        """
        if len(self.crafting_actions) == 0:
            return {}

        result = self.crafting_actions[self.current_action_index]

        # move the pointer to the next action in the list
        self.current_action_index = (self.current_action_index + 1) % len(self.crafting_actions)
        return result

    def reset_index(self):
        self.current_action_index = 0


class CraftInnerWrapper(gym.Wrapper):
    """
    Wrapper for crafting actions
    """

    def __init__(self, env, crafts_agent):
        """
        :param env: env to wrap
        :param crafts_agent: instance of LoopCraftingAgent
        """
        super().__init__(env)
        self.crafts_agent = crafts_agent

    def step(self, action):
        """
        mix craft action with POV action
        :param action: POV action
        :return:
        """
        craft_action = self.crafts_agent.get_crafting_action()
        action = {**action, **craft_action}
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info


class RememberFullTrajectoryWrapper(gym.Wrapper):
    def __init__(self, env, where_to_add):
        super().__init__(env)
        self.trajectory = None

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)

        assert len(self.trajectory) > 0

        if len(self.trajectory) == 1:
            current_state = self.trajectory[0]
            self.trajectory = []
        else:
            current_state, _, _, _, _ = self.trajectory[-1]
        self.trajectory.append((current_state, action, reward, next_state, done))
        return next_state, reward, done, _

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        if self.trajectory and len(self.trajectory) > 10:
            self.trajectory = [observation]
        return observation


class InnerEnvWrapper(gym.Wrapper):
    def __init__(self, env, item, count, last_observation):
        super().__init__(env)
        self.item = item
        self.count = count
        self.previous_count = None
        self.is_core_env_done = None
        self.last_observation = last_observation
        self.count_steps = 0

    def reset(self, **kwargs):
        return self.last_observation

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        self.count_steps += 1
        reward = 0
        if done:
            self.is_core_env_done = True
        if self.item not in state['inventory']:
            state['inventory'][self.item] = 0
        if state['inventory'][self.item] >= self.count:
            done = True

        if self.previous_count is not None and self.previous_count < state['inventory'][self.item]:
            reward += 1.0 * (state['inventory'][self.item] - self.previous_count)
        self.last_observation = state
        self.previous_count = state['inventory'][self.item]
        return state, reward, done, _

# print wrapper
class InventoryPrintWrapper(gym.Wrapper):
    def __init__(self, env, items=("log", "planks","crafting_table", "stick", "wooden_pickaxe", "dirt", "cobblestone", "stone",
                                   "coal", "stone_pickaxe", "iron_ore", "furnace", "iron_ingot", "iron_pickaxe",)):
        super().__init__(env)
        self.items = items
        self.inventory = None

    def get_inventory_info(self):
        inventory_info = ":"
        for item in self.items:
            inventory_info += f"[{item}:{self.inventory[item]}] "
        return inventory_info

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        inventory = state['inventory']
        if inventory != self.inventory:
            self.inventory = inventory
            sys.stdout.write('\r' + self.get_inventory_info())
            sys.stdout.flush()
        return state, reward, done, _


class ActionNoiseWrapper(gym.Wrapper):

    def step(self, action: dict):
        return self.env.step(self.apply_noise(action))

    @staticmethod
    def apply_noise(action):
        if 'camera' in action:
            x, y = list(np.random.normal(0, 0.7, 2))
            action['camera'] = np.add([x, y], action['camera'])
        return action


class ItemAgentNode:
    """
    combined info about each agent
    """

    def __init__(self, node_name, count_, pov_agent, crafting_agent):
        self.name = node_name
        self.count = count_
        self.pov_agent = pov_agent
        self.crafting_agent = crafting_agent
        self.success = deque([0], maxlen=10)
        self.eps_to_save = 0
        self.model_dir = 'train/' + self.name
        self.exploration_force = True
        self.fixed = False

    def load_agent(self, load_dir=None):
        if load_dir is None:
            load_dir = self.model_dir
        self.pov_agent.load_agent(load_dir)


class ItemAgent:
    pov_agents = {}

    def __init__(self, chain, nodes_dict=None):
        """
        :param chain: item/action chain
        :param nodes_dict:
        """
        self.nodes_dict = nodes_dict
        self.chain = chain

        self.nodes = self.create_nodes(self.chain)

    @staticmethod
    def str_to_action_dict(action_):
        """
        str -> dict
        :param action_:
        :return:
        """
        a_, _, value = action_.split(":")
        return {a_: value}

    @classmethod
    def get_crafting_actions_from_chain(cls, chain_, node_name_):
        """
        getting crafting actions from chain for node_name_ item
        :param chain_:
        :param node_name_: item
        :return:
        """
        previous_actions = []
        for vertex in chain_:
            if vertex == node_name_:
                break
            if not cls.is_item(vertex):
                previous_actions.append(vertex)
            else:
                previous_actions = []
        return [cls.str_to_action_dict(action_) for action_ in previous_actions]

    @staticmethod
    def is_item(name):
        """
        method to differ actions and items
        :param name:
        :return:
        """
        return len(name.split(":")) == 2

    @classmethod
    def create_nodes(cls, chain):
        nodes_names = [item for item in chain if cls.is_item(item)]

        craft_agents = []
        for node_name in nodes_names:
            get_crafting_actions_from_chain = cls.get_crafting_actions_from_chain(chain, node_name)
            craft_agents.append(LoopCraftingAgent(get_crafting_actions_from_chain))

        nodes_dict = {}
        nodes = []
        for index, (name, count) in enumerate([_.split(":") for _ in nodes_names]):
            if name not in nodes_dict.keys():
                nodes_dict[name] = ItemAgentNode(node_name=name,
                                                 count_=int(count),
                                                 pov_agent=None,
                                                 crafting_agent=craft_agents[index])

            nodes.append(nodes_dict[name])
        return nodes

    def train(self, agent_config, buffer_config, wrapper_config,
              env_name="MineRLObtainDiamond-v0", episodes=100,
              agents_to_train=("cobblestone", "iron_ore"), **eps_kwargs):
        """
        training method
        :param agent_config:
        :param buffer_config:
        :param wrapper_config:
        :param agents_to_train: names of agents to train
        :param env_name: name for training env
        :param episodes: number of episodes to train
        :param eps_kwargs: epsilon
        :return:
        """
        save_video_wrapper = SaveVideoWrapper(gym.make(env_name))
        core_env = ActionNoiseWrapper(InventoryPrintWrapper(save_video_wrapper))
        t_env = make_env(core_env, **wrapper_config)
        env_dict, dtype_dict = get_dtype_dict(t_env)
        wandb = agent_config["wandb"]

        #load node.pov_agent
        print('==============================Load agent(s) now==============================')
        for node in self.nodes:
            if node.name not in self.pov_agents:
                print('----' * 10)
                print(f"loading {node.name} agent")

                self.pov_agents[node.name] = RfDAgent(env_dict=env_dict,
                                                      item_dir='train/' + node.name,
                                                      agent_config=agent_config, buffer_config=buffer_config,
                                                      build_model=get_network_builder('minerl_dqfd'),
                                                      obs_space=t_env.observation_space, act_space=t_env.action_space,
                                                      dtype_dict=dtype_dict, **eps_kwargs)
                self.pov_agents[node.name].load_agent()
            node.pov_agent = self.pov_agents[node.name]

        for episode in range(episodes):

            print('\n==============================This is the episode {}=============================='.format(episode))

            current_node_index = 0
            for agent in self.nodes:
                agent.crafting_agent.reset_index()
            last_observation = core_env.reset()

            accu_timestep =0
            record = defaultdict()
            for node in self.nodes:
                record[str(node.name)] = defaultdict(list)
                record[str(node.name)]['reward'] = 0
                record[str(node.name)]['timestep'] = 0

            episode_timestep = 0

            while True:
                current_node = self.nodes[current_node_index]
                inner_env = CraftInnerWrapper(InnerEnvWrapper(
                    env=core_env, item=current_node.name, count=current_node.count, last_observation=last_observation),
                    crafts_agent=current_node.crafting_agent)
                t_env = make_env(inner_env, **wrapper_config)

                current_node.crafting_agent.reset_index()

                print('\nNow is {} agent.'.format(current_node.name))

                if current_node.name in agents_to_train or 'all' in agents_to_train:
                    reward, timestep_record = current_node.pov_agent.train(t_env)
                else:
                    reward, timestep_record = current_node.pov_agent.run(t_env)

                total_reward = save_video_wrapper.get_reward()
                timestep_record = timestep_record*4

                print('\ntimestep_record is: ',timestep_record)
                accu_timestep += timestep_record
                episode_timestep += timestep_record

                record[str(current_node.name)]['reward']=reward
                record[str(current_node.name)]['timestep'] = accu_timestep#inner_env.count_steps

                current_node_index += 1
                if episode % 100 == 0 and episode > 0:
                    if current_node.name in ['log', 'cobblestone', 'iron_ore']:
                        current_node.pov_agent.save_agent(pre_train = False)

                if inner_env.is_core_env_done or current_node_index >= len(self.nodes):
                    break
                last_observation = inner_env.last_observation

            print('\nrecord is:\n',record)

            for i in record:
                if record[i]['reward']!=0:
                    wandb.log({i + " reward": record[i]['reward'], "episode": episode, i + " time_step": record[i]['timestep']})
                else:
                    wandb.log({i + " reward": record[i]['reward'], "episode": episode, i + " time_step": 0})

            #keys, values = zip(*inner_env.last_observation['inventory'].items())
            #values = tuple(int(i) for i in values)
            #for i in range(len(keys)):
            #    wandb.log({"Item "+ keys[i]: values[i], "episode": episode})
            wandb.log({"Total reward": total_reward, "episode": episode})


            print('\nThe timestep of the whole episode is: ', episode_timestep)

            print('==============================This is the episode {}=============================='.format(episode))


    def run(self, agent_config, wrapper_config,
            env_name="MineRLObtainDiamond-v0", episodes=100):
        return self.train(agent_config, None, wrapper_config,
                          env_name, episodes, agents_to_train=(), epsilon=0.01)

    def pre_train(self, agent_config, buffer_config, wrapper_config,
                  env_name, pretrain_config):
        """
        pre-train method for ForgER
        :return:
        """
        sliced_trajectories = self.load_sliced_trajectories(envs=("MineRLObtainIronPickaxeDense-v0",),
                                                            data_dir='demonstrations')

        print('=======================load_sliced_trajectories=======================================')
        agents_to_train = pretrain_config['agents_to_train']

        class DummyDataLoader:
            def __init__(self, data, items_to_add):
                self.data = data
                self.items_to_add = items_to_add

            def batch_iter(self, *args, **kwargs):
                for item in self.items_to_add:
                    for slice_ in self.data[item]:
                        yield slice_

        test_env = gym.make(env_name)
        test_env = make_env(test_env, **wrapper_config)
        obs_space = test_env.observation_space
        act_space = test_env.action_space
        env_dict, dtype_dict = get_dtype_dict(test_env)
        test_env.close()

        for index, node in enumerate(self.nodes):
            if node.name not in agents_to_train and 'all' not in agents_to_train:
                continue

            items_dir = 'train/' + node.name
            node.pov_agent = RfDAgent(items_dir, agent_config, buffer_config, env_dict,
                                      build_model=get_network_builder('minerl_dqfd'),
                                      obs_space=obs_space, act_space=act_space,
                                      dtype_dict=dtype_dict)

            data = DummyDataLoader(data=sliced_trajectories, items_to_add=[node.name])
            data = TreechopLoader(data, **wrapper_config, threshold=pretrain_config['min_demo_reward'])
            node.pov_agent.agent.add_demo(data)
            print('pre-train node.pov_agent.agent.add_demo(data.data) finished.......')
            if pretrain_config['hie_aug']:
                all_except_current = [name for name in sliced_trajectories if name != node.name]
                data = DummyDataLoader(data=sliced_trajectories, items_to_add=all_except_current)
                data = TreechopLoader(data, **wrapper_config, threshold=0)
                node.pov_agent.agent.add_demo(data, 0, fixed_reward=0)
                print('pre-train hie finished.......')


            print(f'Pre-training {node.name} agent')
            node.pov_agent.agent.pre_train(pretrain_config['steps'])
            node.pov_agent.save_agent(pre_train = True)
            self.pov_agents[node.name] = node.pov_agent
            print('{} agent finished'.format(node.name))

    def load_agents(self, agent_config, buffer_config, wrapper_config,
                  env_name, pretrained=False):

        test_env = gym.make(env_name)
        test_env = make_env(test_env, **wrapper_config)
        obs_space = test_env.observation_space
        act_space = test_env.action_space
        env_dict, dtype_dict = get_dtype_dict(test_env)
        test_env.close()

        print('----------load agents begin----------')
        for index, node in enumerate(self.nodes):
            items_dir = 'train/' + node.name

            node.pov_agent = RfDAgent(items_dir, agent_config, buffer_config, env_dict,
                                      build_model=get_network_builder('minerl_dqfd'),
                                      obs_space=obs_space, act_space=act_space,
                                      dtype_dict=dtype_dict)

            if pretrained==False:
                node.pov_agent.load_agent()
            else:
                node.pov_agent.load_agent()
            print('{} agent finished'.format(node.name))

        print('----------load agents finished----------')


    @staticmethod
    def load_sliced_trajectories(envs, data_dir, max_trajectories_to_load=30):
        result = defaultdict(list)

        uploaded = 0
        for env in envs:
            path_to_trajectories = os.path.join(data_dir, env)
            for trajectory_name in tqdm.tqdm(TrajectoryDataPipeline.get_trajectory_names(path_to_trajectories)):
                uploaded += 1
                if uploaded >= max_trajectories_to_load:
                    break
                file_dir = os.path.join(data_dir, env, trajectory_name)
                trajectory = TrajectoryDataPipeline.load_data(file_dir=file_dir)
                trajectory_info = TrajectoryInformation(path_to_trajectory=file_dir)
                sliced_trajectory = trajectory_info.slice_trajectory_by_item(trajectory)
                for item in sliced_trajectory:
                    result[item] = [*result[item], *sliced_trajectory[item]]
        return result


def make_env(env, frame_skip=4, frame_stack=2, always_attack=1):
    env = ObtainPoVWrapper(env)
    env = FrameSkip(env, frame_skip)
    env = FrameStack(env, frame_stack)
    env = TreechopDiscretWrapper(env, always_attack)
    return env
