import sys
from copy import deepcopy
import gym
import pathlib

from typing import List, Dict

from policy.agent import create_flat_agent, Agent
from utils.config_validation import Task, Subtask, Action
from utils.wrappers import wrap_env


class LoopCraftingAgent:
    def __init__(self, crafting_actions):
        self.crafting_actions = crafting_actions
        self.current_action_index = 0

    def get_crafting_action(self):
        if len(self.crafting_actions) == 0:
            return {}
        action: Action = self.crafting_actions[self.current_action_index]
        self.current_action_index = (self.current_action_index + 1) % len(self.crafting_actions)
        return {action.name: action.target}


class CraftInnerWrapper(gym.Wrapper):
    def __init__(self, env, crafting_agent):
        super().__init__(env)
        self.crafting_agent = crafting_agent

    def step(self, action):
        return self.env.step({**action, **self.crafting_agent.get_crafting_action()})


class InnerEnvWrapper(gym.Wrapper):
    def __init__(self, env, item, count, last_observation):
        super().__init__(env)
        self.item = item
        self.count = count
        self.previous_count = None
        self.is_core_env_done = None
        self.last_observation = last_observation

    def reset(self, **kwargs):
        return self.last_observation

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
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


class InventoryPrintWrapper(gym.Wrapper):
    def __init__(self, env, items=("log", "cobblestone", "planks", "stick", "wooden_pickaxe", "crafting_table",
                                   "stone_pickaxe", "iron_ore", "furnace", "iron_pickaxe",)):
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


class ItemAgent:

    def __init__(self, task: Task, nodes_dict=None):
        self.nodes_dict = nodes_dict
        self.subtasks: List[Subtask] = task.subtasks
        self.cfg = task.cfg
        self.task = task
        self.pov_agents: Dict[str:Agent] = {}
        self.agent_tasks = {}

    def train(self, core_env, task: Task, agents_to_train=("log", "cobblestone",)):

        t_env = wrap_env(core_env, task.cfg.wrappers)
        for subtask in self.subtasks:
            if subtask.item_name in self.pov_agents:
                continue
            agent_task = deepcopy(task)
            agent_task.max_train_episodes = 1
            agent_task.cfg.agent.save_dir = str(pathlib.Path(agent_task.cfg.agent.save_dir) / subtask.item_name)
            agent_task.evaluation = subtask.item_name not in agents_to_train

            self.agent_tasks[subtask.item_name] = agent_task
            self.pov_agents[subtask.item_name] = create_flat_agent(agent_task, t_env)

        for episode in range(1000):
            subtask_idx = 0
            obs = core_env.reset()
            while True:
                current_subtask: Subtask = self.subtasks[subtask_idx]
                inner_env = InnerEnvWrapper(env=core_env, item=current_subtask.item_name, count=current_subtask.item_count, last_observation=obs)
                inner_env = CraftInnerWrapper(inner_env, crafting_agent=LoopCraftingAgent(current_subtask.actions))

                t_env = wrap_env(inner_env, self.agent_tasks[current_subtask.item_name].cfg.wrappers)
                print('\n', current_subtask.item_name, "agent started")
                pov_agent = self.pov_agents[current_subtask.item_name]
                pov_agent.train(t_env, self.agent_tasks[current_subtask.item_name])

                subtask_idx += 1
                if inner_env.is_core_env_done or subtask_idx >= len(self.subtasks):
                    break
                obs = inner_env.last_observation
