from collections import defaultdict
from time import sleep

import gym
import minerl
from typing import List

from utils.fake_env import FakeEnv
from utils.config_validation import Action, Subtask


class TrajectoryInformation:
    def __init__(self, env_name='MineRLObtainIronPickaxeDense-v0', data_dir='demonstrations/',
                 trajectory_name='v3_rigid_mustard_greens_monster-11_878-4825'):
        data = minerl.data.make(env_name, data_dir)
        trajectory = data.load_data(stream_name=trajectory_name)

        self.name = trajectory_name

        current_states, actions, rewards, next_states, dones = [], [], [], [], []
        for current_state, action, reward, next_state, done in trajectory:
            current_states.append(current_state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        self.trajectory = (current_states, actions, rewards, next_states, dones)

        self.reward = int(sum(rewards))
        self.length = len(rewards)

        if 'Treechop' in env_name:
            self.subtasks = [Subtask(item_name='log', item_count=self.reward, start_idx=0, end_idx=len(current_states))]
            self.trajectory_by_subtask = {'log': self.trajectory}
        else:
            self.subtasks = self.extract_subtasks(self.trajectory)
            self.trajectory_by_subtask = self.slice_trajectory_by_item(self.extract_subtasks(self.trajectory, compress_items=False))

    def slice_trajectory_by_item(self, subtasks, minimal_length=4, fix_done=True, scale_rewards=True):

        states, actions, rewards, next_states, dones = self.trajectory

        sliced_states = defaultdict(list)
        sliced_actions = defaultdict(list)
        sliced_rewards = defaultdict(list)
        sliced_next_states = defaultdict(list)
        sliced_dones = defaultdict(list)

        result = defaultdict(list)

        for subtask in subtasks:
            # skip short ones
            if subtask.end_idx - subtask.start_idx < minimal_length:
                continue

            if fix_done:
                true_dones = [0 for _ in range(subtask.start_idx, subtask.end_idx)]
                true_dones[-1] = 1
            else:
                true_dones = dones[subtask.start_idx:subtask.end_idx]

            if scale_rewards:
                true_rewards = [0 for _ in range(subtask.start_idx, subtask.end_idx)]
                true_rewards[-1] = next_states[subtask.end_idx]['inventory'][subtask.item_name] - \
                                   states[subtask.end_idx]['inventory'][subtask.item_name]
            else:
                true_rewards = rewards[subtask.start_idx:subtask.end_idx]

            sliced_states[subtask.item_name] += states[subtask.start_idx:subtask.end_idx]
            sliced_actions[subtask.item_name] += actions[subtask.start_idx:subtask.end_idx]
            sliced_rewards[subtask.item_name] += true_rewards
            sliced_next_states[subtask.item_name] += next_states[subtask.start_idx:subtask.end_idx]
            sliced_dones[subtask.item_name] += true_dones

        unique_item_names = set([item.item_name for item in self.subtasks])
        for item_name in unique_item_names:
            result[item_name] = [sliced_states[item_name],
                                 sliced_actions[item_name],
                                 sliced_rewards[item_name],
                                 sliced_next_states[item_name],
                                 sliced_dones[item_name]]

        return result

    @classmethod
    def extract_subtasks(cls, trajectory,
                         excluded_actions=("attack", "back", "camera",
                                           "forward", "jump", "left",
                                           "right", "sneak", "sprint"),
                         compress_items=True) -> List[Subtask]:

        states, actions, rewards, next_states, _ = trajectory
        items = states[0].get('inventory', {}).keys()

        # add fake items to deal with crafting actions
        result: List[Subtask] = [Subtask(start_idx=0, end_idx=0)]

        for index in range(len(rewards)):

            for action in actions[index]:
                if action not in excluded_actions:
                    target = str(actions[index][action])
                    if target == 'none':
                        continue

                    a = Action(name=action, target=target)
                    last_subtask = result[-1]
                    if a.target:
                        if not last_subtask.actions or last_subtask.actions[-1] != a:
                            last_subtask.actions.append(a)
            for item in items:
                if next_states[index]['inventory'][item] > states[index]['inventory'][item]:
                    s = Subtask(item_name=item, start_idx=result[-1].end_idx, end_idx=index, item_count=next_states[index]['inventory'][item])
                    last_subtask = result[-1]
                    if s.item_name == last_subtask.item_name and compress_items:
                        # update the required number of items
                        last_subtask.item_count = s.item_count
                        last_subtask.end_idx = index
                    else:
                        # add new subtask
                        result.append(s)

        result.append(Subtask())
        for item, next_item in zip(reversed(result[:-1]), reversed(result[1:])):
            item.actions, next_item.actions = next_item.actions, item.actions

        # remove empty items
        result = [subtask for subtask in result if subtask.item_name is not None]

        return result


class FrameSkip(gym.Wrapper):
    """Return every `skip`-th frame and repeat given action during skip.
    Note that this wrapper does not "maximize" over the skipped frames.
    """

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


def main():
    trj_info = TrajectoryInformation(env_name='MineRLTreechop-v0', trajectory_name='v3_absolute_grape_changeling-7_14600-16079')
    # trj_info = TrajectoryInformation()
    env = FakeEnv(data=trj_info.trajectory_by_subtask['log'])
    env = FrameSkip(env, 2)
    # env = Monitor(env, 'monitor/')
    env.reset()
    done = False

    while True:
        done = False
        while not done:
            obs, rew, done, info = env.step(None)
            env.render(), sleep(0.01)

        if env.reset() is None:
            break


if __name__ == '__main__':
    main()
