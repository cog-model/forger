import random
import timeit
from collections import deque
import pathlib
import numpy as np
import tensorflow as tf

from policy.models import get_network_builder
from policy.replay_buffer import AggregatedBuff
from utils.config_validation import AgentCfg, Task
from utils.discretization import get_dtype_dict
from utils.tf_util import huber_loss, take_vector_elements


def create_flat_agent(task: Task, env):
    make_model = get_network_builder(task.model_name)
    env_dict, dtype_dict = get_dtype_dict(env)
    replay_buffer = AggregatedBuff(env_dict, task.cfg.buffer)
    agent = Agent(task.cfg.agent, replay_buffer, make_model, env.observation_space, env.action_space, dtype_dict)
    if not task.from_scratch:
        agent.load(task.cfg.agent.save_dir)
    return agent


class Agent:
    def __init__(self, cfg: AgentCfg, replay_buffer, build_model, obs_space, act_space,
                 dtype_dict=None, log_freq=100):

        self.cfg = cfg
        self.n_deque = deque([], maxlen=cfg.n_step)

        self.replay_buff = replay_buffer
        self.priorities_store = list()
        if dtype_dict is not None:
            ds = tf.data.Dataset.from_generator(self.sample_generator, output_types=dtype_dict)
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            self.sampler = ds.take
        else:
            self.sampler = self.sample_generator
        self.online_model = build_model('Online_Model', obs_space, act_space, self.cfg.l2)
        self.target_model = build_model('Target_Model', obs_space, act_space, self.cfg.l2)
        self.optimizer = tf.keras.optimizers.Adam(self.cfg.learning_rate)
        self._run_time_deque = deque(maxlen=log_freq)
        self._schedule_dict = dict()
        self._schedule_dict[self.target_update] = self.cfg.update_target_net_mod
        self._schedule_dict[self.update_log] = log_freq
        self.avg_metrics = dict()
        self.action_dim = act_space.n
        self.global_step = 0

    def train(self, env, task: Task):
        print('starting from step:', self.global_step)
        scores = []

        epsilon = self.cfg.initial_epsilon
        current_episode = 0
        while self.global_step < task.max_train_steps and current_episode < task.max_train_episodes:
            score = self.train_episode(env, task, epsilon)
            print(f'Steps: {self.global_step}, Episode: {current_episode}, Reward: {score}, Eps Greedy: {round(epsilon, 3)}')
            current_episode += 1
            if self.global_step >= self.cfg.epsilon_time_steps:
                epsilon = self.cfg.final_epsilon
            else:
                epsilon = (self.cfg.initial_epsilon - self.cfg.final_epsilon) * \
                          (self.cfg.epsilon_time_steps - self.global_step) / self.cfg.epsilon_time_steps

            scores.append(score)
            tf.summary.scalar("reward", score, step=self.global_step)
            tf.summary.flush()

        return scores

    def train_episode(self, env, task: Task, epsilon=0.0):
        if self.global_step == 0:
            self.target_update()
        done, score, state = False, 0, env.reset()
        while not done:
            action = self.choose_act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            if task.cfg.wrappers.render:
                env.render()
            score += reward

            self.global_step += 1
            if not task.evaluation:
                # print(f'saving to {task.cfg.agent.save_dir}')
                self.perceive(to_demo=0, state=state, action=action, reward=reward, next_state=next_state,
                              done=done, demo=False)
                if self.replay_buff.get_stored_size() > self.cfg.replay_start_size:
                    if self.global_step % self.cfg.frames_to_update == 0:
                        self.update(task.cfg.agent.update_quantity)
                        self.save(task.cfg.agent.save_dir)
                        print(f'saving to {task.cfg.agent.save_dir}')

            state = next_state
        return score

    def pre_train(self, task):
        """
        pre_train phase in policy alg.
        :return:
        """
        print('Pre-training ...')
        self.target_update()
        self.update(task.pretrain_num_updates)
        # self.save(os.path.join(self.cfg.save_dir, "pre_trained_model.ckpt"))
        print('All pre-train finish.')

    def update(self, num_updates):
        start_time = timeit.default_timer()
        for batch in self.sampler(num_updates):
            indexes = batch.pop('indexes')
            priorities = self.q_network_update(gamma=self.cfg.gamma, **batch)
            self.schedule()
            self.priorities_store.append({'indexes': indexes.numpy(), 'priorities': priorities.numpy()})
            stop_time = timeit.default_timer()
            self._run_time_deque.append(stop_time - start_time)
            start_time = timeit.default_timer()
        while len(self.priorities_store) > 0:
            priorities = self.priorities_store.pop(0)
            self.replay_buff.update_priorities(**priorities)

    def sample_generator(self, steps=None):
        steps_done = 0
        finite_loop = bool(steps)
        steps = steps if finite_loop else 1
        while steps_done < steps:
            yield self.replay_buff.sample(self.cfg.batch_size)
            if len(self.priorities_store) > 0:
                priorities = self.priorities_store.pop(0)
                self.replay_buff.update_priorities(**priorities)
            steps += int(finite_loop)

    @tf.function
    def q_network_update(self, state, action, next_state, done, reward, demo,
                         n_state, n_done, n_reward, actual_n, weights,
                         gamma):
        print("Q-nn_update tracing")
        online_variables = self.online_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(online_variables)
            q_value = self.online_model(state, training=True)
            margin = self.margin_loss(q_value, action, demo, weights)
            self.update_metrics('margin', margin)

            q_value = take_vector_elements(q_value, action)

            td_loss = self.td_loss(q_value, next_state, done, reward, 1, gamma)
            huber_td = huber_loss(td_loss, delta=0.4)
            mean_td = tf.reduce_mean(huber_td * weights)
            self.update_metrics('TD', mean_td)

            ntd_loss = self.td_loss(q_value, n_state, n_done, n_reward, actual_n, gamma)
            huber_ntd = huber_loss(ntd_loss, delta=0.4)
            mean_ntd = tf.reduce_mean(huber_ntd * weights)
            self.update_metrics('nTD', mean_ntd)

            l2 = tf.add_n(self.online_model.losses)
            self.update_metrics('l2', l2)

            all_losses = mean_td + mean_ntd + l2 + margin
            self.update_metrics('all_losses', all_losses)

        gradients = tape.gradient(all_losses, online_variables)
        self.optimizer.apply_gradients(zip(gradients, online_variables))
        priorities = tf.abs(td_loss)
        return priorities

    def td_loss(self, q_value, n_state, n_done, n_reward, actual_n, gamma):
        n_target = self.compute_target(n_state, n_done, n_reward, actual_n, gamma)
        n_target = tf.stop_gradient(n_target)
        ntd_loss = q_value - n_target
        return ntd_loss

    def compute_target(self, next_state, done, reward, actual_n, gamma):
        print("Compute_target tracing")
        q_network = self.online_model(next_state, training=True)
        argmax_actions = tf.argmax(q_network, axis=1, output_type='int32')
        q_target = self.target_model(next_state, training=True)
        target = take_vector_elements(q_target, argmax_actions)
        target = tf.where(done, tf.zeros_like(target), target)
        target = target * gamma ** actual_n
        target = target + reward
        return target

    def margin_loss(self, q_value, action, demo, weights):
        ae = tf.one_hot(action, self.action_dim, on_value=0.0,
                        off_value=self.cfg.margin)
        ae = tf.cast(ae, 'float32')
        max_value = tf.reduce_max(q_value + ae, axis=1)
        ae = tf.one_hot(action, self.action_dim)
        j_e = tf.abs(tf.reduce_sum(q_value * ae, axis=1) - max_value)
        j_e = tf.reduce_mean(j_e * weights * demo)
        return j_e

    def add_demo(self, expert_env, expert_data=1):
        while not expert_env.are_all_frames_used():
            done = False
            obs = expert_env.reset()

            while not done:
                next_obs, reward, done, info = expert_env.step(0)
                action = info['expert_action']
                self.perceive(to_demo=1, state=obs, action=action, reward=reward, next_state=next_obs, done=done,
                              demo=expert_data)
                obs = next_obs

    def perceive(self, **kwargs):
        self.n_deque.append(kwargs)

        if len(self.n_deque) == self.n_deque.maxlen or kwargs['done']:
            while len(self.n_deque) != 0:
                n_state = self.n_deque[-1]['next_state']
                n_done = self.n_deque[-1]['done']
                n_reward = sum([t['reward'] * self.cfg.gamma ** i for i, t in enumerate(self.n_deque)])
                self.n_deque[0]['n_state'] = n_state
                self.n_deque[0]['n_reward'] = n_reward
                self.n_deque[0]['n_done'] = n_done
                self.n_deque[0]['actual_n'] = len(self.n_deque)
                self.replay_buff.add(**self.n_deque.popleft())
                if not n_done:
                    break

    def choose_act(self, state, epsilon=0.01):
        nn_input = np.array(state)[None]
        q_value = self.online_model(nn_input, training=False)
        if random.random() <= epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(q_value)

    def schedule(self):
        for key, value in self._schedule_dict.items():
            if tf.equal(self.optimizer.iterations % value, 0):
                key()

    def target_update(self):
        self.target_model.set_weights(self.online_model.get_weights())

    def save(self, out_dir=None):
        self.online_model.save_weights(pathlib.Path(out_dir) / 'model.ckpt')

    def load(self, out_dir=None):

        if pathlib.Path(out_dir).exists():
            self.online_model.load_weights(pathlib.Path(out_dir) / 'model.ckpt')
        else:
            raise KeyError(f"Can not import weights from {pathlib.Path(out_dir)}")

    def update_log(self):
        update_frequency = len(self._run_time_deque) / sum(self._run_time_deque)
        print("LearnerEpoch({:.2f}it/sec): ".format(update_frequency), self.optimizer.iterations.numpy())
        for key, metric in self.avg_metrics.items():
            tf.summary.scalar(key, metric.result(), step=self.optimizer.iterations)
            print('  {}:     {:.5f}'.format(key, metric.result()))
            metric.reset_states()
        tf.summary.flush()

    def update_metrics(self, key, value):
        if key not in self.avg_metrics:
            self.avg_metrics[key] = tf.keras.metrics.Mean(name=key, dtype=tf.float32)
        self.avg_metrics[key].update_state(value)
