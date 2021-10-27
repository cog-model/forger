import os
import random
import timeit
from collections import deque

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.tf_util import huber_loss, take_vector_elements


class Agent:
    def __init__(self, config, replay_buffer, build_model, obs_space, act_space,
                 dtype_dict=None, log_freq=100):
        # global
        self.frames_to_update = config['frames_to_update']
        self.save_dir = config['save_dir']
        self.update_quantity = config['update_quantity']
        self.update_target_net_mod = config['update_target_net_mod']
        self.batch_size = config['batch_size']
        self.margin = np.array(config['margin']).astype('float32')
        self.replay_start_size = config['replay_start_size']
        self.gamma = config['gamma']
        self.learning_rate = config['learning_rate']
        self.reg = config['reg'] if 'reg' in config else 1e-5
        self.n_deque = deque([], maxlen=config['n_step'])

        if 'wandb' in config.keys():
            self.wandb = config['wandb']
        else:
            self.wandb = None

        self.replay_buff = replay_buffer
        self.priorities_store = list()
        if dtype_dict is not None:
            ds = tf.data.Dataset.from_generator(self.sample_generator, output_types=dtype_dict)
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            self.sampler = ds.take
        else:
            self.sampler = self.sample_generator
        self.online_model = build_model('Online_Model', obs_space, act_space, self.reg)
        self.target_model = build_model('Target_Model', obs_space, act_space, self.reg)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self._run_time_deque = deque(maxlen=log_freq)
        self._schedule_dict = dict()
        self._schedule_dict[self.target_update] = self.update_target_net_mod
        self._schedule_dict[self.update_log] = log_freq
        self.avg_metrics = dict()
        self.action_dim = act_space.n
        self.act_space = act_space

    def train(self, env, episodes=200, seeds=None, name="max_model.ckpt", save_mod=50,
              epsilon=0.1, final_epsilon=0.01, eps_decay=0.99, save_window=10):
        scores, counter = [], 0
        max_reward = -np.inf
        window = deque([], maxlen=save_window)
        for e in range(episodes):
            score, counter = self.train_episode(env, seeds, counter, epsilon)
            if self.replay_buff.get_stored_size() > self.replay_start_size:
                epsilon = max(final_epsilon, epsilon * eps_decay)
            scores.append(score)
            window.append(score)
            print("episode: {}  score: {}  counter: {}  epsilon: {}  max: {}"
                  .format(e, score, counter, epsilon, max_reward))
            tf.summary.scalar("reward", score, step=e)
            tf.summary.flush()
            if self.wandb:
                self.wandb.log({"reward": score, "episode": e})
            avg_reward = sum(window) / len(window)
            if avg_reward >= max_reward:
                print("MaxAvg reward moved from {:.2f} to {:.2f} (save model)".format(max_reward,
                                                                                      avg_reward))
                max_reward = avg_reward
                self.save(os.path.join(self.save_dir, name))
            if e % save_mod == 0:
                self.save(os.path.join(self.save_dir, "{}_model.ckpt".format(e)))
        return scores

    def train_episode(self, env, seeds=None, current_step=0, epsilon=0.0):
        counter = 0
        if current_step == 0:
            self.target_update()
        if seeds:
            env.seed(random.choice(seeds))
        done, score, state = False, 0, env.reset()

        # template_action = self.act_space #env.action_space.noop()

        while done is False:
            action = self.choose_act(state, epsilon)

            #if current_task == 'wodden_pickaxe':
            #    action = template_action['nearbyCraft'] = 'wooden_pickaxe'
            #    print('\nwooden pickaxe manully!!!!!!!!!!!!')

            next_state, reward, done, _ = env.step(action)
            score += reward
            self.perceive(to_demo=0, state=state, action=action, reward=reward, next_state=next_state,
                          done=done, demo=False)
            counter += 1
            state = next_state
            if self.replay_buff.get_stored_size() > self.replay_start_size \
                    and counter % self.frames_to_update == 0:
                self.update(self.update_quantity)
        return score,counter

    def test(self, env, name="train/max_model.ckpt", number_of_trials=1, render=False):
        if name:
            self.load(name)

        total_reward = 0
        timestep_record = 0
        for trial_index in range(number_of_trials):
            reward = 0
            done = False
            observation = env.reset()
            rewards_dict = {}
            count = 0
            while not done:
                action = self.choose_act(observation)
                observation, r, done, _ = env.step(action)
                if render:
                    env.render()
                if int(r) not in rewards_dict:
                    rewards_dict[int(r)] = 0
                rewards_dict[int(r)] += 1
                reward += r
                count += 1
            timestep_record +=count
            total_reward += reward
        env.reset()
        return total_reward,timestep_record

    def pre_train(self, steps=150000):
        """
        pre_train phase in ForgER alg.
        :return:
        """
        print('Pre-training ...')
        self.target_update()
        self.update(steps)
        self.save(os.path.join(self.save_dir, "pre_trained_model.ckpt"))
        print('All pre-train finish.')

    def update(self, steps):
        start_time = timeit.default_timer()
        for batch in self.sampler(steps):
            indexes = batch.pop('indexes')
            priorities = self.q_network_update(gamma=self.gamma, **batch)
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
            yield self.replay_buff.sample(self.batch_size)
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
            action_cast = tf.cast(action, tf.int32)

            q_value = take_vector_elements(q_value, action_cast)

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
        # for i, g in enumerate(gradients):
        #     gradients[i] = tf.clip_by_norm(g, 10)
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
                        off_value=self.margin)
        ae = tf.cast(ae, 'float32')
        max_value = tf.reduce_max(q_value + ae, axis=1)
        ae = tf.one_hot(action, self.action_dim)
        j_e = tf.abs(tf.reduce_sum(q_value * ae, axis=1) - max_value)
        j_e = tf.reduce_mean(j_e * weights * demo)
        return j_e

    def add_demo(self, data, expert_data=1, fixed_reward=None):
        all_data = 0
        progress = tqdm(total=self.replay_buff.get_buffer_size())
        for state, action, reward, next_state, done in data.sarsd_iter(1, 400000):
            all_data+=1

            self.perceive(to_demo=1, state=state, action=action, reward=fixed_reward if fixed_reward else reward,
                          next_state=next_state, done=done, demo=int(expert_data))
            progress.update(1)
            if progress.total == all_data:
                break

        print('demo data added to buff')
        progress.close()
        print("***********************")
        print("all data set", all_data)
        print("***********************")

    def perceive(self, **kwargs):
        self.n_deque.append(kwargs)
        if len(self.n_deque) == self.n_deque.maxlen or kwargs['done']:
            while len(self.n_deque) != 0:
                n_state = self.n_deque[-1]['next_state']
                n_done = self.n_deque[-1]['done']
                n_reward = sum([t['reward'] * self.gamma ** i for i, t in enumerate(self.n_deque)])
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
        self.online_model.save_weights(out_dir)

    def load(self, out_dir=None):
        self.online_model.load_weights(out_dir)

    def update_log(self):
        update_frequency = len(self._run_time_deque) / sum(self._run_time_deque)
        print("LearnerEpoch({:.2f}it/sec): ".format(update_frequency), self.optimizer.iterations.numpy())
        c=0
        for key, metric in self.avg_metrics.items():
            c+=1
            if c%10:
                tf.summary.scalar(key, metric.result(), step=self.optimizer.iterations)
                print('  {}:     {:.5f}'.format(key, metric.result()))
                metric.reset_states()
        tf.summary.flush()

    def update_metrics(self, key, value):
        if key not in self.avg_metrics:
            self.avg_metrics[key] = tf.keras.metrics.Mean(name=key, dtype=tf.float32)
        self.avg_metrics[key].update_state(value)
