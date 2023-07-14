import numpy as np
from model import Model
from memory import Memory, Transition
import time
import datetime
import psutil
from nn_without_frameworks import numpy_nn as nn


class Agent:
    def __init__(self, env, **config):

        self.config = config
        self.env = env
        self.epsilon = self.config["epsilon"]
        self.min_epsilon = self.config["min_epsilon"]
        self.decay_rate = self.config["epsilon_decay_rate"]
        self.n_actions = self.config["n_actions"]
        self.n_states = self.config["n_states"]
        self.max_steps = self.config["max_steps"]
        self.max_episodes = self.config["max_episodes"]
        self.batch_size = self.config["batch_size"]
        self.gamma = self.config["gamma"]
        self.memory = Memory(self.config["memory_size"])

        self.eval_model = Model(self.n_states, self.n_actions)
        self.target_model = Model(self.n_states, self.n_actions)
        self.target_model.set_weights(self.eval_model.parameters)

        self.loss_fn = nn.losses.MSE()
        self.optimizer = nn.optims.Adam(self.eval_model.parameters, lr=self.config["lr"])

        self.to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

    def choose_action(self, state):

        exp = np.random.rand()
        if self.epsilon > exp:
            return np.random.randint(self.n_actions)

        else:
            state = np.expand_dims(state, axis=0)
            return np.argmax(self.eval_model(state))

    def update_train_model(self):
        self.target_model.set_weights(self.eval_model.parameters)
        # self.target_model.eval()

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0  # as no loss
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)

        q_next = self.target_model(next_states).max(axis=-1)

        q_evals = self.eval_model(states)
        q_eval = q_evals[np.arange(self.batch_size), actions]
        q_target = rewards + self.gamma * q_next * (1 - dones)
        dqn_loss = self.loss_fn(q_eval, q_target)

        grads = np.zeros_like(q_evals)
        grads[np.arange(self.batch_size), actions] = dqn_loss.delta
        dqn_loss.delta = grads

        self.eval_model.backward(dqn_loss)
        self.optimizer.apply()

        return dqn_loss.value

    def run(self):

        total_global_running_reward = []
        global_running_reward = 0
        for episode in range(1, 1 + self.max_episodes):
            start_time = time.time()
            state = self.env.reset()
            episode_reward = 0
            for step in range(1, 1 + self.max_steps):
                action = self.choose_action(state)
                next_state, reward, done, _, = self.env.step(action)
                episode_reward += reward
                self.store(state, reward, done, action, next_state)
                dqn_loss = self.train()
                if done:
                    break
                state = next_state

                if (episode * step) % self.config["hard_update_period"] == 0:
                    self.update_train_model()

            self.epsilon = self.epsilon - self.decay_rate if self.epsilon > self.min_epsilon + self.decay_rate else self.min_epsilon

            if episode == 1:
                global_running_reward = episode_reward
            else:
                global_running_reward = 0.99 * global_running_reward + 0.01 * episode_reward

            total_global_running_reward.append(global_running_reward)
            ram = psutil.virtual_memory()
            if episode % self.config["print_interval"] == 0:
                print(f"EP:{episode}| "
                      f"DQN_loss:{dqn_loss:.2f}| "
                      f"EP_reward:{episode_reward}| "
                      f"EP_running_reward:{global_running_reward:.3f}| "
                      f"Epsilon:{self.epsilon:.2f}| "
                      f"Memory size:{len(self.memory)}| "
                      f"EP_Duration:{time.time()-start_time:.3f}| "
                      f"{self.to_gb(ram.used):.1f}/{self.to_gb(ram.total):.1f} GB RAM| "
                      f'Time:{datetime.datetime.now().strftime("%H:%M:%S")}')
                self.save_weights()

        return total_global_running_reward

    def store(self, state, reward, done, action, next_state):
        self.memory.add(state, reward, done, action, next_state)

    def unpack_batch(self, batch):

        batch = Transition(*zip(*batch))

        states = np.concatenate(batch.state).reshape(self.batch_size, self.n_states)
        actions = np.stack(batch.action)
        rewards = np.stack(batch.reward)
        next_states = np.concatenate(batch.next_state).reshape(self.batch_size, self.n_states)
        dones = np.stack(batch.done)
        return states, actions, rewards, next_states, dones

    def save_weights(self):
        nn.save(self.eval_model.parameters, "weights.pkl")

    def load_weights(self):
        params = nn.load("weights.pkl")
        self.eval_model.set_weights(params)

    def set_to_eval_mode(self):
        # self.eval_model.eval()
        pass