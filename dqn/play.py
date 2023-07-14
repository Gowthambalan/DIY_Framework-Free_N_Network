import torch
import gym


class Play:
    def __init__(self, env, agent, max_episode=4):
        self.env = env
        self.env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True, force=True)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.load_weights()
        self.agent.set_to_eval_mode()
        self.agent.epsilon = self.agent.min_epsilon

    def evaluate(self):

        for _ in range(self.max_episode):
            s = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.agent.choose_action(s)
                s_, r, done, _ = self.env.step(action)
                episode_reward += r
                s = s_
                self.env.render()
            print(f"reward:{episode_reward:3.3f}")
        self.env.close()
