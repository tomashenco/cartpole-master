import gym
import numpy as np
import time

from agent import Agent


class CartPole:
    def __init__(self):
        self.replay_batch_size = 500
        self.training_episodes = 500
        self.show_episodes = 3
        self.env = gym.make("CartPole-v1")
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = Agent(self.state_size, self.action_size)

    def train(self):
        for episode in range(self.training_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            done = False
            score = 0
            while not done:
                action = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                score += 1

            print("Episode #{} Score: {}".format(episode, score))
            self.agent.replay(self.replay_batch_size)

    def show(self):
        self.agent = Agent(self.state_size, self.action_size)
        self.agent.load_model()

        for index_episode in range(self.show_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            done = False
            score = 0
            while not done:
                self.env.render()
                time.sleep(0.01)
                action = self.agent.act(state)
                state, reward, done, info = self.env.step(action)
                state = np.reshape(state, [1, self.state_size])
                score += 1

            print("The score was: {}".format(score))
