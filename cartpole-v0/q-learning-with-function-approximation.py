#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

n_episodes = 3000
gamma = 0.99

env = gym.make('CartPole-v0')

def plot_running_avg(all_scores):
    n = len(all_scores)
    running_avg = np.empty(n)
    for t in range(n):
        running_avg[t] = all_scores[max(0, t-100): t+1].mean()
    plt.plot(running_avg)
    plt.show()

class SGDRegressor:
    def __init__(self, n_feature):
        self.w = np.random.randn(n_feature) / np.sqrt(n_feature)
        self.lr = 0.1

    def partial_fit(self, X, y):
        self.w += self.lr * (y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)

class FeatureTransformer:
    def __init__(self, env):
        # observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        observation_examples = np.random.random((20000, 4)) * 2 - 1
        self.scaler = StandardScaler()
        self.scaler.fit(observation_examples)
        self.featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.05, n_components=1000)),
            ("rbf2", RBFSampler(gamma=1.0, n_components=1000)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=1000)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=1000))
            ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

    def transform(self, observation):
        scaled = self.scaler.transform([observation])
        return self.featurizer.transform(scaled)[0]

class Estimator:
    def __init__(self, env):
        self.feature_transformer = FeatureTransformer(env)
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(4000)
            self.models.append(model)

    def predict(self, s):
        features = self.feature_transformer.transform(s)
        return np.array([m.predict(np.array([features])) for m in self.models])

    def update(self, s, a, y):
        features = self.feature_transformer.transform(s)
        self.models[a].partial_fit(np.array([features]), [y])

def get_epsilon(t):
    return 1.0 / np.sqrt(t + 1)

def choose_action(estimator, state, epsilon):
    return env.action_space.sample() if (np.random.uniform(0, 1) <= epsilon) else np.argmax(estimator.predict(state))

def run():
    estimator = Estimator(env)
    all_scores = np.empty(n_episodes)
    for e in range(n_episodes):
        state = env.reset()
        epsilon = get_epsilon(e)
        done = False
        score = 0
        while not done:
            action = choose_action(estimator, state, epsilon)
            next_state, reward, done, _ = env.step(action)
            if done:
                reward -= 200
            q_values_next = estimator.predict(next_state)
            td_target = reward + gamma * np.max(q_values_next)
            estimator.update(state, action, td_target)
            state = next_state
            score += 1
        all_scores[e] = score
        if e > 0 and e % 100 == 0:
            print("Episode: {}, avg reward (last 100): {}".format(e, all_scores[max(0, e-100):(e+1)].mean()))
    plot_running_avg(all_scores)


if __name__ == '__main__':
    run()

