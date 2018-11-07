#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import matplotlib.pyplot as plt

n_episodes = 3000
learning_rate = 0.0005
gamma = 0.99

env = gym.make('CartPole-v0')
state = env.reset()[None, :]

def plot_running_avg(all_scores):
    n = len(all_scores)
    running_avg = np.empty(n)
    for t in range(n):
        running_avg[t] = all_scores[max(0, t-100): t+1].mean()
    plt.plot(running_avg)
    plt.show()

def policy(state, w):
    z = state.dot(w)
    exp = np.exp(z)
    return exp / np.sum(exp)

def softmax_grad(softmax):
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

def run():
    w = np.random.rand(4, 2)
    all_scores = np.empty(n_episodes)
    for e in range(n_episodes):
        state = env.reset()[None, :]
        grads = []
        rewards = []
        score = 0
        done = False

        while not done:
            probs = policy(state, w)
            action = np.random.choice(env.action_space.n, p=probs[0])
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[None, :]

            dsoftmax = softmax_grad(probs)[action, :]
            dlog = dsoftmax / probs[0, action]
            grad = state.T.dot(dlog[None, :])

            grads.append(grad)
            rewards.append(reward)

            score += reward

            state = next_state
        
        for i in range(len(grads)):
            w += learning_rate * grads[i] * sum([r * (gamma ** t) for t, r in enumerate(rewards[i:])])
        all_scores[e] = score
        if e > 0 and e % 100 == 0:
            print("Episode: {}, avg reward (last 100): {}".format(e, all_scores[max(0, e-100):(e+1)].mean()))
    plot_running_avg(all_scores)


if __name__ == "__main__":
    run()
