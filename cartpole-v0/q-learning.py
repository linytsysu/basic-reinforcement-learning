#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import matplotlib.pyplot as plt

n_episodes = 3000

gamma = 0.99
min_alpha = 0.2
min_epsilon = 1e-5
ada_divisor = 25

env = gym.make('CartPole-v0')
q_table = np.random.uniform(low=-1, high=1, size=(4 ** 4, env.action_space.n))

def plot_running_avg(all_scores):
    n = len(all_scores)
    running_avg = np.empty(n)
    for t in range(n):
        running_avg[t] = all_scores[max(0, t-100): t+1].mean()
    plt.plot(running_avg)
    plt.show()

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1: -1]

def discretize(observation):
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [np.digitize(cart_pos, bins=bins(-2.4, 2.4, 4)),
                np.digitize(cart_v, bins=bins(-3.0, 3.0, 4)),
                np.digitize(pole_angle, bins=bins(-0.5, 0.5, 4)),
                np.digitize(pole_v, bins=bins(-2.0, 2.0, 4))]
    return sum([x * (4 ** i) for i, x in enumerate(digitized)])

def get_epsilon(t):
    return 0.5 * (0.99 ** t)
    # return max(min_epsilon, min(1, 1.0 - np.log10((t + 1) / ada_divisor)))

def get_alpha(t):
    return max(min_alpha, min(1.0, 1.0 - np.log10((t + 1) / ada_divisor)))

def update_q_table(state, action, reward, next_state, alpha):
    q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])


def choose_action(state, epsilon):
    return env.action_space.sample() if (np.random.uniform(0, 1) <= epsilon) else np.argmax(q_table[state])

def run():
    all_scores = np.empty(n_episodes)
    for e in range(n_episodes):
        observation = env.reset()
        state = discretize(observation)
        alpha = get_alpha(e)
        epsilon = get_epsilon(e)
        done = False
        score = 0

        while not done:
            action = choose_action(state, epsilon)
            observation, reward, done, _ = env.step(action)
            next_state = discretize(observation)
            if done:
                reward -= 200
            update_q_table(state, action, reward, next_state, alpha)
            state = next_state
            score += 1
        all_scores[e] = score
        if e > 0 and e % 100 == 0:
            print("Episode: {}, avg reward (last 100): {}".format(e, all_scores[max(0, e-100):(e+1)].mean()))
    plot_running_avg(all_scores)


if __name__ == "__main__":
    run()
