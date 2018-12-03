#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import tensorflow as tf

def plot_running_avg(all_scores):
    n = len(all_scores)
    running_avg = np.empty(n)
    for t in range(n):
        running_avg[t] = all_scores[max(0, t-100): t+1].mean()
    plt.plot(running_avg)
    plt.show()


gamma = 0.9
initial_epsilon = 0.5
final_epsilon = 0.01
replay_size = 10000
batch_size = 32


class DQNSolver:
    def __init__(self, env):
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = initial_epsilon
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_q_network()
        self.create_training_method()

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_q_network(self):
        W1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])

        self.state_input = tf.placeholder("float", [None, self.state_dim])
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        self.q_value = tf.matmul(h_layer, W2) + b2

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])
        q_action = tf.reduce_sum(tf.multiply(self.q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def preceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > replay_size:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > batch_size:
            self.train_q_network()

    def train_q_network(self):
        self.time_step += 1
        minibatch = random.sample(self.replay_buffer, batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        y_batch = []
        q_value_batch = self.q_value.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, batch_size):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + gamma * np.max(q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch,
            })

    def egreedy_action(self, state):
        q_value = self.q_value.eval(feed_dict={
                        self.state_input: [state]
                    })[0]
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(q_value)
        self.epsilon -= (initial_epsilon - final_epsilon) / 10000

    def action(self, state):
        return np.argmax(self.q_value.eval(feed_dict={
                    self.state_input: [state]
                })[0])

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

episode = 10000
step = 300
test = 10


def run():
    env = gym.make('CartPole-v0')
    agent = DQNSolver(env)

    for e in range(episode):
        state = env.reset()

        for s in range(step):
            action = agent.egreedy_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.preceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        if e % 100 == 0:
            total_reward = 0
            for i in range(test):
                state = env.reset()
                for j in range(step):
                    action = agent.action(state)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            avg_reward = total_reward / test
            print(e, avg_reward)


if __name__ == "__main__":
    run()
