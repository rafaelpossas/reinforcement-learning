
import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
  sys.path.append("../")

from collections import defaultdict
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting


env = WindyGridworldEnv()


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn

def sarsa_lambda(env, num_episodes, discount_factor=0.99, alpha=0.1, epsilon=0.1,lambd=1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    all_actions = range(env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 10 == 0:
            print("\rEpisode {}/{}.\n".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        observation = env.reset()
        action = np.random.choice(all_actions, p=policy(observation))
        E = defaultdict(lambda: np.zeros(env.action_space.n))

        for t in itertools.count():

            next_obs, reward, done, _ = env.step(action)
            next_action = np.random.choice(all_actions, p=policy(next_obs))

            td_target = reward + discount_factor * Q[next_obs][next_action]
            delta = (td_target - Q[observation][action])
            E[observation][action] += 1

            for s in Q:
                for a in all_actions:
                    Q[s][a] += alpha * delta * E[s][a]
                    E[s][a] = discount_factor * lambd * E[s][a]
            if done:
                break

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            observation = next_obs
            action = next_action

    return Q, stats
def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    all_actions = range(env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 10 == 0:
            print("\rEpisode {}/{}.\n".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        observation = env.reset()
        action = np.random.choice(all_actions, p=policy(observation))

        for t in itertools.count():

            next_obs, reward, done, _ = env.step(action)
            next_action = np.random.choice(all_actions, p=policy(next_obs))

            td_target = reward + discount_factor * Q[next_obs][next_action]
            Q[observation][action] += alpha * (td_target - Q[observation][action])

            if done:
                break

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            observation = next_obs
            action = next_action

    return Q, stats
# Q_true, stats_true = sarsa(env,  200)
Q, stats = sarsa_lambda(env, 200)
# currentStateValues = np.asarray([Q[state][action] for state in Q for action in actions])
# error[nInd][alphInd] += np.sqrt(np.sum(np.power(currentStateValues - realStateValues, 2)) / len(states))
plotting.plot_episode_stats(stats)