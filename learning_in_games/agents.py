import numpy as np
from dataclasses import dataclass
from typing import Union
from .games import GameConfig
import math


@dataclass
class QAgentConfig:
    alpha: Union[np.ndarray, float]  # learning rate
    gamma: Union[np.ndarray, float]  # discount gact
    qinit: Union[np.ndarray]  # initial q-values


def initialize_q_table(q_input, gameConfig: GameConfig, qmin=0, qmax=1):
    if type(q_input) == np.ndarray:
        if q_input.shape == (gameConfig.n_agents, gameConfig.n_states, gameConfig.n_actions):
            q_table = q_input
        else:
            q_table = q_input.T * np.ones((gameConfig.n_agents, gameConfig.n_states, gameConfig.n_actions))
    elif q_input == "UNIFORM":
        q_table = (qmax - qmin) * np.random.random_sample(size=(gameConfig.n_agents, gameConfig.n_states, gameConfig.n_actions)) + qmin
    elif q_input == "ALIGNED":
        if gameConfig.n_actions == 3:
            q_table = np.array([[-1, -2, -2], [-2, -1, -2], [-2, -2, -1]]).T * np.ones((gameConfig.n_agents, gameConfig.n_states, gameConfig.n_actions))
        elif gameConfig.n_actions == 2:
            q_table = np.array([[-1, -2], [-2, -1]]).T * np.ones((gameConfig.n_agents, gameConfig.n_states, gameConfig.n_actions))
    elif q_input == "MISALIGNED":
        if gameConfig.n_actions == 3:
            q_table = np.array([[-2, -1, -2], [-2, -2, -1], [-1, -2, -2]]).T * np.ones((gameConfig.n_agents, gameConfig.n_states, gameConfig.n_actions))
        elif gameConfig.n_actions == 2:
            q_table = np.array([[-2, -1], [-1, -2]]).T * np.ones((gameConfig.n_agents, gameConfig.n_states, gameConfig.n_actions))
    return q_table


def initialize_learning_rates(agentConfig: QAgentConfig, gameConfig: GameConfig):
    if agentConfig.alpha == "UNIFORM":
        agentConfig.alpha = np.random.random_sample(size=gameConfig.n_agents)
        return agentConfig
    elif type(agentConfig.alpha) == np.ndarray:
        return agentConfig
    elif type(agentConfig.alpha) == float:
        return agentConfig


def initialize_exploration_rates(gameConfig, agentConfig):  # default mask 1 leads to no change
    # if the policy is epsilon greedy
    if agentConfig.epsilon == "DECAYED":
        exp_start = 1
        exp_end = 0
        exp_decay = gameConfig.n_iter / 8
    else:
        exp_start = agentConfig.epsilon
        exp_end = agentConfig.epsilon
        exp_decay = 1

    return exp_start, exp_end, exp_decay


def update_exploration_rates(t, n_iter, exp_start=1, exp_end=0):
    exp_decay = n_iter / 8
    return exp_end + (exp_start - exp_end) * math.exp(-1. * t / exp_decay)


def bellman_update_q_table(Q, S, A, R, S_, agentConfig: QAgentConfig):
    """
    Performs a one-step update using the bellman update equation for Q-learning.
    :param agentConfig:
    :param Q: np.ndarray Q-table indexed by (agents, states, actions)
    :param S: np.ndarray States indexed by (agents)
    :param A: np.ndarray Actions indexed by (agents)
    :param R: np.ndarray Rewards indexed by (agents)
    :param S_: np.ndarray Next States indexed by (agents)
    :return: np.ndarray Q-table indexed by (agents, states, actions)
    """
    ind = np.arange(len(S))
    all_belief_updates = agentConfig.alpha * (R + agentConfig.gamma * Q[ind, S_].max(axis=1) - Q[ind, S, A])
    Q[ind, S, A] = Q[ind, S, A] + all_belief_updates
    return Q, np.abs(all_belief_updates).sum()


@dataclass
class EpsilonGreedyConfig(QAgentConfig):
    epsilon: Union[float, str]


def e_greedy_select_action(Q, S, agentConfig: EpsilonGreedyConfig):
    """
    Select actions based on an epsilon greedy policy. Epsilon determines the probability with which
    an action is selected at random. Otherwise, the action is selected as the argmax of the state.
    During the argmax operation and given a tie the argmax operator selects the first occurrence.
    :param agentConfig:
    :param Q: np.ndarray Q-table indexed by (agents, states, actions)
    :param S: np.ndarray States indexed by (agents)
    :return: np.ndarray Actions indexed by (agents)
    """
    indices = np.arange(len(S))
    rand = np.random.random_sample(size=len(S))
    randA = np.random.randint(len(Q[0, 0, :]), size=len(S))
    A = np.where(rand >= agentConfig.epsilon,
                 np.argmax(Q[indices, S, :], axis=1),
                 randA)
    return A


def e_greedy_select_action_randomized_argmax(Q, S, agentConfig: EpsilonGreedyConfig):
    """
    Select actions based on an epsilon greedy policy. Epsilon determines the probability with which
    an action is selected at random. Otherwise, the action is selected as the argmax of the state.
    Additionally, this function takes care of ties during the argmax operation, and selects at random
    in the event of a tie. Standard numpy argmax selects the first element of a tie.
    :param agentConfig:
    :param Q: np.ndarray Q-table indexed by (agents, states, actions)
    :param S: np.ndarray States indexed by (agents)
    :return: np.ndarray Actions indexed by (agents)
    """
    n_agents = len(S)
    indices = np.arange(n_agents)
    rand = np.random.random_sample(size=n_agents)

    randA = np.random.randint(len(Q[0, 0, :]), size=n_agents)

    rng = np.random.default_rng()
    argmax_actions = rng.choice(np.where(
        [np.isclose(vals.max(axis=0), vals) for vals in Q[indices, S, :]]
    ))
    A = np.where(rand >= agentConfig.epsilon, argmax_actions, randA)
    return A


@dataclass
class BoltzmannAgentConfig(QAgentConfig):
    temperature: Union[float, str]


def boltzmann_select_action(Q, S, agentConfig: BoltzmannAgentConfig):
    """
    Selects actions by drawing from a distribution determined by a softmax operator on the q-values.
    The temperature parameter approaching 0 leads to a distribution which approaches an argmax, while
    approaching infinity leads to the uniform random distribution.
    WARNING: low temperature values can easily lead to NaNs or infs.
    :param agentConfig:
    :param Q: np.ndarray Q-table indexed by (agents, states, actions)
    :param S: np.ndarray States indexed by (agents)
    :return: np.ndarray Actions indexed by (agents)
    """
    indices = np.arange(len(S)).astype(int)
    rng = np.random.default_rng()
    values_exponential = np.exp(Q[indices, S] / agentConfig.temperature)
    denominator = np.sum(values_exponential, axis=1)
    probabilities = np.divide(values_exponential.T, denominator).T
    actions = np.array(
        [rng.choice(
            a=Q.shape[-1],  # n_actions
            p=probabilities[i]
        ) for i in range(Q.shape[0])]  # n_agents
    )
    return actions


def follow_the_regularized_leader_select_action(Q, S, agentConfig: BoltzmannAgentConfig):
    """
    Selects actions based on the follow the regularized leader algorithms which apply an argmax operation
    to q-values which have been regularized with a softmax operation.
    :param agentConfig:
    :param Q: np.ndarray Q-table indexed by (agents, states, actions)
    :param S: np.ndarray States indexed by (agents)
    :return: np.ndarray Actions indexed by (agents)
    """
    indices = np.arange(len(S))
    values_exponential = np.exp(Q[indices, S] / agentConfig.temperature)
    denominator = np.sum(values_exponential, axis=1)
    regularization = np.divide(values_exponential.T, denominator).T
    x = Q[indices, S] - regularization
    A = np.argmax(x, axis=1)
    return A


def average_rolled_q_tables(Q, neighborhood):
    """
    Creates an array where successive rows are "rolled" versions of Q, along the axis of agents.
    This is created and used for experiments where agents average their q-values with each others for
    different sizes of neighborhoods.
    :param Q: np.ndarray Q-table indexed by (agents, states, actions)
    :param neighborhood:
    :return: np.ndarray Q-table with averaged entries indexed by (agents, states, actions)
    """
    tmp_Q = np.array([np.roll(Q, shift=i, axis=0) for i in range(neighborhood)])
    Q = np.mean(tmp_Q, axis=0)
    return Q
