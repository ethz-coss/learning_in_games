import numpy as np
from dataclasses import dataclass


@dataclass
class GameConfig:
    """
    The minimal parameters to specify games.
    This excludes the payoff functions which are defined separately as functions.
    """
    n_agents: int
    n_actions: int
    n_states: int
    n_iter: int


@dataclass
class RouteConfig(GameConfig):
    cost: float


def braess_augmented_network(actions, config: RouteConfig):
    """
    Network from the Braess Paradox with the added link, and the Nash Equilibrium average travel time is 2,
    but the optimal average travel time is 1.5: and no players take the added link.
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    n_agents = config.n_agents
    n_up = (actions == 0).sum()
    n_down = (actions == 1).sum()
    n_cross = (actions == 2).sum()

    r_0 = 1 + (n_up + n_cross) / n_agents
    r_1 = 1 + (n_down + n_cross) / n_agents
    r_2 = (n_up + n_cross) / n_agents + (n_down + n_cross) / n_agents + config.cost

    T = np.array([-r_0, -r_1, -r_2])
    R = T[actions]
    S = None
    return R, S


def braess_initial_network(actions, config: RouteConfig):
    """
    Network from the Braess Paradox without the added link, and the Nash Equilibrium average travel time is 1.5,
    which is also the optimal average travel time is 1.5 where players split evenly over the two paths (0.5, 0.5).
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    n_agents = config.n_actions
    n_up = (actions == 0).sum()
    n_down = (actions == 1).sum()

    r_0 = 1 + n_up / n_agents
    r_1 = 1 + n_down / n_agents

    T = np.array([-r_0, -r_1])
    R = T[actions]
    return R, T


def two_route_game(actions, config: RouteConfig):
    """
    A two path routing game where the cost parameter can be used to vary the edge costs from one extreme, where the
    network resembles the Pigou network, to another extreme where the Nash Equilibrium corresponds to the optimal
    average travel time.
    :param actions: np.ndarray of Actions indexed by agents
    :param cost:
    :return:
    """
    n_agents = config.n_agents
    n_up = (actions == 0).sum()

    r_0 = n_up / n_agents + config.cost
    r_1 = (1 - n_up / n_agents) + (1 - config.cost)

    T = np.array([-r_0, -r_1])
    R = T[actions]
    return R, T


def pigou(actions, config):
    """
    The Pigou network routing game which has two paths, one with a fixed cost and the other with a variable cost equal
    to the percentage of players that take that path. The classic Pigou game has a fixed cost of 1.
    :param actions: np.ndarray of Actions indexed by agents
    :param cost:
    :return:
    """
    n_agents = config.n_agents
    n_down = (actions == 1).sum()
    pct = n_down / n_agents

    r_0 = config.cost
    r_1 = pct

    T = np.array([-r_0, -r_1])
    R = T[actions]
    return R, T


def pigou3(actions, config: GameConfig):
    """
    A version of a Pigou network with three paths, two fixed cost paths and one variable cost path.
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    n_agents = config.n_agents
    n_up = (actions == 0).sum()

    r_0 = n_up / n_agents
    r_1 = 1
    r_2 = 1

    T = np.array([-r_0, -r_1, -r_2])
    R = T[actions]
    return R, T


@dataclass
class MinorityConfig(GameConfig):
    threshold: float


def minority_game(actions, config: MinorityConfig):
    """
    A minority game where the minority group is determined by a threshold
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    n_agents = config.n_agents
    n_up = (actions == 0).sum()

    if n_agents * config.threshold >= n_up:  # up is minority
        r_0 = 1
        r_1 = 0
    else:
        r_0 = 0
        r_1 = 1

    T = np.array([r_0, r_1])
    R = T[actions]
    return R, T


def minority_game_2(actions, config: GameConfig):
    """
    A minority game variant.
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    n_agents = config.n_agents
    n_a = (actions == 0).sum()
    fraction_a = n_a / n_agents
    fraction_b = 1 - fraction_a

    r_a = 1 - 2 * fraction_a
    r_b = 1 - 2 * fraction_b

    T = np.array([r_a, r_b])
    R = T[actions]
    S = None  # stateless
    return R, S


def el_farol_bar(actions, config: MinorityConfig):
    """
    The El Farol Bar game where all those that stay home get a payoff of 1, while
    all those that go to the bar get payoffs better than 1 only if the fraction (pct)
    of players that go to the bar is below a threshold.
    :param actions: np.ndarray of Actions indexed by agents
    :return:
    """
    n_agents = len(actions)
    n_bar = (actions == 1).sum()
    pct = n_bar / n_agents

    r_0 = 1
    r_1 = 2 - 4 * pct if (pct > config.threshold) else 4 * pct - 2

    T = np.array([-r_0, -r_1])
    R = T[actions]
    return R, T


def duopoly(actions, config: GameConfig):
    """
    A duopoly pricing game, intended to be played as a turn taking game, but can also be played as a simultaneous game.
    The state of the game for each player is the previous action of the other player.
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    a1 = actions[0]
    a2 = actions[1]

    p1 = a1 / config.n_actions
    p2 = a2 / config.n_actions

    if p1 < p2:
        r1 = (1 - p1) * p1
        r2 = 0
    elif p1 == p2:
        r1 = 0.5 * (1 - p1)
        r2 = r1
    elif p1 > p2:
        r1 = 0
        r2 = (1 - p2) * p2

    R = np.array([r1, r2])
    S = np.array([a2, a1])

    return R, S


@dataclass
class PrisonersDilemmaConfig(GameConfig):
    reward_payoff: float
    suckers_payoff: float


def prisoners_dilemma(actions, config: PrisonersDilemmaConfig):
    """
    The Prisoner's Dilemma game parameterized by the reward and suckers payoffs.
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    a1 = actions[0]
    a2 = actions[1]

    if a1 == 0 and a2 == 0:
        r1 = config.reward_payoff
        r2 = config.reward_payoff
    elif a1 == 0 and a2 == 1:
        r1 = -config.suckers_payoff
        r2 = 1
    elif a1 == 1 and a2 == 0:
        r1 = 1
        r2 = -config.suckers_payoff
    elif a1 == 1 and a2 == 1:
        r1 = 0
        r2 = 0

    state = a1 + a2

    R = np.array([r1, r2])
    S = np.array([state, state])

    return R, S


@dataclass()
class PopulationConfig(GameConfig):
    V: float
    K: float
    exponent: float
    cost: float


def population_game(actions, config: PopulationConfig):
    """
    A population game as found in the paper 'Catastrophe by Design in Population Games: A Mechanism to Destabilize
    Inefficient Locked-in Technologies' (https://doi.org/10.1145/3583782).
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    n_players = len(actions)
    fraction_weak = (actions == 0).sum() / n_players
    fraction_strong = (actions == 1).sum() / n_players

    utility_weak = config.V * (fraction_weak * config.K) ** (config.exponent - 1) - config.cost
    utility_strong = config.V * (fraction_strong * config.K) ** (config.exponent - 1)  # no added cost

    T = [utility_weak, utility_strong]
    R = np.array([T[a] for a in actions])
    return R, T


@dataclass
class PublicGoodsConfig(GameConfig):
    multiplier: float
    beta: float


def public_goods_game(actions, config: PublicGoodsConfig):
    """
    A public goods game parametrized by the multiplier, and Beta, a parameter which controls the slope of
    the marginal contributions of each action.
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    norm_A = actions / config.n_actions
    pot = config.multiplier * np.power(norm_A, config.beta).sum()
    R = 1 - norm_A + pot
    return R
