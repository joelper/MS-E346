from typing import Tuple
from modules.MDP import MDP, Policy, V
from modules.RL_interface import RL_interface, action_sampler
import random
import numpy as np


def first_visit_mc(policy: Policy, mdp: MDP, num_epi: int, num_steps: int) -> V:
    # follows the first visit MC algorithm outlined in Sutton's RL book
    v = {}
    returns = {}
    gamma = mdp.gamma
    for s in mdp.States:
        v[s] = 0
        returns[s] = []

    for i in range(num_epi):
        # generate an episode
        s_list, a_list, r_list = generate_path(policy, mdp, num_steps)
        # initialize the episode return
        G = 0
        for j in range(num_steps -1, 0, -1):
            G = gamma *G + r_list[j]
            if s_list[j] not in s_list[:j]:
                returns[s_list[j]].append(G)

    for s in mdp.States:
        v[s] = np.mean(returns[s])

    return v


def generate_path(policy: Policy, mdp: MDP, num_steps: int) -> Tuple[list, list, list]:
    # generate a sample path that follows the provided policy
    # the function returns: S_0, A_0, R_1, ... , S_(T-1), A_(T-1), R_T
    s_list = []
    a_list = []
    r_list = []

    s0 = random.sample(mdp.States, 1)
    s_list.append(s0)
    a0 = action_sampler(policy, s0)
    a_list.append(a0)
    for i in range(num_steps - 1):
        sp, r = RL_interface(mdp, s_list[-1], a_list[-1])
        a = action_sampler(policy, sp)
        s_list.append(sp)
        a_list.append(a)
        r_list.append(r)

    # sample the last reward
    _, r = RL_interface(mdp, s_list[-1], a_list[-1])
    r_list.append(r)

    return s_list, a_list, r_list


def TD_0(policy: Policy, mdp: MDP, alpha: float, num_epi: int, num_steps: int) -> V:
    # follows the TD(0) algorithm as it is outlined in the Sutton RL book
    v = {}
    gamma = mdp.gamma
    for s in mdp.States:
        v[s] = 0

    for i in range(num_epi):
        s = random(mdp.States)

        for j in range(num_steps):
            a = action_sampler(policy, s)
            sp, r = RL_interface(mdp, s, a)
            v[s] += alpha * (r + gamma * v[sp] - v[s])
            s = sp

    return v