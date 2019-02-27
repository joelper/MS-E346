from typing import Tuple
from modules.MDP import MDP, Policy
from modules.state_action_vars import S, A
import numpy as np


def RL_interface(mdp: MDP, s: S, a: A) -> Tuple[S, float]:
    # interface that takes in a state 's' and an action 'a', and returns a new state 'sp' and an observed reward 'r'
    sp = sp_sampler(mdp, s, a)
    # check how the reward is defined
    if type(mdp.R[s][a]) == float:
        r = mdp.R[s][a]
    else:
        r = mdp.R[s][a][sp]

    return sp, r


def sp_sampler(mdp: MDP, s: S, a: A) -> S:
    # function that takes in an MDP, a state and an action and samples a new state sp from that distribution
    p_cum = 0
    prob = np.random.rand()
    for sp in mdp.P[s][a].keys():
        p_cum += mdp.P[s][a][sp]
        if prob <= p_cum:
            return sp

    return sp


def action_sampler(policy: Policy, s: S) -> A:
    # function that takes in a policy and a state and samples an action according to this policy
    p_cum = 0
    prob = np.random.rand()
    for a in policy[s].keys():
        p_cum += policy[s][a]
        if prob <= p_cum:
            return a

    return a