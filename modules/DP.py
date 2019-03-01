from typing import Dict
from modules.MDP import MDP, Policy, V, Q
from modules.state_action_vars import S, A
import numpy as np


def policy_eval(mdp: MDP, policy: Policy, tol: float) -> V:
    # implementation of policy evaluation
    vf = {s: 0. for s in mdp.States}

    while True:
        new_vf = {}
        for s in mdp.States:
            new_vf[s] = 0.
            for a in policy[s]:
                new_vf[s] += policy[s][a] * mdp.R[s][a]
                for sp in mdp.P[s][a]:
                    new_vf[s] += policy[s][a] * mdp.gamma * mdp.P[s][a][sp] * vf[sp]
        diff = 0
        for s in vf:
            diff += np.abs(vf[s] - new_vf[s])

        vf = new_vf.copy()
        if diff < tol:
            break

    return vf


def policy_iter(mdp: MDP, policy: Policy, tol: float) -> Policy:
    same = False
    while not same:
        new_policy = {}
        v = policy_eval(mdp, policy, tol)

        for s in mdp.States:
            best_value = -1e10
            best_a = None
            new_policy[s] = {}

            for a in mdp.P[s]:
                # reinitialize the new policy
                new_policy[s][a] = 0.0
                value = mdp.R[s][a]

                for sp in mdp.P[s][a]:
                    value += mdp.gamma * mdp.P[s][a][sp] * v[sp]

                if value > best_value:
                    best_value = value
                    best_a = a
            # make the policy deterministic
            new_policy[s][best_a] = 1.0

        diff = 0.
        for s in new_policy:
            for a in new_policy[s]:
                diff += np.abs(new_policy[s][a] - policy[s][a])

        if diff < tol:
            same = True

        policy = new_policy.copy()

    return policy


def value_iter(mdp: MDP, tol: float) -> V:
    # implementation of value iteration, the code is very similar to that of policy iteration
    # the difference is what kind of information we store

    # initialize the value function
    v = {s: 0. for s in mdp.States}

    while True:
        # initialize new dictionary to store the values for this iteration
        new_v = {}
        for s in mdp.States:
            # variable to store the best value when looping over the actions
            best_value = -1000000
            for a in mdp.P[s]:
                # variable for storing the value for action a
                value = mdp.R[s][a]
                for sp in mdp.P[s][a]:
                    value += mdp.gamma * mdp.P[s][a][sp] * v[sp]
                if value > best_value:
                    best_value = value
            # store the best value
            new_v[s] = best_value

        diff = 0
        for s in v:
            diff += np.abs(v[s] - new_v[s])
        # copy the value function
        v = new_v.copy()
        if diff < tol:
            break

    return v


def convert_reward(R: Dict[S, Dict[A, Dict[S, float]]], P: Dict[S, Dict[A, Dict[S, float]]]) -> Dict[S, Dict[A, float]]:
    # function to convert an mdp using r(s, s', a) to R(s,a)
    new_r = {}
    for s in R:
        new_r[s] = {}
        for a in R[s]:
            new_r[s][a] = 0
            assert(type(R[s][a]) == dict)
            for sp in P[s][a]:
                new_r[s][a] += R[s][a][sp] * P[s][a][sp]

    return new_r
