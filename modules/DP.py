from modules.MDP import MDP, Policy, V, Q
from modules.state_action_vars import S, A


def policy_eval(mdp: MDP, policy: Policy, n_iter: int) -> V:
    # implementation of policy evaluation
    vf = {s: 0. for s in mdp.States}
    for i in range(n_iter):
        new_vf = {}
        for s in mdp.States:
            new_vf[s] = 0
            for a in policy[s]:
                new_vf[s] += policy[s][a]*mdp.R[s][a]
                for sp in mdp.P[s][a]:
                    new_vf[s] += policy[s][a]*mdp.gamma*mdp.P[s][a][sp]*vf[sp]
        vf = new_vf
    return vf


def policy_iter(mdp: MDP, policy: Policy, n_iter: int) -> Policy:
    for i in range(n_iter):
        new_policy = policy
        v = policy_eval(mdp, policy, n_iter)
        for s in mdp.States:
            best_value = -1000000
            best_a: A
            for a in mdp.P[s]:
                # reinitialize the new policy
                new_policy[s][a] = 0.
                value = mdp.R[s][a]
                for sp in mdp.P[s][a]:
                    value += mdp.gamma * mdp.P[s][a][sp] * v[sp]
                if value > best_value:
                    best_value = value
                    best_a = a
            # make the policy deterministic
            new_policy[s][best_a] = 1.0

        policy = new_policy
    return policy


def value_iter(mdp: MDP, n_iter: int) -> V:
    # implementation of value iteration, the code is very similar to that of policy iteration
    # the difference is what kind of information we store
    v = {}
    for s in mdp.States:
        # initialize the value function
        v[s] = 0

    for i in range(n_iter):
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
        # copy the value function
        v = new_v

    return v