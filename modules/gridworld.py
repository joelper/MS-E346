from typing import Set, Tuple
from modules.state_action_vars import S, A
from modules.MDP import MDP


def is_in_grid(state: Tuple[int, int], size: int) -> bool:
    # helper function to check whether a state is in the grid
    return  state[0] >= 0 and state[0] < size and state[1] >= 0 and state[1] < size


def get_neighbor_states(s: Tuple[int, int], size: int) -> Set[Tuple[int, int]]:
    # function to return a set of neighboring states in the grid
    nbr_states = set()

    up_state = s[0 ] -1, s[1]
    if is_in_grid(up_state, size):
        nbr_states.add(up_state)

    down_state = s[0 ] +1, s[1]
    if is_in_grid(down_state, size):
        nbr_states.add(down_state)

    left_state = s[0], s[1 ] -1
    if is_in_grid(left_state, size):
        nbr_states.add(left_state)

    right_state = s[0], s[1 ] +1
    if is_in_grid(right_state, size):
        nbr_states.add(right_state)

    return nbr_states


def get_neighbor_direction(s: S, sp: S) -> int:
    # function to figure out in which direction the state sp is from state s
    # assume that both states are adjacent
    if s[1] > sp[1]:
        # sp is to the left of s
        return 1
    elif s[1] < sp[1]:
        # sp is to the right of s
        return 2
    elif s[0] > sp[0]:
        # sp is above s
        return 3
    elif s[0] < sp[0]:
        # sp is below s
        return 4
    else:
        # sp is equal to s
        return 0


def gridworld_sa():
    # define the gridworld parameters
    States = set()
    P = {}
    A = set()
    for i in range(4):
        # 1 is move left, 2 is move right, 3 is up, 4 is down
        A.add(i+1)
        for j in range(4):
            state = (i, j)
            States.add(state)

    for s in States:
        P[s] = {}
        nbrs = get_neighbor_states(s, 4)
        for a in A:
            P[s][a] = {}
            if s == (0,0) or s == (3,3):
                P[s][a][s] = 1.0
            else:
                agg_p = 0
                for sp in nbrs:
                    if get_neighbor_direction(s, sp) == a:
                        P[s][a][sp] = 0.7
                        agg_p += 0.7
                    else:
                        P[s][a][sp] = 0.1
                        agg_p += 0.1
                if len(nbrs) < 4:
                    P[s][a][s] = 1. - agg_p

    return States, P, A


def gridworld_rew(States: Set[S], A: Set[A]):
    # here the reward is just a function of the current state
    R = {}
    for s in States:
        R[s] = {}
        for a in A:
            if s == (0, 0) or s == (3, 3):
                R[s][a] = 3.
            elif s == (1, 2):
                R[s][a] = -2.
            else:
                R[s][a] = 0.

    return R


def gridworld(gamma: float):
    States, P, A = gridworld_sa()
    R = gridworld_rew(States, A)

    return MDP(States, P, A, R, gamma)