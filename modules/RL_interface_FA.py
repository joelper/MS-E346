from typing import NamedTuple, Callable, Tuple, Set
from modules.state_action_vars import S, A


class RL_interface_FA(NamedTuple):
    # interface for reinforcement learning with value function approximation,
    # largely inspired by Professor Ashwin Rao's implementation

    # function that takes in a state and return a set of possible actions
    state_action_func: Callable[[S], Set[A]]
    # function that takes in a state and an action, and returns a new state sp and the reward
    state_reward_func: Callable[[S, A], Tuple[S, float]]
    # initial state generator
    init_state_gen: Callable[[], S]
    # inital state generator
    init_state_action_gen: Callable[[], Tuple[S, A]]
    # discount factor
    gamma: float