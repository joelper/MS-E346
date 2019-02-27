from typing import NamedTuple, Dict, Set, Union
from modules.state_action_vars import S, A

class MDP(NamedTuple):
    States: Set[S]
    # the transitions depend on s, a, and s'
    # mapping from a state to a mapping of an action to a mapping of a state to a float (probability)
    P: Dict[S, Dict[A, Dict[S, float]]]
    Actions: A
    # reward is a function of the current state,  and the action
    R: Union[Dict[S, Dict[A, float]], Dict[S, Dict[A, Dict[S, float]]]]
    gamma: float


class Policy(NamedTuple):
    # state to action to a probability
    pi: Dict[S, Dict[A, float]]


class V(NamedTuple):
    vf: Dict[S, float]


class Q(NamedTuple):
    q: Dict[S, Dict[A, float]]