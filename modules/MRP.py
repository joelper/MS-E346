from typing import NamedTuple, Dict, Union
from modules.state_action_vars import S, A
from modules.MP import MP

class MRP(NamedTuple):
    # assumes that the reward is just a function of the current state
    mp: MP
    R: Union[Dict[S, float], Dict[S, Dict[S, float]]]
    gamma: float