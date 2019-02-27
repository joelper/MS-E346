from typing import NamedTuple, Dict, Set
from modules.state_action_vars import S, A

class MP(NamedTuple):
    States: Set[S]
    P: Dict[S, Dict[S, float]]