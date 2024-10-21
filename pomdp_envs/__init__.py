from .tiger import Tiger_FiniteHorizon
from .finite_pomdp import SparseRewardPOMDP
from .river_swim import RiverSwim

from gymnasium import register

# register POMDP environments
register(id="Tiger-v0", entry_point="pomdp_envs.tiger:Tiger_FiniteHorizon")
register(id="RiverSwim-v0", entry_point="pomdp_envs.river_swim:RiverSwim")
register(id="SparseRewardPOMDP-v0", entry_point="pomdp_envs.finite_pomdp:SparseRewardPOMDP")