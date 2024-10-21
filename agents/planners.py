# Disclaimer: The following implementation are adapted from `pomdp_py` with slight modification:
#    https://h2r.github.io/pomdp-py/html/_modules/pomdp_py/utils/interfaces/solvers.html#sarsop
import os
import subprocess
import xml.etree.ElementTree as ET
import jax.numpy as jnp
from pomdp_envs.utils import (
    create_tiger_pomdp_file
)
from pomdp_envs.finite_pomdp import FinitePOMDP

class AlphaVectorPolicy(object):
    def __init__(self, alphas, associated_actions, planner_params=None) -> None:
        """
        alphas: alpha vectors
        ind_actions: action indices associated with each alpha vector
        """
        self.alphas = jnp.asarray(alphas)
        self.associated_actions = jnp.asarray(associated_actions, dtype=int)
        self.planner_params = planner_params # sampled params used by the planner

        self.belief_state_shape = self.alphas.shape[-1]

    def __call__(self, belief_state, timestep=None):
        """
        Return an action at the belief state according to the alpha vector representation 
        of the POMDP policy. The best action has the largest dot product <alpha, belief_state>. 
        Belief state will be expanded to account for time-indexed state copies in the 
        equivalent infinite horizon problem.
        
        NOTE: Each alpha vector is associated with an action; SARSOP may return several alpha vectors
            associated with the same action.
        """
        if timestep is not None:
            n_states = belief_state.shape[0]
            offset = 1 + n_states * timestep
            belief_state = jnp.zeros(self.belief_state_shape).at[offset:offset + n_states].set(belief_state)
            
        ind_best_action = jnp.argmax(self.alphas @ belief_state)
        return int(self.associated_actions[ind_best_action])

    def value(self, belief_state=None):
        if belief_state is None:
            # let initial beleif be concentrated at `Start` state (i.e., position 0)
            return self.alphas[:, 0].max()
        if (scale := jnp.sum(belief_state)) != 1:
            # belief state is unnormalized
            belief_state = belief_state / scale
        
        return jnp.max(self.alphas @ belief_state)

    @classmethod
    def from_file(cls, policy_path):
        # Parse SARSOP policy in terms of alpha vectors
        alphas = []
        associated_actions = []
        root = ET.parse(policy_path).getroot()
        for vector in root.find("AlphaVector").iter("Vector"):
            alpha_vector = tuple(map(float, vector.text.split()))
            alphas.append(alpha_vector)
            associated_actions.append(vector.attrib["action"])

        return cls(alphas, associated_actions)



def run_sarsop(
    pomdp_path,
    pomdpsol_path="sarsop/src/pomdpsol",
    timeout=30,
    memory=100,
    precision=0.5,
    policy_path="temp.policy",
    remove_generated_files=False,
    logfile=None,
    **kwargs,
):
    """
    SARSOP, using the binary from https://github.com/AdaCompNUS/sarsop
    This is an anytime POMDP planning algorithm

    Args:
        pomdp_path (str): Path to the POMDP file
        pomdpsol_path (str): Path to the `pomdpsol` binary
        timeout (int): The time limit (seconds) to run the algorithm until termination
        memory (int): The memory size (mb) to run the algorithm until termination
        precision (float): solver runs until regret is less than `precision`
        policy_path (str): Name of the policy file that will be created after solving
        remove_generated_files (bool): Remove created files during solving after finish.
        logfile (str): Path to file to write the log of both stdout and stderr
    Returns:
       AlphaVectorPolicy: The policy returned by the solver.
    """
    if logfile is None:
        stdout = None
        stderr = None
    else:
        logf = open(logfile, "w")
        stdout = subprocess.PIPE
        stderr = subprocess.STDOUT

    proc = subprocess.Popen(
        [
            pomdpsol_path,
            "--timeout",
            str(timeout),
            "--memory",
            str(memory),
            "--precision",
            str(precision),
            "--output",
            policy_path,
            pomdp_path,
        ],
        stdout=stdout,
        stderr=stderr,
    )
    if logfile is not None:
        for line in proc.stdout:
            line = line.decode("utf-8")
            # sys.stdout.write(line)
            logf.write(line)
    proc.wait()

    policy = AlphaVectorPolicy.from_file(policy_path)

    # Remove temporary files
    if remove_generated_files:
        os.remove(policy_path)
    if logfile is not None:
        logf.close()

    return policy


class POMDPPlanner(object):
    def solve(self, env: FinitePOMDP):
        pass

class SARSOPPlanner(POMDPPlanner):
    _default_sarsop_config = {
        'pomdpsol_path': 'sarsop/src/pomdpsol', 
        'timeout': 30,
        'memory': 1024,
        'precision': 0.05,
        'logfile': 'tmp_sarsop.log'
    }
    def __init__(self, discount=0.99, pomdp_path='tmp.pomdp', policy_path='tmp.policy', sarsop_args=None):
        self.discount = discount
        self.pomdp_path = pomdp_path
        self.policy_path = policy_path
        self.sarsop_config = {
            **SARSOPPlanner._default_sarsop_config,
            **(sarsop_args if sarsop_args else {})
        }

    def solve(self, env: FinitePOMDP):
        env.to_pomdp_file(self.discount, self.pomdp_path)
        return run_sarsop(self.pomdp_path, **self.sarsop_config)


def tiger_find_policy(
    horizon,
    theta,
    tiger_penalty=-100,
    treasure_reward=10,
    listen_cost=-1,
    discount=0.999,
    pomdp_path=None,
    sarsop_config=None,
) -> AlphaVectorPolicy:
    """Find optimal policy with given config via SARSOP"""
    if sarsop_config is None:
        sarsop_config = {}

    create_tiger_pomdp_file(
        horizon=horizon,
        theta=theta,
        discount=discount,
        tiger_penalty=tiger_penalty,
        treasure_reward=treasure_reward,
        listen_cost=listen_cost,
        pomdp_path=pomdp_path, # pomdp file will be saved to this path
    )
    policy = run_sarsop(pomdp_path, **sarsop_config)
    policy.planner_theta = theta # store theta used for planning
    return policy

