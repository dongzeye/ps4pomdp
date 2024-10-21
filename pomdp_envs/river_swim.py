from numpyro import param
from pomdp_envs.finite_pomdp import FinitePOMDP
import jax
import jax.numpy as jnp
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RiverSwim(FinitePOMDP):
    metadata = {'render_modes': ['human']}

    @staticmethod
    def build_transition_kernel_static(params, n_states, n_actions=2):
        """
        Construct transition prob. matrixs for a batch of params.
        """
        params = jnp.array(params)
        p1, p2, p3 = params[..., 0], params[..., 1], params[..., 2]
        batch_shape = params.shape[:-1]
        left = 0
        right = 1
        transit_probs = jnp.zeros(batch_shape + (n_states, n_actions, n_states))
        
        # Action "Left" at state s always leads to s' = max(s-1, 0) 
        states = jnp.arange(n_states)
        next_states = jnp.clip(states - 1, min=0)
        transit_probs = transit_probs.at[..., states, left, next_states].set(1.0)

        # Action "Right":
        #   if s = 0, s' = 0 w.p. p1 and s' = 1 w.p. 1 - p1
        transit_probs = transit_probs.at[..., 0, right, 0].set(p1)
        transit_probs = transit_probs.at[..., 0, right, 1].set(p2 + p3)
        #   if s = L-1, s' = L-1 w.p. p1 and s'= L-2 w.p. 1 - p1
        transit_probs = transit_probs.at[..., -1, right, -1].set(p1)
        transit_probs = transit_probs.at[..., -1, right, -2].set(p2 + p3)
        #   if s = 1, ..., L-2:
        #       s' = s w.p. p1
        #       s' = s + 1 w.p. p2
        #       s' = s - 1 w.p. p3
        states = jnp.arange(1, n_states - 1)
        transit_probs = transit_probs.at[..., states, right, states].set(p1)
        transit_probs = transit_probs.at[..., states, right, states + 1].set(p2)
        transit_probs = transit_probs.at[..., states, right, states - 1].set(p3)
        return transit_probs
        
    @staticmethod
    def build_observation_kernel_static(params, n_states, n_obs):
        """
        Construct observation prob. matrix Z[s, o] = Z(o | s) according to the params.
        """
        params = jnp.array(params)
        q1, q2, q3 = params[..., 0], params[..., 1], params[..., 2]
        batch_shape = params.shape[:-1]
        obs_probs = jnp.zeros(batch_shape + (n_states, n_obs))
        # At s = 0 or L-1, no observation noise
        obs_probs = obs_probs.at[..., 0, 0].set(1.0)
        obs_probs = obs_probs.at[..., -1, -1].set(1.0)
        # At s = 1, observe o = 2 w.p. q1 and o = 3 w.p. 1 - q1
        # At s = L-2, observe o = L-2 w.p. q1 and o = L-3 w.p. 1 - q1
        value = jnp.array([q1, q2 + q3])
        obs_probs = obs_probs.at[..., 1, [1, 2]].set(value)
        obs_probs = obs_probs.at[..., -2, [-2, -3]].set(value)
        # At s = 2, ..., L-2:
        #   o = s w.p. q1
        #   o = s + 1 w.p. q2
        #   o = s - 1 w.p. q3
        states = jnp.arange(2, n_states - 2)
        obs_probs = obs_probs.at[..., states, states].set(q1) 
        obs_probs = obs_probs.at[..., states, states + 1].set(q2)
        obs_probs = obs_probs.at[..., states, states - 1].set(q3)
        return obs_probs
    
    @staticmethod
    def build_reward_matrix_static(params, n_obs, n_actions=2):
        """
        Construct reward matrix R[o, a] = R(o, a) according to the params.
        """
        params = jnp.array(params)
        r1, r2 = params[..., 0], params[..., 1]
        batch_shape = params.shape[:-1]
        # R(o, a) = r1 if o = 0, r2 if o = L-1, and 0 otherwise.
        rewards = np.zeros(batch_shape + (n_obs, n_actions))
        rewards[..., 0, :] = r1
        rewards[..., -1, :] = r2
        return rewards
    
    @staticmethod
    def build_init_state_probs_static(params):
        """
        Construct initial state prob. vector according to the params.
        param: prob. vector for the initial state with the correct ordering.
        """
        return params
    
    def __init__(self, river_length, horizon, params, discount=1.0):
        super().__init__(
            state_labels=[f's{i}' for i in range(river_length)],
            action_labels=['Left', 'Right'],
            obs_labels=[f'o{i}' for i in range(river_length)],
            horizon=horizon,
            params=params,
            discount=discount,
        )
        self.river_length = river_length