from copy import deepcopy

import jax
import jax.numpy as jnp
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpyro import param

from pomdp_envs.utils import create_finite_horizon_stationary_pomdp_file


class FinitePOMDP(gym.Env):
    metadata = {'render_modes': ['human']}

    @staticmethod
    def build_transition_kernel_static(params, n_states, n_actions):
        """
        Construct transition prob. matrixs for a batch of params.
        NOTE: 
            - Must be staticmethod to be used in external NumPyro inference models. 
            - Input and output should be JAX arrays 
        """
        raise NotImplementedError

        
    @staticmethod
    def build_observation_kernel_static(params, n_states, n_obs):
        """
        Construct observation prob. matrix Z[s, o] = Z(o | s) according to the params.
        NOTE: 
            - Must be staticmethod to be used in external NumPyro inference models. 
            - Input and output should be JAX arrays 
        """
        raise NotImplementedError
    
    @staticmethod
    def build_reward_matrix_static(params, n_obs, n_actions):
        """
        Construct reward matrix R[o, a] = R(o, a) according to the params.
        NOTE: 
            - Must be staticmethod to be used in external NumPyro inference models. 
            - Input and output should be JAX arrays 
        """
        raise NotImplementedError
    
    @staticmethod
    def build_init_state_probs_static(params, n_states):
        """
        Construct initial state prob. vector according to the params.
        NOTE: 
            - Must be staticmethod to be used in NumPyro inference models. 
            - Input and output should be JAX arrays 
        """
        raise NotImplementedError
    
    def __init__(
        self, 
        state_labels, 
        action_labels, 
        obs_labels, 
        horizon, 
        params,
        discount=None,
    ):
        super().__init__()
        # State, action, and observation labels
        self.state_labels = state_labels
        self.action_labels = action_labels
        self.obs_labels = obs_labels

        # Number of states, actions, and observations
        self.n_states = len(self.state_labels)
        self.n_actions = len(self.action_labels)
        self.n_obs = len(self.obs_labels)

        # Spaces for state, action, and observation
        self.state_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_obs)

        self.horizon = horizon
        self.timestep = 0  # time-step counter
        
        # Discounting factor
        self.discount = discount

        # Internal state and observation trackers
        self._state = None  # current state
        self._obs = None    # current observation

        # POMDP parameters (initialize to empty; some env may be un-parameterized)
        self.params = {}
        # Transition prob. matrix T[s, a, s'] = T(s' | s, a)
        self.transit_probs = None
        # Observation prob. matrix Z[s, o] = Z(o | s)
        self.obs_probs = None
        # Reward matrix R[o, a] = R(o, a)
        self.rewards = None
        # Initial state prob. vector b0[s] = b0(s)
        self.init_state_probs = None

        # Set model params
        self.set_params(params)
    
    def build_transition_kernel(self, params, *args, **kwargs):
        return self.build_transition_kernel_static(params, self.n_states, self.n_actions)
    
    def build_observation_kernel(self, params, *args, **kwargs):
        return self.build_observation_kernel_static(params, self.n_states, self.n_obs)
    
    def build_reward_matrix(self, params, *args, **kwargs):
        return self.build_reward_matrix_static(params, self.n_obs, self.n_actions)
    
    def build_init_state_probs(self, params, *args, **kwargs):
        return self.build_init_state_probs_static(params, self.n_states)
    
    def set_params(self, params, **kwargs):
        params = params | kwargs
        params = {k: np.array(v) for k, v in params.items()}
        if 'transition_kernel' in params:
            self.transit_probs = params['transition_kernel']

        if 'observation_kernel' in params:
            self.obs_probs = params['observation_kernel']

        if 'reward_matrix' in params:
            self.rewards = params['reward_matrx']

        if 'init_state_probs' in params:
            self.init_state_probs = params['init_state_probs']

        if 'transition_params' in params:
            self.transit_probs = self.build_transition_kernel(params['transition_params'])

        if 'observation_params' in params:
            self.obs_probs = self.build_observation_kernel(params['observation_params'])

        if 'reward_params' in params:
            self.rewards = self.build_reward_matrix(params['reward_params'])

        if 'init_state_params' in params:
            self.init_state_probs = self.build_init_state_probs(params['init_state_params'])

        self.params = self.params | params
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state and observation.
        """
        if seed: 
            np.random.seed(seed)
        self._state = np.random.choice(self.n_states, p=self.init_state_probs)
        self._obs = self._get_obs()
        self.timestep = 0
        return self._obs, self._get_info()

    def _get_info(self):
        return {'timestep': self.timestep}
    
    def _get_obs(self):
        obs_probs = self.obs_probs[self._state] # Z(* | s)
        return np.random.choice(self.n_obs, p=obs_probs)

    def _get_reward(self, obs, action):
        if self.discount is not None:
            return np.power(self.discount, self.timestep) * self.rewards[obs, action]
        else:
            return self.rewards[obs, action]
    
    def step(self, action):
        # Move to next time-step
        self.timestep += 1
        # Get new state
        next_state_probs = self.transit_probs[self._state, action] # T( * | s, a)
        self._state = np.random.choice(self.n_states, p=next_state_probs)
        # Get reward r(o, a)
        reward = self._get_reward(self._obs, action)
        # Get new obs
        self._obs = self._get_obs()

        # Episode ends once we enter time H
        terminated = (self.timestep >= self.horizon)
        
        return self._obs, reward, terminated, False, self._get_info()

    def render(self, mode='human', close=False):
        if close:
            return
        print(f"Observation: {self._obs}")

    def clone(self, remove_params=True):
        # Create a deep copy
        new_env = deepcopy(self)
        if remove_params:
            new_env.params = {}
            new_env.transit_probs = None
            new_env.obs_probs = None
            new_env.rewards = None
            new_env.init_state_probs = None
        return new_env
    
    def to_pomdp_file(self, discount, filepath, decimals=6, header=None):
        return create_finite_horizon_stationary_pomdp_file(
            horizon=self.horizon,
            states=self.state_labels,
            actions=self.action_labels,
            observations=self.obs_labels,
            init_state_probs=self.init_state_probs,
            transition_matrix=self.transit_probs,
            observation_matrix=self.obs_probs,
            reward_matrix=self.rewards,
            discount=discount,
            pomdp_path=filepath,
            decimals=decimals,
            header=header,
        )


class SparseRewardPOMDP(FinitePOMDP):
    @staticmethod
    def build_transition_kernel_static(params, n_states, n_actions):
        """
        Construct transition prob. matrixs for a batch of params.
        """
        raise NotImplementedError

        
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
    def build_reward_matrix_static(params, n_obs, n_actions):
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
    def build_init_state_probs_static(params, n_states):
        """
        Construct initial state prob. vector according to the params.
        """
        raise NotImplementedError

    def __init__(self, n_states, n_actions, n_obs, horizon, params, discount=None):
        super().__init__(
            state_labels=[f's{i}' for i in range(n_states)],
            action_labels=[f'a{i}' for i in range(n_actions)],
            obs_labels=[f'o{i}' for i in range(n_obs)],
            horizon=horizon,
            params=params,
            discount=discount,
        )

