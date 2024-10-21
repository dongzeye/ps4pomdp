from itertools import count

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Tiger_FiniteHorizon(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(
        self, 
        horizon=10,
        discount=1.0,
        theta=0.35,
        listen_cost=-1, 
        treasure_reward=10, 
        tiger_penalty=-100,
        seed=None,
        options=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.theta = theta
        self.listen_cost = listen_cost
        self.treasure_reward = treasure_reward
        self.tiger_penalty = tiger_penalty
        self.discount = discount
        # Pre-compute the discount rates at each timestep; To match the infinite horizon
        # problem used by the solver, we add 1 step to account for the extra `Start` state.
        self.discount_rates = np.power(self.discount, np.arange(1, self.horizon+1))
        
        self.state_labels = ['Tiger-Left', 'Tiger-Right', 'Win', 'Lose', 'End']
        self.action_labels = ['Listen', 'Open-Left', 'Open-Right']
        self.obs_labels = ['Hear-Left', 'Hear-Right', 'Win', 'Lose']

        self.n_states = len(self.state_labels)
        self.n_actions = len(self.action_labels)
        self.n_obs = len(self.obs_labels)

        self.state_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_obs)

        self.state = None # current state
        self.obs = None # current observation
        
        self.steps_taken = 0 # time-step counter

        # Transition probabilities T[s, a, s'] = T(s' | s, a)
        # T is time-invariant for steps_taken < H
        self.transit_probs = np.array([
            [ # State 0 = Tiger-Left
                [1., 0., 0., 0., 0.], # Listen -> stay in Tiger-Left (0)
                [0., 0., 0., 1., 0.], # Open-Left -> Lose (3)
                [0., 0., 1., 0., 0.], # Open-Right -> Win (2)
            ], 
            [ # State 1 = Tiger-Right
                [0., 1., 0., 0., 0.], # Listen -> stay in Tiger-Right (1)
                [0., 0., 1., 0., 0.], # Open-Left -> Win (2)
                [0., 0., 0., 1., 0.], # Open-Right -> Lose (2)
            ], 
            [ # State 2 = Win
                [0., 0., 0., 0., 1.], # All actions -> End (4)
                [0., 0., 0., 0., 1.],  
                [0., 0., 0., 0., 1.], 
            ], 
            [ # State 3 = Lose
                [0., 0., 0., 0., 1.], # All actions -> End (4)
                [0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 1.],
            ], 
            [ # State 4 = End
                [0., 0., 0., 0., 1.], # All actions -> End (4)
                [0., 0., 0., 0., 1.],  
                [0., 0., 0., 0., 1.], 
            ], 
        ])

        # Observation probabilities Z[s, o] = Z(o | s)
        p = 0.5 + self.theta
        q = 0.5 - self.theta
        self.obs_probs = np.array([
            [p,  q,  0., 0.], # Tiger-Left
            [q,  p,  0., 0.], # Tiger-Right
            [0., 0., 1., 0.], # Win
            [0., 0., 0., 1.], # Lose
            [.5, .5, 0., 0.], # End
        ])

        self.reset(seed, options)

    def reward_fn(self, obs, action):
        # Apply discounting to reflect the reward in the infinite horizon problem for SARSOP
        # Rewards for Win/Lose are delayed in the finite horizon problem, thus receives discount at time t-1.
        prev_discount = self.discount_rates[self.steps_taken - 1]
        curr_discount = self.discount_rates[self.steps_taken]
        discounted_reward = (
            curr_discount * self.listen_cost * (action == 0)    # action 0 = Listen
            + prev_discount * self.treasure_reward * (obs == 2) # obs 2 = Win (i.e., found treasure)
            + prev_discount * self.tiger_penalty * (obs == 3)   # obs 3 = Lose (i.e., found tiger)
        )
        return discounted_reward
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.state = np.random.randint(0, 2)  # s_init = 0 (tiger-left) or 1 (tiger-right) with equal prob.
        self.steps_taken = 0
        self.obs = self._get_obs()
        return self.obs, self._get_info()

    def _get_info(self):
        return {'steps_taken': self.steps_taken}
    
    def _get_obs(self):
        obs_probs = self.obs_probs[self.state] # Z( * | s)
        return np.random.choice(self.n_obs, p=obs_probs)

    def step(self, action):
        # Compute reward r_t(o, a); the time-index is only related to the discounting
        reward = self.reward_fn(self.obs, action)

        # Move to next time-step
        self.steps_taken += 1
        # Get new state
        next_state_probs = self.transit_probs[self.state, action] # T( * | s, a)
        self.state = np.random.choice(self.n_states, p=next_state_probs)
        # Get new observation
        self.obs = self._get_obs()

        # Episode ends once we enter time H or state End (4)
        terminated = (self.steps_taken >= self.horizon) or (self.state == 4)
        return self.obs, reward, terminated, False, self._get_info()

    def generate_trajectory(self, actions=None):
        obs, _ = self.reset()
        if actions is None:
           actions = np.zeros(self.horizon//2, dtype=int)
           _more_actions = np.random.randint(0, self.n_actions, size=self.horizon - actions.shape[0])
           actions = np.concatenate([actions, _more_actions])
           
        traj = {'obs': [obs], 'action': [], 'reward': []}
        for h in count():
            action = actions[h]
            obs, reward, terminated, _, _ = self.step(action)

            traj['action'].append(action)
            traj['obs'].append(obs)
            traj['reward'].append(reward)

            if terminated:
                break # end current episode
        return traj

    def generate_random_trajectories(self, n_episodes):
        return [self.generate_trajectory() for _ in range(n_episodes)]


    def render(self, mode='human', close=False):
        if close:
            return
        print("Current State:", self.state_labels[self.state])
