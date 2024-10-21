from experiments.utils import load_configuration

import jax
import jax.numpy as jnp

import numpyro.distributions as dist
from agents.ps4tiger import (
    tiger_construct_observation_matrix,
    tiger_expand_belief_state_by_time,
)
from agents.planners import AlphaVectorPolicy


def mdp_backward_policy_evaluation(layered_state_space, transition_probs, rewards, policy, horizon):
    """
    Backward policy evaluation for a finite horizon MDP with layered state space using a deterministic policy.

    Args:
    - layered_state_space (list): List of lists, where each inner list represents states at a given time step.
    - transition_probs (dict): T[s, a, s'] = P(s' | s, a).
    - rewards (dict): R_t(s, a) - {(t, s, a): reward}, time dependent.
    - policy (dict): Deterministic Policy - {s: a}.
    - horizon (int): Time horizon.

    Returns:
    - V (list): Value function - list of dicts {s: value} for each time step.
    """
    # Initialize the value function for each time step
    V = [{s: 0 for s in layer} for layer in layered_state_space]
    
    # Iterate backwards from T-1 to 0
    for t in range(horizon - 1, -1, -1):
        current_states = layered_state_space[t]
        next_states = layered_state_space[t + 1]
        for s in current_states:
            a = policy[s]  # Get the action from the deterministic policy
            V[t][s] = float(
                rewards[t, s, a] + sum(transition_probs.get((s, a, s_next), 0) * V[t + 1][s_next] 
                                       for s_next in next_states)
            )
    return V


def mdp_find_optimal_policy(layered_state_space, actions, transition_probs, rewards, horizon):
    """
    Find the optimal policy for a finite horizon MDP with layered state space.

    Args:
    - layered_state_space (list): List of lists, where each inner list represents states at a given time step.
    - actions (list): List of actions.
    - transition_probs (dict): T[s, a, s'] = P(s' | s, a).
    - rewards (dict): R_t(s, a) - {(t, s, a): reward}, time dependent.
    - horizon (int): Time horizon.

    Returns:
    - V (list): Optimal value function - list of dicts {s: value} for each time step.
    - optimal_policy (list): Optimal policy - list of dicts {s: a} for each time step.
    """
    # Initialize the value function for each time step
    V = [{s: 0 for s in layer} for layer in layered_state_space]
    optimal_policy = [{s: None for s in layer} for layer in layered_state_space]

    # Iterate backwards from T-1 to 0
    for t in range(horizon - 1, -1, -1):
        current_states = layered_state_space[t]
        next_states = layered_state_space[t + 1]
        
        for s in current_states:
            action_values = jnp.array([
                rewards[t, s, a] + sum(V[t + 1][s_next] * transition_probs.get((s, a, s_next), 0) 
                                       for s_next in next_states)
                for a in actions
            ])

            best_action_idx = jnp.argmax(action_values)
            best_action = actions[best_action_idx]
            V[t][s] = action_values[best_action_idx]
            optimal_policy[t][s] = best_action

    return V, optimal_policy

@jax.jit
def tiger_belief_state_from_counts(obs_counts, obs_probs, init_state_probs):
    """
    obs_counts: array of counts (int) for each possible observation. Shape: (n_obs,)
    obs_probs: observation probability matrix. Shape: (n_states, n_obs)
    state_prior: prior distrbituion of (hidden) states drawn during each reset. Shape: (n_states, )
    NOTE: Cannot handle zero probabilities due to computation with log-probabilities. 
    """
    log_init_state_probs = jnp.log(init_state_probs)
    log_obs_probs = jnp.log(obs_probs)
    log_likelihoods = jnp.sum(log_obs_probs * obs_counts, axis=-1)

    unnormalized_log_posterior = log_likelihoods + log_init_state_probs
    
    # Compute unnormalized posterior with the max-log trick
    max_log_posterior = jnp.max(unnormalized_log_posterior)
    unnormalized_posterior = jnp.exp(unnormalized_log_posterior - max_log_posterior)
    
    posterior = unnormalized_posterior / jnp.sum(unnormalized_posterior)
    return posterior

def tiger_expanded_belief_state_from_counts(obs_counts, obs_probs, init_state_probs, horizon):
    t = int(jnp.sum(obs_counts) - 1)
    belief_state = tiger_belief_state_from_counts(obs_counts, obs_probs, init_state_probs)
    return tiger_expand_belief_state_by_time(belief_state, t, horizon, (t + 1 >= horizon))



@jax.jit
def tiger_belief_state_from_counts_via_multinomial(obs_counts, obs_probs, state_prior):
    """
    (Same as `tiger_belief_state_from_counts`)
    obs_counts: array of counts (int) for each possible observation. Shape: (n_obs,)
    obs_probs: observation probability matrix. Shape: (n_states, n_obs)
    state_prior: prior distrbituion of (hidden) states drawn during each reset. Shape: (n_states, )
    NOTE: Cannot handle zero probabilities due to computation with log-probabilities. 
    """
    log_state_prior = jnp.log(state_prior)
    log_likelihoods = dist.Multinomial(probs=obs_probs).log_prob(obs_counts)
    
    unnormalized_log_posterior = log_likelihoods + log_state_prior
    
    # Compute unnormalized posterior with the max-log trick
    max_log_posterior = jnp.max(unnormalized_log_posterior)
    unnormalized_posterior = jnp.exp(unnormalized_log_posterior - max_log_posterior)
    
    posterior = unnormalized_posterior / jnp.sum(unnormalized_posterior)
    return posterior


@jax.jit
def tiger_statistics_mdp_next_count_probs(current_count, obs_probs, init_state_probs):
    """
    Compute the probabiliyt of next conuting state given current counting state, i.e.
        P(xi_{t+1} = (j, t + 1 - j) | xi_t = (i, t-i), a_t = listen) for j = i, i+1 
    """

    current_belief = tiger_belief_state_from_counts(
        obs_counts=current_count, # +1 due to 0-indexed timesteps
        obs_probs=obs_probs,
        init_state_probs=init_state_probs,
    ) 

    # Using Eq (6)
    # obs_probs = Array([[0.5 + theta, 0.5 - theta], 
    #                    [0.5 - theta, 0.5 + theta]])
    p_hear_left = jnp.sum(obs_probs[0] * current_belief)
    return jnp.array([p_hear_left, 1 - p_hear_left])


class TigerStatisticsMDP(object):
    def __init__(
        self, 
        theta, 
        horizon, 
        discount=1,
        listen_cost=-1,
        tiger_penalty=-100,
        treasure_reward=10,
        init_tiger_state_probs=None,
    ):
        # Environment configs
        self.theta = theta
        self.horizon = horizon
        self.listen_cost = listen_cost
        self.tiger_penalty = tiger_penalty
        self.treasure_reward = treasure_reward
        self.discount = discount
        # Pre-compute the discount rates at each timestep; To match the infinite horizon
        # problem used by the solver, we add 1 step to account for the extra `Start` state.
        self.discount_rates = jnp.power(self.discount, jnp.arange(1, self.horizon+1))

        # Statistics state space
        self.counts_space = [(i, j) for i in range(horizon + 1) for j in range(horizon + 1) if 0 < i + j <= horizon]
        self.outcome_space = ["Win", "Lose", "End"]
        self.state_space = self.counts_space + self.outcome_space
        self.timed_state_space = [
            [(i, t + 1 - i) for i in range(t + 2)]
            for t in range(horizon + 1)
        ]
        self.timed_state_space = self.timed_state_space[:1] \
            + [layer + self.outcome_space for layer in self.timed_state_space[1:]]
        
        self.action_space = ['Listen', 'Open-Left', 'Open-Right']

        self.state_index_map = {state: index for index, state in enumerate(self.state_space)}
        self.action_index_map = {action: index for index, action in enumerate(self.action_space)}

        # self.log_init_state_probs = jnp.log(self.init_state_probs)
        self.obs_probs = tiger_construct_observation_matrix(theta)
        self.init_tiger_state_probs = jnp.array([0.5, 0.5]) if init_tiger_state_probs is None \
            else init_tiger_state_probs
        self.init_state_probs = jnp.sum(self.obs_probs * self.init_tiger_state_probs, axis=-1)


        self.next_count_probs, self.belief_state_map = self.compute_next_count_probs_and_belief_state_map(theta)

        self.transition_probs = {
            ((i, j), 'Listen', next_counts): prob
            for (i, j) in self.counts_space
            for next_counts, prob in zip([(i+1, j), (i, j+1)], self.next_count_probs[i, j])
        }

        # Tranistion probs: T[s, a, s'] = P(s' | s, a)
        # belief_state[counts] = [P(Tiger-Left | counts), P(Tiger-Right | counts)]
        self.transition_probs.update({
            (counts, 'Open-Left', outcome): prob
            for counts in self.counts_space
            for outcome, prob in zip(['Lose', 'Win'], self.belief_state_map[counts])
        })
        self.transition_probs.update({
            (counts, 'Open-Right', outcome): prob
            for counts in self.counts_space
            for outcome, prob in zip(['Win', 'Lose'], self.belief_state_map[counts])
        })
        self.transition_probs.update({
            (state, action, 'End'): 1.0
            for state in ['Win', 'Lose', 'End']
            for action in self.action_space
        })
    
        self.timed_reward_map = {
            (t, s, a): self.reward_fn(t, s, a)
            for t in range(horizon)
            for s in self.timed_state_space[t]
            for a in self.action_space
        }

        # cache optimal policy and values once computed
        self._optimal_policy = None
        self._optimal_value_fn = None
        self._optimal_value = None


    def compute_next_count_probs_and_belief_state_map(self, theta):
        next_count_probs = {}
        belief_state_map = {}

        obs_probs_given_tiger_left = jnp.array([0.5 + theta, 0.5 - theta])
        for (i, j) in self.counts_space:
            # At time t = 0, ... H-1, there have been t+1 observations.
            belief_state = tiger_belief_state_from_counts(jnp.array([i, j]), self.obs_probs, self.init_state_probs)
            belief_state_map[i, j] = belief_state
            p_hear_left = jnp.sum(obs_probs_given_tiger_left * belief_state)
            next_count_probs[i, j] = jnp.array([p_hear_left, 1 - p_hear_left])
        return next_count_probs, belief_state_map

    def reward_fn(self, timestep, state, action):
        # Apply discounting to reflect the reward in the infinite horizon problem for SARSOP
        # Rewards for Win/Lose are delayed in the finite horizon problem, thus receives discount at time t-1.
        reward = 0
        if state == 'Win':
            reward += self.discount_rates[timestep - 1] * self.treasure_reward
        elif state == 'Lose':
            reward += self.discount_rates[timestep - 1] * self.tiger_penalty
        
        if action == 'Listen':
            reward += self.discount_rates[timestep] * self.listen_cost

        return reward
    
    def _find_optimal_value(self):
        self._optimal_value_fn, self._optimal_policy = mdp_find_optimal_policy(
            layered_state_space=self.timed_state_space,
            actions=self.action_space,
            transition_probs=self.transition_probs,
            rewards=self.timed_reward_map,
            horizon=self.horizon,
        )
        initial_values = jnp.asarray(list(self._optimal_value_fn[0].values()))
        self._optimal_value = (self.init_state_probs @ initial_values).item()

    @property
    def optimal_policy(self):
        if self._optimal_policy is None:
            self._find_optimal_value()
        return self._optimal_policy
    
    @property
    def optimal_value_fn(self):
        if self._optimal_value_fn is None:
            self._find_optimal_value()
        return self._optimal_value_fn
    
    @property
    def optimal_value(self):
        if self._optimal_value is None:
            self._find_optimal_value()
        return self._optimal_value

    def convert_alpha_vector_to_count_based_policy(self, policy: AlphaVectorPolicy | str):
        if isinstance(policy, str):
            policy = AlphaVectorPolicy.from_file(policy)
        theta_eval = policy.planner_theta
        obs_probs_eval = jnp.array([[0.5 + theta_eval, 0.5 - theta_eval],
                                    [0.5 - theta_eval, 0.5 + theta_eval]])
        
        def _count_based_policy(obs_counts):
            expanded_belief_state = tiger_expanded_belief_state_from_counts(
                jnp.asarray(obs_counts), obs_probs_eval, self.init_state_probs, self.horizon)
            action_index = policy(expanded_belief_state)
            return self.action_space[action_index]
        
        count_based_policy = {
            counts: _count_based_policy(counts) for counts in self.counts_space
        }
        exapnded_terminal_state = jnp.zeros(2 * self.horizon).at[-1].set(1)
        end_action = self.action_space[policy(exapnded_terminal_state)]
        count_based_policy.update({outcome: end_action for outcome in self.outcome_space})
        return count_based_policy
    
    def evaluate_alpha_vector_policy(self, policy: AlphaVectorPolicy):
        assert policy.planner_theta is not None, "Missing planner_theta"
        count_based_policy = self.convert_alpha_vector_to_count_based_policy(policy)
        return mdp_backward_policy_evaluation(
            layered_state_space=self.timed_state_space, 
            transition_probs=self.transition_probs,
            rewards=self.timed_reward_map,
            policy=count_based_policy,
            horizon=self.horizon,
        )
    
    def compute_policy_values(self, policies: list[AlphaVectorPolicy]):
        initial_values = jnp.array([
            list(self.evaluate_alpha_vector_policy(policy)[0].values())
            for policy in policies
        ]) # Shape: (n_policies, 2)
        return initial_values @ self.init_state_probs
    
    def compute_per_round_regrets(self, policies: list[AlphaVectorPolicy]):
        policy_values = self.compute_policy_values(policies) 
        return self.optimal_value - policy_values

    
    @classmethod
    def from_exp_config(cls, config: dict | str):
        if isinstance(config, str):
            config = load_configuration(config)
        env_config = config['env_config']
        return cls(
            theta=env_config['theta'],
            horizon=env_config['horizon'], 
            discount=env_config['discount'],
            listen_cost=env_config['listen_cost'],
            tiger_penalty=env_config['tiger_penalty'],
            treasure_reward=env_config['treasure_reward'],
        )

