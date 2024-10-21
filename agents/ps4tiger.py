import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro as npyro
import numpyro.distributions as dist
from numpyro.infer import HMC, MCMC, MixedHMC

from agents.planners import tiger_find_policy

from constants import MIN_THETA, MAX_THETA

@jax.jit
def tiger_update_belief_state(belief_state, theta, obs):
    """
    Update `belief_state` according to the new `obs`. 
    Return the belief state (expanded if `expand_by_time` is True)
    TODO: Improve numerical stability by using log-likeilhoods for computation
    """   
    # Sampled obs kernel: Z_theta[s, o] = Z_theta(o|s)
    # State 0: Tiger-Left, state 1: Tiger-Right
    # Obs 0: Hear-Left, obs 1: Hear-Right
    sampled_obs_kernel = 0.5 + jnp.array([[theta, - theta],
                                          [- theta, theta]])
    # Compute and save new belief state
    likelihoods = sampled_obs_kernel[:, obs].squeeze()  # [Z(o|s=0), Z(o|s=1)]
    unnormalized_next_belief_state = likelihoods * belief_state
    return unnormalized_next_belief_state / jnp.sum(unnormalized_next_belief_state)


# @partial(jax.jit, static_argnames=['timestep', 'horizon', 'ended'])
def tiger_expand_belief_state_by_time(belief_state, timestep, horizon, ended):
    # NOTE: Should not be called when timestep == horizon
    zeros = jnp.zeros(2 * horizon) # initialize with zeros
    if ended:
        return zeros.at[-1].set(1) # state is `End` w.p. 1
    else:
        i = 2 * timestep + 1 # indices of (TigerLeft_t, TigerRight_t) = (2t+1, 2t+2) for t = 0, ..., H-2
        return zeros.at[i:i+2].set(belief_state)


@jax.jit
def tiger_construct_observation_matrix(theta):
    return jnp.array([[theta, -theta], [-theta, theta]]) + 0.5
 


class PS4Tiger(object):
    available_modes = ['online', 'offline']
    def __init__(
        self, 
        horizon, 
        prng_key=None, 
        discount=0.99, 
        reward_config=None,
        mcmc_config=None, 
        sarsop_config=None,
        pomdp_path=None,
        mode='online',
    ) -> None:
        # Defaults
        self.prng_key = jr.PRNGKey(0) if prng_key is None else prng_key
        self.pomdp_path = 'tmp.pomdp' if pomdp_path is None else pomdp_path

        self.reward_config = reward_config if reward_config is not None \
            else {
                'listen_cost': -1,
                'treasure_reward': 10,
                'tiger_penalty': -100,
            }
        self.mcmc_config = mcmc_config if mcmc_config is not None \
            else {
                'num_warmup': 2000, 
                'num_samples': 2000, 
                'num_chains': 4
            }
        self.sarsop_config = sarsop_config if sarsop_config is not None \
            else {
                'pomdpsol_path': 'sarsop/src/pomdpsol', 
                'timeout': 30,
                'memory': 1024,
                'precision': 0.05,
            }
        
        self.set_mode(mode)

        # Relevant env configurations
        self.horizon = horizon
        self.n_actions = 3 # Listen (0), Open-Left (1), Open-Right (1)
        self.n_states = 2 # Tigher-Left (0) or Tiger-Right (1) (never deal with other states)

        # Discount factor for the inifinite-horizon POMDP solvers
        self.discount = discount
        
        # MCMC for posterior inference
        self.mcmc = None
        self.posterior = None

        # Belief state configurations
        self.shape_belief_state = (self.n_states, )
        # Exapnd belief state with time indices for the infinite horizon problem: 
        #   Start (0), TL_0 (1), TR_0 (2), TL_1 (3), TR_1 (4), ..., TR_H (2H-2), End (2H-1)
        self.shape_expanded_belief_state = (self.n_states * self.horizon, )

        # Belief state: posterior dist. of state given obs in current traj. and sampled theta
        self.belief_state = None
        self.expanded_belief_state = None

        # POMDP dynamics are indexed by parameter theta
        # NOTE: Belief over theta is only updated once per episode
        self.planner_theta = None # a posterior sample of theta for planning
        # Conditionally-optimal policy given the sampled theta
        self.policy = None

        # Counters
        self.episode = None
        self.timestep = None
        
        # Trajectory containers
        self.past_trajectories = None
        self.current_trajectory = None

        # Training data: flattened obs and eposide indices (ind_episodes) associated with each obs.
        self.train_data = None
        
    def set_mode(self, mode):
        assert mode in self.available_modes, \
            f'Availabe modes: {PS4Tiger.available_modes}; Only update policy when online'
        self.mode = mode

    def reset(self, prng_key=None, theta=None, policy=None, mode=None):
        if prng_key is not None:
            self.prng_key = prng_key
        if mode is not None:
            self.set_mode(mode)
        
        # Reset counters and data containers
        self.episode = 0
        self.timestep = 0
        self.past_trajectories = {}
        self.current_trajectory = {'obs': jnp.empty(0), 'action': jnp.empty(0)}
        self.train_data = {'obs': jnp.empty(0), 'ind_episodes': jnp.empty(0, dtype=int)}

        # Reset planning parameter theta
        if theta is None:
            self.prng_key, subkey = jr.split(self.prng_key)
            self.planner_theta = jr.uniform(
                subkey, (1,), minval=MIN_THETA, maxval=MAX_THETA).item() # sample from uniform prior
        else:
            self.planner_theta = theta
        self.posterior = {'theta': self.planner_theta}

        # Reset policy
        if policy is None:
            # Update policy according to current theta
            self.update_policy(self.planner_theta)
        else:
            # Reset to the given policy
            self.policy = policy

        self.reset_belief_state()

    def reset_belief_state(self):
        # Reset to uniform prior over states
        self.belief_state = jnp.array([.5, .5]) # `uniform` prior
        # Skip the fixed "Start" state at position 0 in the expanded state space
        self.expanded_belief_state = tiger_expand_belief_state_by_time(self.belief_state, self.timestep, self.horizon, ended=False)

    def select_action(self, obs):
        """
        Return an action according to the expanded belief state and the current policy
        represented in terms of alpha vectors.
        NOTE: Alpha vectors from SARSOP operates on the time-indexed (i.e., expanded) 
            state space, which is necessary to implement time-dependent 
            policy for finite horizon POMDP control. A time-independent policy is 
            highly likely sub-optimal in finite horizon.
        """
        expanded_belief_state = self.update_belief_state(obs, expand_by_time=True)
        # Save obs to memory
        self.current_trajectory['obs'] = jnp.append(self.current_trajectory['obs'], obs)
        
        # If `obs` > 1: a door has been chosen as `obs` is not HearLeft (0) or HearRight (1).
        #   The state will go to End (4) regardless of the final action. We thus 
        #   select OpenLeft (1) to avoid listen cost.
        # Get new belief state
        # Else: Select action based on the new belief state and alpha vectors
        action = self.policy(expanded_belief_state) if obs <= 1 else 1
        
        # Save action to memory
        self.current_trajectory['action'] = jnp.append(self.current_trajectory['action'], action)
        # Go to next timestep by performing the chosen action
        self.timestep += 1
        return action
    
    def step_episode(self, final_obs, update_policy=True):
        # Add final obs to memory
        obs_traj = jnp.append(self.current_trajectory['obs'], final_obs)
        self.current_trajectory['obs'] = obs_traj
        
        if self.mode == 'online':
            # Store current trajectory to memory
            self.past_trajectories[self.episode] = self.current_trajectory
            # Update training data for MCMC
            # NOTE: We only use the obs's that are Hear-Left (0) or Hear-Right (1)
            #   because only those data are relevant to the learnable parameter theta.
            t_open = jnp.argmax(obs_traj >= 2)
            t_open = t_open if obs_traj[t_open] >= 2 else obs_traj.shape[0]
            obs_traj = obs_traj[:t_open]
            self.train_data['obs'] = jnp.concat([self.train_data['obs'], obs_traj])
            self.train_data['ind_episodes'] = jnp.concat(
                [self.train_data['ind_episodes'], jnp.full_like(obs_traj, self.episode, dtype=int)])

            if update_policy:
                # Update posterior and policy
                self.update_posterior(self.train_data)
                self.posterior = self.mcmc.get_samples()
                self.planner_theta = self.sample_theta_from_posterior()
                self.update_policy(self.planner_theta)

        # Clean-up current trajectory
        self.current_trajectory = {'obs': jnp.empty(0), 'action': jnp.empty(0)}
        
        # Reset time counter and belief state
        self.timestep = 0
        self.reset_belief_state()
        
        # Go to next episode
        self.episode += 1

    def update_posterior(self, data):
        """Run MCMC (MixedHMC) for posterior sampling"""
        if self.mcmc is None:
            # Initialize MCMC
            kernel = MixedHMC(HMC(PS4Tiger.inference_model), random_walk=False, modified=False)
            self.mcmc = MCMC(kernel, **self.mcmc_config, progress_bar=False)
        self.prng_key, subkey = jr.split(self.prng_key)
        self.mcmc.run(subkey, **data, n_episodes=self.episode + 1)
        self.posterior = self.mcmc.get_samples()

    def update_policy(self, theta):
        # Update and return policy found by SARSOP with the given theta
        self.policy = tiger_find_policy(
            horizon=self.horizon,
            theta=theta,
            discount=self.discount,
            **self.reward_config,
            pomdp_path=self.pomdp_path,
            sarsop_config=self.sarsop_config
        )
        return self.policy
    
    def update_belief_state(self, obs, expand_by_time=True):
        self.belief_state = tiger_update_belief_state(self.belief_state, self.planner_theta, obs)
        if expand_by_time: 
            # Expand belief state with time-index
            t = self.timestep
            ended = (t >= self.horizon - 1) or (obs > 1)
            self.expanded_belief_state = tiger_expand_belief_state_by_time(self.belief_state, t, self.horizon, ended)
            return self.expanded_belief_state
        return self.belief_state


    @staticmethod
    def inference_model(obs, ind_episodes, n_episodes):
        """
        obs: flattened observations
        ind_episodes: indices of episodes associated with each observation
        """
        # Global prior
        # theta = Z("Hear-Right" | "Tiger-Right") = Z("Hear-Left" | "Tiger-Left") 
        theta = npyro.sample('theta', dist.TruncatedNormal(0.25, 0.5, low=MIN_THETA, high=MAX_THETA))
        
        # Episode-level prior
        with npyro.plate('episodes_plate', n_episodes):
            # "uniform" prior = initial state distribution
            states = npyro.sample('states', dist.Bernoulli(0.5))
        
        # Z("Hear-Right" | states[k], "Listen") for each episode k
        hear_right_probs = jnp.where(states == 1, 0.5 + theta, 0.5 - theta)

        # Likelihood
        with npyro.plate('obs_plate', obs.shape[0]):
            npyro.sample('obs', dist.Bernoulli(hear_right_probs[ind_episodes]), obs=obs)

    # Utilites
    def sample_theta_from_posterior(self, size=1):
        self.prng_key, subkey = jr.split(self.prng_key)
        posterior_samples = self.posterior['theta']
        # draw sample for MCMC-posterior uniformly at random
        planner_theta = jr.choice(subkey, posterior_samples, shape=(size,))
        return planner_theta if size > 1 else planner_theta.item()
    
    