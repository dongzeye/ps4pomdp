import jax.numpy as jnp
import jax.random as jr

from numpyro.infer import NUTS, MCMC, Predictive, init_to_uniform

from agents.planners import POMDPPlanner
from pomdp_envs.finite_pomdp import FinitePOMDP


class PS4POMDP(object):
    def __init__(
        self, prng_key, internal_env: FinitePOMDP, planner: POMDPPlanner, 
        infer_config, known_params=None, 
    ) -> None:
        self.prng_key = prng_key
        self.planner = planner 
        self.infer_config = infer_config

        self.known_params = known_params if known_params else {}

        # Internal env configurations
        self.internal_env = internal_env
        self.internal_env.set_params(self.known_params)

        self.horizon = self.internal_env.horizon
        self.n_actions = self.internal_env.n_actions
        self.n_states = self.internal_env.n_states
        self.n_obs = self.internal_env.n_obs

        # Belief state: posterior dist. of state given obs in current traj. and sampled theta
        self.belief_state_shape = (self.n_states,)
        # Unnormalized_belief_state ~ belief_state up to a positive factor 
        # (does not affect action selection)
        self.unnorm_belief_state = None


        # MCMC for posterior inference
        self.posterior = {} # Dict of posterior samples of the POMDP params from MCMC
        self.sampled_params = {} # a random posterior sample of the POMDP params
        self.policy = None # "Conditionally-optimal" policy given the sampled params
        
        # Counters
        self.episode = None
        self.timestep = None
        
        # Data containers
        self.past_trajectories = None
        self.current_trajectory = None
        self.train_data = None

    def set_params(self, params):
        self.sampled_params = self.sampled_params | params
        self.internal_env.set_params(params)

    def sample_from_prior(self, n_samples=1):
        self.prng_key, subkey = jr.split(self.prng_key)
        sampler = Predictive(self.infer_config['prior'], num_samples=n_samples)
        sampled_params = sampler(subkey, **self.known_params)
        return {k: jnp.squeeze(v) for k, v in sampled_params.items()}
    
    def reset(self):
        self.posterior = {}
        self.set_params(self.sample_from_prior()) # Reset params to a prior sample
        self.policy = self.update_policy() # reset policy to the new params
        self.reset_belief_state()

    def reset_belief_state(self):
        self.unnorm_belief_state = self.internal_env.init_state_probs

    def set_policy(self, policy, params):
        # params needs to be given for beleif state updates
        self.set_params(params)
        self.policy = policy
                
    def update_posterior(self, data, print_summary=False, show_progress=False):
        """
        Update posterior through MCMC. 
        data: dict with all inputs to the inference model (`self.infer_config['model']`) 
        """
        self.prng_key, subkey = jr.split(self.prng_key)
        
        # Setup MCMC
        kernel = NUTS(self.infer_config['model'])
        mcmc = MCMC(kernel, **self.infer_config['mcmc_args'], progress_bar=show_progress)
        mcmc.run(subkey, **data, **self.known_params)
        if print_summary:
            print('\nPosterior summary:')
            mcmc.print_summary()

        # Store and return the posterior
        self.posterior = self.posterior | mcmc.get_samples()
        return self.posterior
    
    def sample_from_posterior(self, n_samples=1):
        self.prng_key, subkey = jr.split(self.prng_key)

        # Get the sample size of the posterior
        n_samples = next(iter(self.posterior.values())).shape[0]

        # Get a random index from [0, n_samples]
        idx = jr.randint(subkey, (1,), 0, n_samples + 1).item()
        sampled_params = {k: v[idx] for k, v in self.posterior.items()}

        return sampled_params

    def update_policy(self, new_params=None):
        if new_params is not None:
            self.set_params(new_params)

        self.policy = self.planner.solve(self.internal_env)
        return self.policy
    
    def select_action(self, obs, timestep):
        obs_probs = self.internal_env.obs_probs
        transit_probs = self.internal_env.transit_probs

        # Prior to seeing obs, unnormalized_belief_state[s_t] = P(s_t, o_{1:t-1}, a_{1:t-1})
        self.unnorm_belief_state = self.unnorm_belief_state * obs_probs[:, obs]
        # Now, unnormalized_belief_state[s_t] = P(s_t, o_{1:t}, a_{1:t-1})
        
        # Select action
        action = self.policy(self.unnorm_belief_state, timestep)

        # Update unnormalized belief with the new action
        self.unnorm_belief_state = self.unnorm_belief_state @ transit_probs[:, action, :]
        # Now, unnormalized_belief_state[s_t+1] = P(s_t+1, o_{1:t}, a_{1:t})
    
        return action
