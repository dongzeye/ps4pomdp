from functools import partial

import jax
import jax.numpy as jnp
import numpyro as npyro
import numpyro.distributions as dist

from pomdp_envs import RiverSwim

# The forward probability computations are partly inspired by the NumPyro HMM example
#   URL: https://num.pyro.ai/en/stable/examples/hmm.html
# 
# Notice that under a deterministic policy, the controlled POMDP is practically an HMM
# with time-dependent transitions

# @jax.jit
# def trajectory_loglik(obs_seq, action_seq, transit_probs, obs_probs, init_state_probs):
#     probs = init_state_probs
#     for o_t, a_t in zip(obs_seq[:-1], action_seq[:-1]):
#         probs = (obs_probs[:, o_t] * probs) @ transit_probs[:, a_t]
#     # now, log_probs[s_h] = P(s_h, o_1:h, a_1:h-1) ) where h = len(obs_seq)
#     loglik = jnp.log(probs @ obs_probs[:, obs_seq[-1]])
#     return loglik

@jax.jit
def trajectory_loglik(obs_seq, action_seq, transit_probs, obs_probs, init_state_probs):
    def scan_fn(probs, inputs):
        o_t, a_t = inputs
        return (
            (obs_probs[:, o_t] * probs) @ transit_probs[:, a_t],
            None,  # we don't need to collect during scan
        )
    probs, _ = jax.lax.scan(scan_fn, init_state_probs, xs=(obs_seq[:-1], action_seq[:-1]))
    loglik = jnp.log(probs @ obs_probs[:, obs_seq[-1]])
    return loglik


@jax.jit
def batch_trajectory_loglik(batch_obs, batch_actions, transit_probs, obs_probs, init_state_probs):
    # Process each episode and sum the log-likelihoods
    vmap_loglik_fn = jax.vmap(trajectory_loglik, in_axes=(0, 0, None, None, None))
    log_likelihoods = vmap_loglik_fn(
        batch_obs, batch_actions, transit_probs, obs_probs, init_state_probs)
    return jnp.sum(log_likelihoods)


@jax.jit
def trajectory_loglik_log_domain(obs_seq, action_seq, log_transit_probs, log_obs_probs, log_init_state_probs):
    def scan_fn(log_probs, inputs):
        o_t, a_t = inputs
        log_probs =  log_probs +  log_obs_probs[:, o_t]
        return (
            jax.nn.logsumexp(log_probs[..., None] + log_transit_probs[:, a_t], axis=0),
            None,  # we don't need to collect during scan
        )
    log_probs, _ = jax.lax.scan(scan_fn, log_init_state_probs, xs=(obs_seq[:-1], action_seq[:-1]))
    loglik = jax.nn.logsumexp(log_probs + log_obs_probs[:, obs_seq[-1]])
    return loglik


@jax.jit
def batch_trajectory_loglik_log_domain(batch_obs, batch_actions, log_transit_probs, log_obs_probs, log_init_state_probs):
    """
    obs: Observations matrix, shape = (n_episodes, horizon)
    actions: Actions matrix, shape = (n_episodes, horizon)
    log_transition_probs: Log of transition probability matrix, shape = (n_states, n_actions, n_states), with -inf for zero probabilities
    log_obs_probs: Log of observation probability matrix, shape = (n_states, n_obs), with -inf for zero probabilities
    log_init_state_dist: Log of initial state distribution, shape = (n_states,), with -inf for zero probabilities
    """
    log_likelihoods = jax.vmap(trajectory_loglik_log_domain, in_axes=(0, 0, None, None, None))(
        batch_obs, batch_actions, log_transit_probs, log_obs_probs, log_init_state_probs)
    return jnp.sum(log_likelihoods)


def generic_pomdp_log_likelihood(obs, actions, transition_kernel, observation_kernel, init_state_probs):
    """
    Add to NumPyro the log-likelihood of batched trajctories (`obs, actions`) under the specified parameters 
    Args:
        obs: observation data matrix, shape = (n_episodes, horizon + 1)
        actions: action data matrix, shape = (n_episodes, horizon)
        transition_kernel: shape = (n_states, n_actions, n_states)
        observation_kernel: shape = (n_states, n_obs)
        init_state_probs: shape = (n_states)    
    """
    # Manually compute the trajectory log-likelihood
    log_likelihood = batch_trajectory_loglik(
        obs, actions, transition_kernel, observation_kernel, init_state_probs)
    # Incorporate the log-likelihood into the NumPyro model
    npyro.factor("obs_loglik", log_likelihood)


def generic_pomdp_log_likelihood_log_domain(
    obs, actions, log_transition_kernel, log_observation_kernel, log_init_state_probs):
    """
    (Numerically stable version with logsumexp tricks) Add to NumPyro the log-likelihood of batched 
    trajctories (`obs, actions`) under the specified parameters 
    Args:
        obs: observation data matrix, shape = (n_episodes, horizon + 1)
        actions: action data matrix, shape = (n_episodes, horizon)
        log_transition_kernel: shape = (n_states, n_actions, n_states)
        log_observation_kernel: shape = (n_states, n_obs)
        log_init_state_probs: shape = (n_states)    
    """
    # Manually compute the trajectory log-likelihood
    log_likelihood = batch_trajectory_loglik_log_domain(
        obs, actions, log_transition_kernel, log_observation_kernel, log_init_state_probs)
    # Incorporate the log-likelihood into the NumPyro model
    npyro.factor("obs_loglik", log_likelihood)


def generic_transition_kernel_prior(n_states, n_actions, **kwargs):
    # Sample `(n_states, n_actions)` IID random prob vectors from 
    # the "uniform" Dirichlet prior over prob. vectors of dim `n_states`
    return npyro.sample(
        'transition_kernel', 
        dist.Dirichlet(jnp.ones(n_states)).expand((n_states, n_actions))
    )


def generic_observation_kernel_prior(n_states, n_obs, **kwargs):
    # Sample `n_states` IID random prob vectors from the "uniform" 
    # Dirichlet prior over prob. vectors of dim `n_states`
    return npyro.sample(
        'observation_kernel', 
        dist.Dirichlet(jnp.ones(n_obs)).expand((n_states,))
    )


def generic_init_state_probs_prior(n_states, **kwargs):
    return npyro.sample('init_state_probs', dist.Dirichlet(jnp.ones(n_states)))


def generic_pomdp_prior(n_states, n_actions, n_obs, **kwargs):    
    return (
        generic_transition_kernel_prior(n_states, n_actions), 
        generic_observation_kernel_prior(n_states, n_obs), 
        generic_init_state_probs_prior(n_states),
    )

def generic_pomdp_model(obs, actions, n_states, n_actions, n_obs, **kwargs):
    """
    Generic inference model for finite POMDP with "uniform" priors
    obs: observation data matrix, shape = (n_episodes, horizon + 1)
    actions: action data matrix, shape = (n_episodes, horizon)
    n_obs: size of observation space
    n_states: size of state space
    """
    transit_probs, obs_probs, init_state_probs = generic_pomdp_prior(n_states, n_actions, n_obs)
    generic_pomdp_log_likelihood(obs, actions, transit_probs, obs_probs, init_state_probs)



def generic_pomdp_prior_known_init_state(n_states, n_actions, n_obs, **kwargs):
    transit_probs = generic_transition_kernel_prior(n_states, n_actions)
    obs_probs = generic_observation_kernel_prior(n_states, n_obs)
    return transit_probs, obs_probs

def generic_pomdp_model_known_init_state(obs, actions, n_states, n_actions, n_obs, init_state_probs, **kwargs):
    """
    obs: observation data matrix, shape = (n_episodes, horizon)
    actions: action data matrix, shape = (n_episodes, horizon)
    n_obs: size of observation space
    n_states: size of state space
    init_state_probs: init state probability vector (assumed known)
    """
    transit_probs, obs_probs = generic_pomdp_prior_known_init_state(n_states, n_actions, n_obs)
    generic_pomdp_log_likelihood(obs, actions, transit_probs, obs_probs, init_state_probs)


def river_swim_pomdp_prior(**kwargs):
    # RiverSwim-POMDP is parameterized by two 3-dim. probability vectors
    transit_params = npyro.sample('transition_params', dist.Dirichlet(jnp.ones(3)))
    obs_params = npyro.sample('observation_params', dist.Dirichlet(jnp.ones(3)))
    return transit_params, obs_params


def river_swim_pomdp_model(obs, actions, n_states, n_obs, init_state_probs, **kwargs):
    """
    obs: observation data matrix, shape = (n_episodes, horizon)
    actions: action data matrix, shape = (n_episodes, horizon)
    n_obs: size of observation space
    n_states: size of state space
    init_state_probs: init state probability vector (assumed known)
    """
    transit_params, obs_params = river_swim_pomdp_prior()
    transit_probs = RiverSwim.build_transition_kernel_static(transit_params, n_states, n_actions=2)
    obs_probs = RiverSwim.build_observation_kernel_static(obs_params, n_obs, n_states)
    # jax.debug.print('transit_params: {}\n obs_params: {}', transit_params, obs_params)
    generic_pomdp_log_likelihood(obs, actions, transit_probs, obs_probs, init_state_probs)



def river_swim_pomdp_model_log_domain(obs, actions, n_states, n_obs, init_state_probs, **kwargs):
    """
    obs: observation data matrix, shape = (n_episodes, horizon)
    actions: action data matrix, shape = (n_episodes, horizon)
    n_obs: size of observation space
    n_states: size of state space
    init_state_probs: init state probability vector (assumed known)
    """
    transit_params, obs_params = river_swim_pomdp_prior()
    transit_probs = RiverSwim.build_transition_kernel_static(transit_params, n_states, n_actions=2)
    obs_probs = RiverSwim.build_observation_kernel_static(obs_params, n_obs, n_states)
    # jax.debug.print('transit_params: {}\n obs_params: {}', transit_params, obs_params)
    generic_pomdp_log_likelihood_log_domain(
        obs, actions, jnp.log(transit_probs), jnp.log(obs_probs), jnp.log(init_state_probs))


def river_swim_pomdp_prior_generic_transitions(n_states, **kwargs):
    transit_probs = generic_transition_kernel_prior(n_states, n_actions=2)
    obs_params = npyro.sample('observation_params', dist.Dirichlet(jnp.ones(3)))
    return transit_probs, obs_params


def river_swim_pomdp_model_generic_transitions(obs, actions, n_states, n_obs, init_state_probs, **kwargs):
    """
    obs: observation data matrix, shape = (n_episodes, horizon)
    actions: action data matrix, shape = (n_episodes, horizon)
    n_obs: size of observation space
    n_states: size of state space
    init_state_probs: init state probability vector (assumed known)
    """
    transit_probs, obs_params = river_swim_pomdp_prior_generic_transitions(n_states)
    obs_probs = RiverSwim.build_observation_kernel_static(obs_params, n_obs, n_states)
    # jax.debug.print('transit_params: {}\n obs_params: {}', transit_params, obs_params)
    generic_pomdp_log_likelihood(obs, actions, transit_probs, obs_probs, init_state_probs)


def randomT_pomdp_prior(n_states, **kwargs):
    transit_probs = generic_transition_kernel_prior(n_states, n_actions=2)
    obs_params = npyro.sample('observation_params', dist.Dirichlet(jnp.ones(3)))
    return transit_probs, obs_params


def randomT_pomdp_model(obs, actions, n_states, n_obs, init_state_probs, **kwargs):
    """
    obs: observation data matrix, shape = (n_episodes, horizon)
    actions: action data matrix, shape = (n_episodes, horizon)
    n_obs: size of observation space
    n_states: size of state space
    init_state_probs: init state probability vector (assumed known)
    """
    transit_probs, obs_params = randomT_pomdp_prior(n_states)
    obs_probs = RiverSwim.build_observation_kernel_static(obs_params, n_obs, n_states)
    # jax.debug.print('transit_params: {}\n obs_params: {}', transit_params, obs_params)
    generic_pomdp_log_likelihood(obs, actions, transit_probs, obs_probs, init_state_probs)


