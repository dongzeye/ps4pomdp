import os
# Set necessary environment variables
os.environ['JAX_PLATFORMS'] = 'cpu'  # Usually faster than GPU for MCMC in NumPyro (which uses JAX)
os.environ['JAX_ENABLE_X64'] = "True" # Double precision needed to avoid nan's in posterior sampling
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=4" # For running 4 MCMC chains in parallel

import sys
import argparse

import logging
from itertools import count
import importlib

from tqdm import tqdm
from pprint import pformat

import numpy as np

import gymnasium as gym

from pomdp_envs.finite_pomdp import FinitePOMDP
from agents.planners import SARSOPPlanner
from agents.ps4pomdp import PS4POMDP


from experiments.utils import (
    load_configuration,
    pickle_save,
    jr_key_generator,
    summarize_posterior,
    convert_to_mcmc_data,
)

# Constants
from constants import CONFIG_DIR

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Run experiment for PS4POMDP.")
    # Add arguments
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment to load configurations for.")
    parser.add_argument("--run_id", type=str, default="0", help="Run ID for identifying local configurations.")
    parser.add_argument("--max_episodes", type=int, default=None, help="Maximum number of episodes to run. Experiments will be truncated to `max_episodes`")
    parser.add_argument("--show_mcmc_pbar", action='store_true', help='Show mcmc progress bar')
    parser.add_argument("--verbose", action='store_true', help='Log all details of the experiment. [Only recommended for short expeirments]')

    # Parse arguments
    args = parser.parse_args()
    args.config_path = f'{CONFIG_DIR}/{args.exp_name}.json'
    return args


# Set up logging
def log_exception(exc_type, exc_value, exc_traceback):
    logging.error("Exception", exc_info=(exc_type, exc_value, exc_traceback))

def setup_logging(logfile):
    logging.basicConfig(
        level=logging.INFO,          # Set logging level
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        handlers=[
            logging.FileHandler(logfile, mode='w'),
            logging.StreamHandler(sys.stdout)  # Log to both console and file
        ]
    )
    sys.excepthook = log_exception

# Helpers
def build_infer_config(config):
    infer_config = config['infer_config']
    
    infer_module = importlib.import_module('agents.inference')
    prior = getattr(infer_module, infer_config['prior_name'])
    model = getattr(infer_module, infer_config['model_name'])   
    return infer_config | {'model': model, 'prior': prior}

def deploy_once(agent: PS4POMDP, env: FinitePOMDP, verbose: bool):  
    obs, _ = env.reset()
    agent.reset_belief_state()

    if verbose:
        state = env.unwrapped._state
        logging.info(f"Env reset to: s={state}, o={obs}")

    obs_history, action_history, reward_history = [obs], [], []
    for t in count():
        action = agent.select_action(obs, t)
        next_obs, reward, terminated, _ ,_ = env.step(action)
        
        obs_history.append(next_obs)
        action_history.append(action)
        reward_history.append(reward)
        
        if verbose:
            next_state = env.unwrapped._state
            logging.info(
                f"Time {t}: s={state}, o={obs}, a={action}, s'={next_state}, o'={next_obs}, r={reward}")
            state = next_state
        
        obs = next_obs
        if terminated:
            break
    
    traj = {
        'policy': agent.policy,
        'sampled_params': agent.sampled_params,
        'posterior_summary': summarize_posterior(agent.posterior, axis=0),
        'obs': obs_history, 
        'actions': action_history, 
        'rewards': reward_history,
    }
    return traj

# Main
if __name__ == '__main__':
    args = parse_args()
    verbose = args.verbose
    show_mcmc_pbar = args.show_mcmc_pbar

    config = load_configuration(args.config_path, args.run_id)

    os.makedirs(config['result_dir'], exist_ok=True)
    os.makedirs(config['tmp_dir'], exist_ok=True)

    setup_logging(config['log_path'])
    logging.info(f"Experiment config:\n{pformat(config)}")

    # Set seeds for reproducibility
    np.random.seed(config['np_seed'])
    key_gen = jr_key_generator(config['jr_seed'])

    fig_path = config['fig_path']
    result_path = config['result_path']  

    n_episodes = config['n_episodes'] 
    if args.max_episodes:
        n_episodes = min(args.max_episodes, n_episodes)

    env_config = config['env_config']
    planner_config = config['planner_config']
    infer_config = build_infer_config(config)
        
    env = gym.make(**env_config)
    _env = env.unwrapped
    known_params = {k: getattr(_env, k, _env.params.get(k)) for k in config['known_params']}
    internal_env = env.unwrapped.clone(remove_params=True)

    planner = SARSOPPlanner(**planner_config)
    agent = PS4POMDP(
        prng_key=next(key_gen),
        internal_env=internal_env,
        planner=planner,
        infer_config=infer_config,
        known_params=known_params,
    )

    # Trajectory container with all experimental results
    trajs = []

    logging.info("Setup completed.")
    logging.info("Starting the experiment.")
    
    for episode in range(n_episodes):
        logging.info(f"Starting Episode {episode}...")
        
        if episode == 0:
            agent.posterior = agent.sample_from_prior(5000) # set the initial posterior to prior samples
        else:
            data = convert_to_mcmc_data(trajs)
            agent.update_posterior(data, show_progress=show_mcmc_pbar, print_summary=verbose)
        sampled_params = agent.sample_from_posterior()
        agent.update_policy(new_params=sampled_params)
        
        logging.info(f"Agent sampled params:\n{pformat(sampled_params)}\n  Planner value: {agent.policy.value()}")

        traj = deploy_once(agent, env, verbose)
        trajs.append(traj)

        total_reward = sum(traj['rewards'])
        
        logging.info(f"Episode {episode} done: total_reward = {total_reward}")
        
        if (episode % 10 == 0) or (episode == n_episodes - 1):
            pickle_save(trajs, result_path)
        
            logging.info(f"Results saved: {result_path}")
    
    logging.info("Experiment completed successfully.")
