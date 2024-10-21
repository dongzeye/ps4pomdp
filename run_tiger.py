import os
# Set necessary environment variables
os.environ['JAX_PLATFORMS'] = 'cpu'  # Usually faster than GPU for MCMC in NumPyro (which uses JAX)
os.environ['JAX_ENABLE_X64'] = "True" # Double precision needed to avoid nan's in posterior sampling
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=5" # For running 5 MCMC chains in parallel

import sys
import argparse

from datetime import datetime
from itertools import count
from pprint import pprint

import random

import gymnasium as gym

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from evals.belief_mdp_eval import TigerStatisticsMDP
from agents.ps4tiger import PS4Tiger

from experiments.utils import (
    load_configuration,
    pickle_save,
    plot_eval_results,
    jr_key_generator,
    summary_statistics
)

# Constants
from constants import DASH_LINE, CONFIG_DIR

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Run experiment for PS4POMDP with Tiger.")
    # Add arguments
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment to load configurations for.")
    parser.add_argument("--run_id", type=str, default="0", help="Run ID for identifying local configurations.")
    parser.add_argument("--exp_dir", type=str, default='experiments/tiger', help="Main directory for the experiment.")
    parser.add_argument("--config_dir", type=str, default=None, help="Directory of the experiment configuration.")
    parser.add_argument("--max_episodes", type=int, default=None, help="Maximum number of episodes to run. Experiments will be truncated to `max_episodes`")
    parser.add_argument("--verbose", action='store_true', help='Print out all details for the experiment. [Only recommended for short expeirments]')

    # Parse arguments
    args = parser.parse_args()
    
    args.config_file = f'{CONFIG_DIR}/{args.exp_name}.json'
    return args

def evaluate_agent(
    env, agent, n_episodes, update_policy=True, save_policy=False, verbose=False, 
    plot=False, figsize=(14, 5), title=None, show_theta=True, round_decimals=4,
    fig_path=None, result_path=None, show_progress=True,
):
    assert agent.policy is not None, "Agent does not have a policy. Need to run `agent.reset()` first."
    if fig_path is None:
        fig_path = './tmp_eval_results.jpg'
    if verbose:
        print(f'Prior belief for s_{agent.timestep}:', agent.belief_state, 
            "\n  Time-indexed:", agent.expanded_belief_state, end=DASH_LINE)
    if not update_policy: 
        save_policy = False # Do not save policy when there's no policy udpate
    
    env_unwrapped = env.unwrapped

    return_history = []
    sarsop_value_history = []
    planner_theta_history = []
    posterior_theta_summaries = []
    policy_history = []
    # Keep a reference to each item to be stored as results
    results = {
        'timestamp': datetime.now(),
        'true_theta': env_unwrapped.theta,
        'return_history': return_history,
        'sarsop_value_history': sarsop_value_history, 
        'policy_history': policy_history,
        'planner_theta_history': planner_theta_history,
        'posterior_summary': posterior_theta_summaries,
        'trajectories': agent.past_trajectories # Trajectories are stored by agent as `dict`` with `i_episode` as keys
    }
    for i_episode in tqdm(range(n_episodes), miniters=1, disable=not show_progress):
        obs, _ = env.reset()
        cum_reward = 0
        for t in count():
            if verbose:
                print(f'Episode {i_episode}, at time {t}:  '
                      f'(true state s_{t} = {env_unwrapped.state}, '
                      f'true theta = {env_unwrapped.theta}, '
                      f'sampled theta = {round(agent.planner_theta, round_decimals)})')
                print(f'  - Observation o_{t} = {obs}')
            
            action = agent.select_action(obs)
            obs, reward, terminated, _ ,_ = env.step(action)
            cum_reward += reward
            
            if verbose:
                print(f'  - Belief for s_{t}: {agent.belief_state.round(round_decimals)}')
                print(f'  - Time-indexed belief:\n\t{agent.expanded_belief_state.round(round_decimals)}')
                print(f'  - Action a_{t} = {action}')
                print(f'  - Reward r_{t} = {reward}')
                print(f'  - Cum. Reward = {cum_reward}', end=DASH_LINE)

            if terminated:
                break
        agent.step_episode(final_obs=obs, update_policy=update_policy)
        sarsop_value_history.append(agent.policy.value()) # store policy value from the default initial state 
        return_history.append(cum_reward)
        planner_theta_history.append(agent.planner_theta)
        posterior_theta_summaries.append(summary_statistics(agent.posterior['theta']))
        if save_policy:
            policy_history.append(agent.policy)

        # Save results
        if (i_episode % 20 == 0) or (i_episode == n_episodes - 1):
            if plot:
                df = pd.DataFrame({
                    'episode': np.arange(i_episode + 1),
                    'return': return_history,
                    'planner_theta': planner_theta_history,
                    'mean_theta': [summary['mean'] for summary in posterior_theta_summaries],
                    'true_theta': env_unwrapped.theta,
                })
                plot_eval_results(df, fig_path, figsize, title, show_theta)
                plt.close()
            if result_path is not None:
                pickle_save(results, result_path)

    return results

def main():
    args = parse_args()
    start_time = datetime.now()
    print(f"({start_time})   Starting experiment <{args.exp_name}-run{args.run_id}>")
    
    # Load configurations
    config = load_configuration(args.config_file, args.run_id)
    print(DASH_LINE, "Experiment Configurations:", sep='')
    pprint(config)
    print(DASH_LINE)
    
    # Paths for saving results
    os.makedirs(config['result_dir'], exist_ok=True)
    os.makedirs(config['tmp_dir'], exist_ok=True)
    fig_path = config['fig_path']
    result_path = config['result_path']
    
    # Set random seeds for reproducibility
    main_seed = config['main_seed']
    random.seed(main_seed)
    np.random.seed(main_seed + 1)
    key_gen = jr_key_generator(seed=main_seed + 2)
    
    env_config = config['env_config']
    env = gym.make(**env_config)
    horizon = env.unwrapped.horizon
    env_discount = env.unwrapped.discount
    
    reward_config = {k: getattr(env.unwrapped, k) for k in ['listen_cost', 'treasure_reward', 'tiger_penalty']}
    planner_discount = config['planner_discount']
    if planner_discount != env_discount:
        print(f"NOTE: planner_discount = {planner_discount} != env_discount = {env_discount}")

    # Setup agent
    agent = PS4Tiger(
        prng_key=next(key_gen),
        horizon=horizon,
        reward_config=reward_config,
        discount=config['planner_discount'],
        mcmc_config=config['mcmc_config'],
        sarsop_config=config['sarsop_config'],
        pomdp_path=config['pomdp_path'],
    )
    agent.reset() # Reset to sample from prior and compute the first policy

    # Suppress progress bar if the output is stdout and not in verbose mode.
    show_progress = sys.stdout.isatty() and (not args.verbose)
    eval_plot_title = (rf"Evaluating PS4Tiger: {args.exp_name}-run{args.run_id}  ($H = {env_config['horizon']}$"
                       rf", $\theta^* = {env_config['theta']}$, $\gamma = {env_discount}$)")
    
    # Evaluate agent for `n_episodes` according to configl truncate to `max_episodes` from the commandline.
    n_episodes = config['n_episodes'] 
    if args.max_episodes:
        n_episodes = min(args.max_episodes, n_episodes)
    
    print(f'({datetime.now()})   Begin main (online) evaluation of the PS-POMDP agent for {n_episodes} episodes...')
    results = evaluate_agent(
        env, agent, n_episodes, update_policy=True, save_policy=True, verbose=args.verbose,
        plot=True, title=eval_plot_title, fig_path=fig_path, result_path=result_path,
        show_progress=show_progress
    )
    
    print(f'({datetime.now()})   Begin policy evaluation for the optimal policy and policies used in each episode...')

    agent.set_mode('offline')
    agent.sarsop_config.update(config['sarsop_optimal_policy_config'])
    agent.update_policy(theta=env_config['theta'])

    eval_mdp = TigerStatisticsMDP.from_exp_config(config)
    policy_value_history = eval_mdp.compute_policy_values(results['policy_history'])
    per_episode_regrets = eval_mdp.optimal_value - policy_value_history
    cum_regrets = np.array(per_episode_regrets).cumsum()

    results.update({
        'sarsop_optimal_policy': agent.policy,
        # SARSOP optimal value is given by max. of dot products between alphas and the initial belief [1, 0, 0, ...]
        'sarsop_optimal_value': agent.policy.alphas[:, 0].max().item(), 
        'optimal_policy': eval_mdp.optimal_policy,
        'optimal_value': eval_mdp.optimal_value,
        'optimal_value_fn': eval_mdp.optimal_value_fn,
        'policy_value_history': policy_value_history,
        'per_episode_regrets': per_episode_regrets,
        'cum_regrets': cum_regrets,
    })

    # save results
    pickle_save(results, result_path)
    print(f'All results saved: {result_path}')
    end_time = datetime.now()
    print(f"({end_time})   Experiment finished")
    print(f"Time elapsed: {(end_time - start_time)}")

if __name__ == "__main__":
    main()

 
