import os
# Set necessary environment variables
os.environ['JAX_PLATFORMS'] = 'cpu'  # Usually faster than GPU for MCMC in NumPyro (which uses JAX)
os.environ['JAX_ENABLE_X64'] = "True" # Double precision needed to avoid nan's in posterior sampling

import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from experiments.utils import pickle_load
from evals.utils import save_figure_verbose

# Constants
EXP_DIR = "./experiments/tiger"
EXP_NAMES = [f'tiger_theta{i}_discounted' for i in range(2, 5)] # + [<NEW_EXP>] 
LABELS = [rf'$\theta^\star = 0.{i}$' for i in range(2, 5)] # + [<LABEL_FOR_NEW_EXP>] 
LABEL_MAP = dict(zip(EXP_NAMES, LABELS))

FIGSIZE = (4, 3)

def parse_args():
    parser = argparse.ArgumentParser(description="Script to handle file paths and overwrite option.")
    parser.add_argument('--figure_dir', type=str, default="./figures/tiger", help="Directory for figures")
    parser.add_argument('--exp_name', type=str, default=None, help="Name of for the experiment for plotting")
    parser.add_argument('--load_data_from', type=str, default=None, help="Path to load existing plotting data")

    return parser.parse_args()

def load_plotting_data(exp_name, exp_dir):   
    # Process results and create plotting data
    with open(f'{exp_dir}/configs/{exp_name}.json', 'r') as f:
        configs = json.load(f)

    global_config = configs.pop('global')
    env_config = global_config['env_config']
    planner_discount = global_config['planner_discount']

    dfs = []
    # Load results for all runs under the same experiment name `exp_name`
    for run_id, local_config in configs.items():
        result_path = local_config['result_path']
        print(f"Processing {exp_name}-run{run_id}", end='\t')
        try:
            results = pickle_load(result_path)
        except FileNotFoundError: 
            print(f"Skipped {exp_name}-run{run_id}: Missing results at: {result_path}")
            continue
        results_timestamp = datetime.fromtimestamp(os.path.getmtime(result_path))
        print(f"(results timestamp: {results_timestamp})")

        posterior_theta_summary = pd.DataFrame.from_records(results['posterior_summary'])
        df = pd.DataFrame({
            'exp_name': exp_name,
            'theta': env_config['theta'],
            'discount': env_config['discount'],
            'planner_discount': planner_discount,
            'horizon': env_config['horizon'],
            'run': int(run_id),
            'episode': np.arange(len(results['policy_value_history'])),
            'planner_theta': results['planner_theta_history'],
            'posterior_mean_theta': posterior_theta_summary['mean'],
            'posterior_low_theta': posterior_theta_summary[0.1],
            'posterior_high_theta': posterior_theta_summary[0.9],
            'optimal_value': results['optimal_value'],
            'expected_return': results['policy_value_history'],
            'regret': results['per_episode_regrets'],
            'cum_regret': results['cum_regrets'],
        })
        dfs.append(df)

    if len(dfs) == 0:
        return None
    
    df = pd.concat(dfs, axis=0).sort_values(['exp_name', 'episode', 'run'], ignore_index=True)
    df['episode'] += 1
    df['cum_regret/K'] = df['cum_regret'] / (df['episode'])
    df['cum_regret/sqrtK'] = df['cum_regret'] / np.sqrt(df['episode'])
    return df

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.figure_dir, exist_ok=True)

    if args.load_data_from:
        df = pd.read_csv(args.load_data_from)
    else:
        # Load plotting data
        dfs = [
            df for exp_name in EXP_NAMES
            if (df := load_plotting_data(exp_name, EXP_DIR)) is not None
        ]
        assert len(dfs) > 0, "Abort: No results found"
        df = pd.concat(dfs, ignore_index=True)

        # Save a copy
        plotting_data_path = f'{args.figure_dir}/plotting_data.csv'
        df.to_csv(plotting_data_path, index=False)
        print(f'Saved plotting data: {plotting_data_path}')

    # Drop name of experiments for which no result has been found
    exp_names = list(df['exp_name'].unique())
    labels = [LABEL_MAP.get(name, 'missing label') for name in exp_names]
    print('Plotting results for:', exp_names)


    # Regret plot for Tiger (discounted)
    fig, ax = plt.subplots(figsize=FIGSIZE, layout='tight')
    sns.lineplot(df, x='episode', y='cum_regret', hue='exp_name', ax=ax)
    ax.set_ylabel('Regret')
    ax.set_xlabel('Episode $(K)$')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title=None)
    # ax.set_title("Regret for PS-POMDP on Tiger (Discounted)")
    save_figure_verbose(fig, f'{args.figure_dir}/regret_tiger.jpg')

    # Regret/K and /sqrtK plot combined for Tiger (discounted)
    fig, axs = plt.subplots(ncols=2, figsize=(1.5 * FIGSIZE[0], FIGSIZE[1]), layout='constrained', sharex=True)
    ax = axs[0]
    sns.lineplot(df, x='episode', y='cum_regret/K', hue='exp_name', ax=ax)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_ylabel('')
    ax.set_xlabel('Episode $(K)$')
    ax.set_title(r'Regret / $K$')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title=None)
    
    ax = axs[1]    
    ax = sns.lineplot(df, x='episode', y='cum_regret/sqrtK', hue='exp_name', ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel('Episode $(K)$')
    ax.set_title(r'Regret / $\sqrt{K}$')
    ax.get_legend().remove()
    
    fig.suptitle('Tiger ($H=10$)')
    save_figure_verbose(fig, f'{args.figure_dir}/regret_div_K_and_sqrtK_tiger.jpg')


    # Expected returns and sampled parameters across runs
    for exp_name in exp_names:
        df_plot = df[(df['exp_name'] == exp_name)]
        true_theta = df_plot['theta'].unique()[0]
        optimal_value = df_plot['optimal_value'].unique()[0]

        fig, axs = plt.subplots(ncols=2, figsize=(1.5 * FIGSIZE[0], FIGSIZE[1]), layout='tight', sharex=True)

        ax = axs[0]
        # Expected return
        sns.lineplot(df_plot, x='episode', y='expected_return', label="PS4POMDPs", ax=ax)
        ax.axhline(optimal_value, label=rf"Optimal", color='r', linewidth=0.8)
        # ax.set_title(rf"Expected Returns by Episode: $\theta^* = {true_theta}$ (first run only)")
        ax.set_xlabel('Episode')
        ax.set_ylabel('Expected Return')
        ax.legend()

        # Sampled params
        ax = axs[1]
        sns.lineplot(df_plot, x='episode', y='planner_theta', label=r'sampled', errorbar=('ci', 95), ax=ax)
        ax.axhline(y=true_theta, color='r', label=r'truth')
        ax.set_xlabel('Episode')
        ax.set_ylabel(r'$Parameter ($\theta$)')
        # ax.set_title(rf"Tiger ($\theta^* = {true_theta}$)")
        ax.legend()
    
        fig.suptitle(rf'Tiger ($H=10$, $\theta^\star = {true_theta}$)')
        save_figure_verbose(fig, f'{args.figure_dir}/expected_return_and_sampled_params_tiger_{exp_name}.jpg')

        # Expected return in log-scale
        fig, ax = plt.subplots(figsize=FIGSIZE, layout='tight')
        ax = sns.lineplot(df_plot, x='episode', y='expected_return', label="PS4POMDPs", ax=ax)
        ax.axhline(optimal_value, label=rf"Optimal", color='r', linewidth=0.8)
        # ax.set_title(rf"Expected Returns by Episode: $\theta^* = {true_theta}$ (first run only)")
        ax.set_xlabel('Episode')
        ax.set_ylabel(None)
        ax.set_xscale('log')
        ax.legend()
        save_figure_verbose(fig, f'{args.figure_dir}/expected_return_logscale_tiger_{exp_name}.jpg')



    
    # Posterior and expected return plots for first run of each experiment
    for exp_name in exp_names:
        df_plot = df[(df['exp_name'] == exp_name) & (df['run'] == 0)]
        true_theta = df_plot['theta'].unique()[0]
        optimal_value = df_plot['optimal_value'].unique()[0]
        
        # Sampled param
        fig, ax = plt.subplots(figsize=FIGSIZE, layout='tight')
        ax = sns.scatterplot(df_plot, x='episode', y='planner_theta', 
                             marker='x', linewidth=1, label=r'sampled')
        ax.fill_between(x=df_plot['episode'],
                        y1=df_plot['posterior_low_theta'],
                        y2=df_plot['posterior_high_theta'],
                        alpha=0.3,
                        color='orange',
                        label='95% CI')
        ax = sns.lineplot(
            df_plot, x='episode', y='posterior_mean_theta', color='orange', label='posterior mean', ax=ax)
        ax.axhline(y=true_theta, color='r', label=r'truth')
        ax.set_xlabel('Episode ($K$)')
        ax.set_ylabel(None)
        # ax.set_title(rf"Posterior sampling: $\theta^* = {true_theta}$ (first run only)")
        ax.legend()
        save_figure_verbose(fig, f'{args.figure_dir}/sampled_params_tiger_{exp_name}_run0.jpg')



        # Expected return
        fig, ax = plt.subplots(figsize=FIGSIZE, layout='tight')
        ax = sns.lineplot(df_plot, x='episode', y='expected_return', label="PS4POMDPs", ax=ax)
        ax.axhline(optimal_value, label=rf"Optimal", color='r', linewidth=0.8)
        # ax.set_title(rf"Expected Returns by Episode: $\theta^* = {true_theta}$ (first run only)")
        ax.set_xlabel('Episode ($K$)')
        ax.set_ylabel(None)
        ax.legend()
        save_figure_verbose(fig, f'{args.figure_dir}/expected_return_tiger_{exp_name}_run0.jpg')



