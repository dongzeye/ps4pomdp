import os

from experiments.utils import load_configuration
# Set necessary environment variables
os.environ['JAX_PLATFORMS'] = 'cpu'  # Usually faster than GPU for MCMC in NumPyro (which uses JAX)
os.environ['JAX_ENABLE_X64'] = "True" # Double precision needed to avoid nan's in posterior sampling

import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import seaborn as sns
from evals.utils import load_all_results, save_figure_verbose


# Constants
from constants import CONFIG_DIR
FIGSIZE = (12, 3.2)
FIG_DIR = './figures'

if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)

    df_tiger = pd.read_csv('figures/tiger/plotting_data.csv')
    df_randpomdp = pd.read_csv('figures/random_pomdp/plotting_data.csv')
    df_riverswim = pd.read_csv('figures/river_swim/plotting_data.csv')
    df_riverswim_sampled_params = pd.read_csv('figures/river_swim/sampled_params_data.csv')

    # RiverSwim
    x = 'episode'
    xlabel = 'Episode $(K)$'
    hue = 'agent'
    df = df_riverswim.copy()

    fig, axs = plt.subplots(ncols=4, figsize=FIGSIZE, layout='constrained')

    ax = axs[0]
    sns.lineplot(df, x=x, y='expected_return', hue=hue, ax=ax)
    opt_value = df['opt_value'].unique().mean()
    ax.axhline(opt_value, label='Near-Optimal Policy', color='r', linewidth=0.8)
    ax.set_ylabel('')
    ax.set_xlabel(xlabel)
    ax.set_title('Expected Return')
    ax.get_legend().remove()

    ax = axs[1]
    sns.lineplot(df, x=x, y='cum_regret', hue=hue, ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel(xlabel)
    ax.set_title('Regret')
    ax.get_legend().remove()

    ax = axs[2]
    sns.lineplot(df, x=x, y='cum_regret/K', hue=hue, ax=ax)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_ylabel('')
    ax.set_xlabel(xlabel)
    ax.set_title(r'Regret / $K$')
    ax.get_legend().remove()

    ax = axs[3]
    sns.lineplot(df, x=x, y='cum_regret/sqrtK', hue=hue, ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel(xlabel)
    ax.set_title(r'Regret / $\sqrt{K}$')
    ax.get_legend().remove()

    handles, labels = axs[0].get_legend_handles_labels()
    handles = [Line2D([0], [0], color='none')] + handles
    labels = ['Agent:'] + labels

    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.01), ncol=6, frameon=False)
    fig.suptitle(r'RiverSwim ($H=40, S=O=6$)')
    save_figure_verbose(fig, f'{FIG_DIR}/results_river_swim.jpg')


    ## Sampled Params for RiverSwim-KnownStruct
    config_path = f'{CONFIG_DIR}/river_swim_hard.json'
    config = load_configuration(config_path)
    df_riverswim_sampled_params = df_riverswim_sampled_params.set_index(['run', 'episode'])
    param_names = ['transition_params', 'observation_params']
    true_params = np.array([config['env_config']['params'][name] for name in param_names]).flatten()
    labels = ['$p_1$', '$p_2$', '$p_3$', '$q_1$', '$q_2$', '$q_3$']

    fig, axs = plt.subplots(ncols=3, nrows=2, layout='tight')
    axs = axs.ravel()  # Flatten the array to easily iterate over it    
    for ax, col_name, label, true_param in zip(axs, df_riverswim_sampled_params.columns, labels, true_params):
        sns.lineplot(df_riverswim_sampled_params, x='episode', y=col_name, ax=ax, label='sampled')
        ax.axhline(y=true_param, label='truth', color='red')
        ax.set_ylabel(None)
        ax.set_xlabel(None)
        # ax.set_ylim(true_param - 0.1, true_param + 0.1)
        ax.set_title(label)
        ax.get_legend().remove()

    axs[0].legend()
    axs[0].set_xlabel('Episode')
    fig.suptitle('Sampled Parameters for PS4POMDPs-KnownStruct in RiverSwim')
    save_figure_verbose(fig, f'{FIG_DIR}/sampled_params_river_swim.jpg')

    # Random (Sparse Reward) POMDPs
    x = 'episode'
    xlabel = 'Episode $(K)$'
    hue = 'env'

    df = df_randpomdp.copy()

    fig, axs = plt.subplots(ncols=4, figsize=FIGSIZE, layout='constrained')

    ax = axs[0]
    sns.lineplot(df, x=x, y='expected_return', hue=hue, ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel(xlabel)
    ax.set_title('Expected Return')
    ax.get_legend().remove()

    ax = axs[1]
    sns.lineplot(df, x=x, y='cum_regret', hue=hue, ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel(xlabel)
    ax.set_title('Regret')
    ax.get_legend().remove()

    ax = axs[2]
    sns.lineplot(df, x=x, y='cum_regret/K', hue=hue, ax=ax)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_ylabel('')
    ax.set_xlabel(xlabel)
    ax.set_title(r'Regret / $K$')
    ax.get_legend().remove()

    ax = axs[3]
    sns.lineplot(df, x=x, y='cum_regret/sqrtK', hue=hue, ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel(xlabel)
    ax.set_title(r'Regret / $\sqrt{K}$')
    ax.get_legend().remove()

    handles, labels = axs[0].get_legend_handles_labels()
    handles = [Line2D([0], [0], color='none')] + handles
    labels = ['Environment: '] + labels
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.01), ncol=6, frameon=False)

    fig.suptitle(r'RandomPOMDP ($H=20, S=10, A=4, O=10$)')
    save_figure_verbose(fig, f'{FIG_DIR}/results_random_pomdp.jpg')

    # Tiger
    x = 'episode'
    xlabel = 'Episode $(K)$'

    df = df_tiger.copy()
    df['label'] = df_tiger['theta'].map(lambda s: rf'$\theta^\star = {s}$')

    fig, axs = plt.subplots(ncols=4, figsize=FIGSIZE, layout='constrained')

    ax = axs[0]
    sns.lineplot(df, x=x, y='expected_return', hue='label', ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel(xlabel)
    ax.set_title('Expected Return')
    ax.get_legend().remove()

    ax = axs[1]
    sns.lineplot(df, x=x, y='cum_regret', hue='label', ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel(xlabel)
    ax.set_title('Regret')
    ax.get_legend().remove()

    ax = axs[2]
    sns.lineplot(df, x=x, y='cum_regret/K', hue='label', ax=ax)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_ylabel('')
    ax.set_xlabel(xlabel)
    ax.set_title(r'Regret / $K$')
    ax.get_legend().remove()

    ax = axs[3]
    sns.lineplot(df, x=x, y='cum_regret/sqrtK', hue='label', ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel(xlabel)
    ax.set_title(r'Regret / $\sqrt{K}$')
    ax.get_legend().remove()

    handles, labels = axs[0].get_legend_handles_labels()
    handles = [Line2D([0], [0], color='none')] + handles
    labels = ['True Parameter:'] + labels
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=False)

    fig.suptitle(r'Tiger ($H=10, \beta=0.99$)')
    save_figure_verbose(fig, f'{FIG_DIR}/results_tiger.jpg')