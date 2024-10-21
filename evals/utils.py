# Helpers
import glob
import os

import pandas as pd
import numpy as np
import gymnasium as gym

from agents.planners import SARSOPPlanner
from experiments.utils import load_configuration, pickle_load

from constants import CONFIG_DIR


def save_figure_verbose(fig, path):
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f'Saved figure: {path}')

def load_all_results(config, n_runs=None):   
    result_dir = config['result_dir'] 
    if n_runs is None: 
        n_runs = config['n_runs']
    
    results = []

    # Loop over the number of runs and load files
    for i in range(n_runs):
        pattern = os.path.join(result_dir, f"run{i}_trajectories.pkl")
        files = glob.glob(pattern)
        
        if files:
            file_path = files[0]  # Assuming one file per pattern
            print(f"Loading file: {file_path}")
            
            data = pickle_load(file_path)
            
            results.append(data)
        else:
            print(f"No file found for pattern: {pattern}")

    # collect trajectory data into df
    df_raw = [
        pd.DataFrame({'run': run, 
                    'episode': episode, 
                    'time': len(traj['obs']),
                    'obs': traj['obs'],
                    'actions': np.append(traj['actions'], np.nan),
                    'rewards': np.append(traj['rewards'], np.nan)
                    }) 
        for run, trajs in enumerate(results) 
        for episode, traj in enumerate(trajs)
    ]
    df_raw = pd.concat(df_raw, axis=0)
    
    # compute optimal value
    env = gym.make(**config['env_config'])
    planner = SARSOPPlanner(**config['planner_config'])
    opt_policy = planner.solve(env.unwrapped)
    opt_value = opt_policy.value().item()
    # compute regret
    df_values = df_raw.groupby(['run', 'episode']).agg({'rewards': 'sum'})\
        .rename(columns={'rewards': 'return'})
    df_values['episode'] = df_values['episode'] + 1 
    df_values['regret'] = opt_value - df_values['return']
    df_values['cum_regret'] = df_values.groupby('run')['regret'].cumsum()
    df_values['cum_regret/K'] = df_values['cum_regret']  / df_values['episode']
    df_values['cum_regret/sqrtK'] = df_values['cum_regret']  / np.sqrt(df_values['episode'])
    
    return results, df_raw, df_values, opt_value


def convert_to_df_sampled_params(trajs, param_names):
    sampled_param_history = {
        name: np.array([traj['sampled_params'][name] for traj in trajs])
        for name in param_names
    }
    
    df = pd.DataFrame({
        f'{name}[{i}]': samples 
        for name in param_names 
        for i, samples in enumerate(sampled_param_history[name].T)
    })

    return df.rename_axis('episode')