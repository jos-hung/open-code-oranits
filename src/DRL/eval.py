"""Evaluation utilities for trained DRL agents (A2C example).

This module provides a small helper `test_model` that loads a trained A2C
checkpoint, runs a number of evaluation episodes using `SITSEnv`, and writes
basic summary CSV and a bar-plot of average reward. The module also contains a
`__main__` block that constructs a test config and runs `test_model`.

This file performs I/O and evaluation; comments added here are non-functional
and do not change evaluation behavior.
"""

import os
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from DRL.maa2c.A2C import A2C
from DRL.rl_env import *
from utils import Load
from configs.systemcfg import log_configs, verbose, mission_cfg, map_cfg, DEVICE
import gymnasium as gym

import matplotlib.pyplot as plt

device = torch.device('cuda:'+str(DEVICE) if torch.cuda.is_available() else 'cpu')
if device == "cpu":
    print("cannot train with cpu")
    exit(0)
else:
    print("cuda: ", device)


# Load the trained model and test it
def test_model(checkpoint_path, env_config, eval_episodes=10, output_dir="./test_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load environment
    env = SITSEnv(env_config, verbose=verbose, map__=env_config['map'])
    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.shape[0]

    # Initialize the A2C agent
    a2c = A2C(env=env, state_dim=state_size, action_dim=action_size, 
              batch_size=256, entropy_reg=0.01, done_penalty=-1, 
              roll_out_n_steps=30, reward_gamma=0.95, max_grad_norm=None)
    
    # Load the trained model checkpoint
    print(f"Loading model from {checkpoint_path}...")
    a2c.load_model(checkpoint_path)
    
    # Evaluate the model
    rewards, _ = a2c.evaluation(env, eval_episodes)
    rewards_mu, rewards_std = np.mean(rewards), np.std(rewards)
    print(f"Evaluation Results: Average Reward: {rewards_mu:.2f}, Std: {rewards_std:.2f}")
    
    # Save evaluation results
    results = {
        "Average Reward": [rewards_mu],
        "Std Dev Reward": [rewards_std]
    }
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/evaluation_results.csv", index=False)
    
    # Plot evaluation results
    plt.figure()
    plt.bar(["Average Reward"], [rewards_mu], yerr=[rewards_std], capsize=5)
    plt.ylabel("Reward")
    plt.title("Evaluation Results")
    plt.savefig(f"{output_dir}/evaluation_plot.png")
    plt.close()
    
if __name__ == "__main__":
    # Define testing parameters
    checkpoint_num = 15000
    current_datetime = datetime.now().strftime("%Y%m%d")
    checkpoint_path = f"./output_{current_datetime}/checkpoints/check_point_{checkpoint_num}.pth"
    output_dir = f"./test_results_{current_datetime}"

    # Load and configure the environment
    load = Load()
    mission_decoded_data, graph, map_ = load.data_load()
    config = {
        "n_missions": mission_cfg['n_mission'],
        "n_vehicles": mission_cfg['n_vehicle'],
        "n_miss_per_vec": mission_cfg['n_miss_per_vec'],
        "decoded_data": mission_decoded_data,
        "segments": map_.get_segments(),
        "graph": graph,
        "thread": 1,
        "detach_thread": 0,
        "score_window_size": 100,
        "tau": 10 * 6,  # min
        "map": map_
    }
    
    # Run the test
    test_model(checkpoint_path, config, eval_episodes=10, output_dir=output_dir)