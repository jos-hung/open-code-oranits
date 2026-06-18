"""System-wide configuration defaults.

This module exposes low-level runtime defaults used across the project:
- device and random seed defaults
- logging configuration
- mission/task/map/network defaults
- DRL hyperparameters (e.g. `ddqn_cfg`)

Modules that need runtime constants import values from here. Changing values
will affect runs globally.
"""

DEVICE = 0
GLOBAL_SEED = 42

log_configs = {
    'log_dir': 'logs'
}

apply_thread = 0
apply_detach = 0
score_window_size = 100
tau = 60
mission_cfg = {
    'n_mission': 25,
    'benifits': [50,100],
    'n_vehicle': 5,
    'n_miss_per_vec': 5
}

ppo_cfg = {
    'benifits': [50,100],
    'update_frequency': 60,
    'save_dir': 'checkpoints/ppo_more_epoch',
    'max_eps_length': 100,
    'score_window_size': 100,
    'thread': apply_thread,
    'detach_thread': apply_detach,
    'type_': 'MAPPOTrainer'
}

    

task_cfg = {
    'comm_size':[100, 500], #kbytes
    'comp_size':[1,3], #mcycles
    'lambdas': [10, 30, 50], #tasks/second
    'vmax' : 10, #m/s
    'tau' : tau, #phut
    'cost_coefi' : 5*10**-5,
    'max_speed': 20 #m/s
    
}
def get_ideal_avg_reward():
    maximum_earned_path = task_cfg['max_speed']*task_cfg['tau']*60 #m
    avg_ben = (mission_cfg['benifits'][0]+mission_cfg['benifits'][1])/2
    length_path_avg = 2000 #m
    number_of_complete = maximum_earned_path/length_path_avg
    return number_of_complete*avg_ben + number_of_complete*100

avg_reward = get_ideal_avg_reward()

map_cfg = {
    'real_map': True,
    # 'real_center_point': (21.007837, 105.841819), #bkhn
    'real_center_point': (20.995417, 105.950051), #vin-university
    'radius': 2500,
    'n_lines': 15,
    'busy': 1,
    'fromfile': 1
}
network_cfg = {
    "n_MEC":20,
    "CPU_freq": [100, 300],#hz
    "CPU_satelite": 50, #hz
    "satelite_distance": 1000, #km
    "path_loss": 3,
    "channel_gain": "Gausian",
    "best_rate_radius": 100, #m
    "seed":42 
}
eval = False
ddqn_cfg = {
    "discount_factor":0.95,
    "learning_rate": 1e-5,
    "epsilon": 1.0,
    "epsilon_decay": 0.99,
    "epsilon_min": 0.05,
    "batch_size": 512,
    "maxlen_mem": 10000000,
    "modify_reward": True,
    "combine": 0.0
}
