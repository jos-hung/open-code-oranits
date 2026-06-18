from physic_definition.system_base.ITS_based import Map, Task, Point, Mission, Vehicle
from decas import *
from graph import *
import json
import gymnasium as gym
import numpy as np
import ray
import random
import math
import torch
from ray import tune
# from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from gymnasium.spaces import Box
import numpy as np


SEED = 42


def decode_task(obj):
    if 'seg_tasksl' in obj:
        obj['seg_tasksl'] = [Task(*v) for k, v in obj['seg_tasksl'].items()]
    return obj
def decode_mission(obj):
    if 'depart_p' in obj:
        data = obj['depart_p']
        obj['depart_p'] = Point(data[0], data[1])
    if 'depart_s' in obj:
        data = obj['depart_s']
        obj['depart_s'] = Point(data[0], data[1])
    return obj

#inital map from file
map = Map(8, busy=2, fromfile=1)
map.draw_segments()
map.draw_map()
graph = Graph(map.get_segments())
# load offloading tasks and missions from file
with open('./task/task_information.json', 'r') as file:
    json_data = file.read()
decoded_data = json.loads(json_data, object_hook=decode_task)
segments = map.get_segments()
for item in decoded_data:
    seg_id = item['seg_id']
    idx = segments.index(seg_id)
    seg = segments[idx]
    seg.set_offloading_tasks(item['seg_tasksl'])
with open('./task/mission_information.json', 'r') as file:
    json_data = file.read()
decoded_data = json.loads(json_data, object_hook=decode_mission)

# Define the ITS environment
class ITSEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.data = config["data"]
        self.generator = np.random.default_rng(SEED)
        self.vehicles = self.init_vehicles()
        self.missions = self.init_missions()
        self.current_step = 0
        self.max_steps = 100
        self._agent_ids = set()
        self.action_space = Box(-np.inf, np.inf, shape=(data["n_vehicles"]-1,), dtype='float32')
        self.observation_space = Box(-np.inf, np.inf, shape=(data["n_missions"],6), dtype="float32")                    

    def init_missions(self):
        missions = []
        for item in self.data["decoded_data"]:
            m = Mission(item['depart_p'], item['depart_s'], 1, graph=self.data["graph"])
            m.set_depends(item["depends"])
            m.set_profit(item['profit'])
            missions.append(m)
        return missions

    def init_vehicles(self):
        vehicles = []
        for i in range(self.data["n_vehicles"]):
            seg = self.generator.choice(self.data["segments"])
            v = Vehicle(0.5, seg.get_endpoints()[0], map)
            vehicles.append(v)
        return vehicles

    def reset(self, *, seed=None, options=None):
        # super().reset(seed=seed)
        random.seed(seed)
        self.vehicles = self.init_vehicles()

        # Create a dependency graph to track mission dependencies
        self.dependency_graph = {}
        for mission in self.missions:
            mission_id = mission.get_mid()
            self.dependency_graph[mission_id] = mission.get_depends().copy()

        self._agent_ids = {f"vehicle_{i}" for i in range(self.data["n_vehicles"])}
        self.current_step = 0

        # Return initial observations with an empty info dict
        obs = self.get_observations()
        infos = {agent_id: {} for agent_id in self._agent_ids}  
        return obs, infos  

    def get_observations(self):
        sobs = {}

        # Prepare segment and mission information once outside the loop
        segment_info = np.array([[seg.get_long(), seg.get_state()] for seg in self.data["segments"]], dtype=np.float32)
        mission_lengths = np.array([[mis.get_long()[0]] for mis in self.missions], dtype=np.float32)
        num_missions_array = np.full((len(self.missions), 1), len(self.missions), dtype=np.float32)  # Repeat for each mission

        max_vehicle_positions_size = len(self.missions) * 2   #depend on data input
        
        # Pad segment_info to match mission_lengths
        if len(self.missions) > segment_info.shape[0]:
            padding = np.zeros(((len(self.missions) - segment_info.shape[0]), 2))
            segment_info = np.concatenate([segment_info, padding])
        
        # Pad vehicle_positions to match mission_lengths
        for i, v in enumerate(self.vehicles):
            # Vehicle positions (exclude self)
            vehicle_positions = []
            for other_v in self.vehicles:
                if other_v != v:
                    x, y = other_v.get_pos().get_point()
                    vehicle_positions.extend([x, y])
        
            # Padding vehicle_positions
            vehicle_positions.extend([0] * (max_vehicle_positions_size - len(vehicle_positions)))
            vehicle_positions = np.array(vehicle_positions, dtype=np.float32).reshape(-1, 2)

            # Ensure consistent number of rows across all arrays
            min_rows = min(segment_info.shape[0], mission_lengths.shape[0], vehicle_positions.shape[0], num_missions_array.shape[0])
            segment_info = segment_info[:min_rows]
            mission_lengths = mission_lengths[:min_rows]
            vehicle_positions = vehicle_positions[:min_rows]
            num_missions_array = num_missions_array[:min_rows]


            sobs[f"vehicle_{i}"] = np.concatenate([
                segment_info,
                mission_lengths,
                vehicle_positions,
                num_missions_array
            ], axis=1)

        return sobs
    

    def step(self, action_dict):
        rewards = {}
        total_complet_tasks = 0
        total_benef_tasks = 0
        # completed_mission_ids = []
        terminateds = {}  # Track episode termination for each agent
        truncateds = {}   # Track episode truncation for each agent
        infos = {}        # Per-agent info dictionaries
        for i, v in enumerate(self.vehicles):
            if f"vehicle_{i}" in action_dict:
                action = action_dict[f"vehicle_{i}"].astype('int32')
                v.set_mission(action, self.missions)
                
        while (1):
            terminate = True
            for idx, v in enumerate(self.vehicles):
                v.process_mission()
                if v.inprocess():
                    terminate = False
            if terminate:
                break
        for i, v in enumerate(self.vehicles):
            total_complet_tasks += v.get_earn_completes()
            total_benef_tasks += v.get_earn_profit()
            rewards[f"vehicle_{i}"] = v.get_vhicle_prof()
            # Determine if the episode is terminated or truncated for this vehicle
            terminateds[f"vehicle_{i}"] = self.current_step >= self.max_steps
            truncateds[f"vehicle_{i}"] = False  # Assuming no truncation in this example
            infos[f"vehicle_{i}"] = {}          # Empty info dict for this vehicle
        for agent_id in rewards:
            vehicle = self.vehicles[int(agent_id.split('_')[1])]
            print("Vehicle profit:", vehicle.get_vhicle_prof())
            print("Total completed tasks:", total_complet_tasks)
            print("Total benefit from tasks:", total_benef_tasks)
            # rewards[agent_id] = vehicle.get_vhicle_prof() + 0.5 * total_complet_tasks + 0.5 *  total_benef_tasks
            print("Reward for", agent_id, ":", rewards[agent_id])

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        terminateds = {f"vehicle_{i}": terminated for i in range(len(self.vehicles))}
        terminateds["__all__"] = terminated

       
        terminateds["__all__"] = all(terminateds.values())  
        truncateds["__all__"] = any(truncateds.values())

        obs = self.get_observations()
        return obs, rewards, terminateds, truncateds, infos

# Load data
with open('./task/task_information.json', 'r') as file:
    json_data = file.read()
decoded_task_data = json.loads(json_data, object_hook=decode_task)

with open('./task/mission_information.json', 'r') as file:
    json_data = file.read()
decoded_mission_data = json.loads(json_data, object_hook=decode_mission)

# Prepare data
data = {
    "n_missions": len(decoded_mission_data),
    "n_vehicles": 20,
    "n_miss_per_vec": 10,
    "decoded_data": decoded_mission_data,
    "segments": map.get_segments(),
    "graph": graph
}

# Register the environment
register_env("its_env", lambda config: ITSEnv(config))

# Define your custom model (replace with your actual model architecture)
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
import torch.nn as nn

class ClippedActorCriticNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.clip_value_min = model_config["custom_model_config"]["clip_value_min"]
        self.clip_value_max = model_config["custom_model_config"]["clip_value_max"]
        self.intermediate_layer = nn.Linear(19, 128)  # New layer depend on data_input

        self.actor = nn.Sequential(
            nn.Linear(1200, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, action_space.shape[0])
        )

        self.critic = nn.Sequential(
            nn.Linear(1200, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Value function output
        )
        # print(f"--------------------------------------------Action Space Shape: {self.action_space.shape}") 
        self.action_mean = nn.Linear(128, action_space.shape[0])  # For mean
        self.log_std = nn.Parameter(torch.zeros(1, action_space.shape[0]))  # For log std deviation

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"] 

        # print(f"Input shape: {obs.shape}")
        
        # Use the actor and critic modules directly
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        logits = torch.mean(logits, dim=1)
        x = self.actor(input_dict["obs_flat"])
        x = self.intermediate_layer(x)
        mean = self.action_mean(x)  
        log_std = self.log_std
        # Repeat log_std values to match the number of rows in mean
        log_std = np.tile(log_std.detach().cpu().numpy(), (mean.shape[0], 1))  
        # Clip and then apply the exponential
        clipped_log_std = torch.clamp(self.log_std, self.clip_value_min, self.clip_value_max) 
        scale = torch.exp(clipped_log_std) 
        # print(f"------------------------------------------------Actor Output Shape: {logits.shape}")
        self._value = value
        # Create a dictionary with the concatenated observations as a numpy array
        dist_input = np.concatenate([mean.detach().cpu().numpy(), log_std], axis=-1)
        return dist_input, state

    def value_function(self):
        return self._value
    
# Register the custom model
ModelCatalog.register_custom_model("clipped_actor_critic", ClippedActorCriticNetwork)
# RLlib configuration
config = {
    "env": "its_env",
    "env_config": {
        "data": data,
        "seed": SEED
    },
    "num_gpus": 1,
    "framework": "torch",
    "model": {
        "fcnet_hiddens": [128, 128],
        "fcnet_activation": "relu",
        "custom_model": "clipped_actor_critic",
        "custom_model_config": {
            "clip_value_min": 0,  # Adjust these values as needed
            "clip_value_max": 500   # Adjust these values as needed
        },
    },
    "lr": 1e-3,
    "train_batch_size": 64,
    "sgd_minibatch_size": 32,
    "num_sgd_iter": 10,
}

# Train the agent
ray.init(ignore_reinit_error=True)
tune.run(
    PPO,
    config=config,
    stop={"training_iteration": 100},
    checkpoint_freq=10,
    storage_path="~/ray_results/its_paper"
)

ray.shutdown()
