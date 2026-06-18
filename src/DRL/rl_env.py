"""Gym environment wrappers for the ITS simulation.

This module defines two gym-compatible environments:
- ITSEnv: multi-agent style observations (dict of per-vehicle arrays)
- SITSEnv: single-agent environment used for A2C evaluation

Environments build missions/vehicles from `write_config`/`Load` utilities and
convert domain state into large flattened observation vectors. The code below
only contains comments in addition to the original logic; no runtime behavior
is changed.
"""

from physic_definition.system_base.ITS_based import Map, Task, Point, Mission, Vehicle, TaskGenerator
from physic_definition.map.decas import *
from physic_definition.map.graph import *
import json
import gymnasium as gym
import numpy as np
# from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from gymnasium.spaces import Box
import numpy as np
from configs.systemcfg import map_cfg, mission_cfg, avg_reward, eval
from configs.config import ParaConfig
from utils import Load, write_config, write_config_not_fromfile
import threading
import copy

SEED = ParaConfig.SEED_GLOBAL

# Define the ITS environment
class ITSEnv(gym.Env):
    def __init__(self, data, verbose = True, map__ = None, generator = None, max_steps = 100):
        super().__init__()
        self.data = data
        self.verbose = verbose
        self.generator = np.random.default_rng(SEED)
        self.lmap = map__
        self.vehicles = self.init_vehicles()
        self.missions = self.init_missions()
        self.current_step = 0
        self.max_steps = max_steps
        self._agent_ids = set()
        self.action_space = Box(-np.inf, np.inf, shape=(data["n_missions"],), dtype='float32')
        
        self.observation_space = Box(-np.inf, np.inf, shape=(5160,1), dtype="float32")                    
        self.action_memory = np.array([0]*data["n_missions"])
        self.solution =  ['None']*(mission_cfg['n_vehicle']*mission_cfg['n_miss_per_vec'])
        self.max_selection_turn = [self.data['n_miss_per_vec']]*self.data['n_vehicles']
        self.done = True 
        if generator==None:
            self.tg = TaskGenerator(15,self.lmap)
        else:
            self.tg = generator
        
    def init_missions(self):
        missions = []
        lock = threading.Lock()  # For thread-safe access to the missions list
        def create_mission(item):
            m = Mission(item['depart_p'], item['depart_s'], 1, graph=self.data["graph"], verbose=self.verbose)
            m.set_depends(item["depends"])
            m.register_observer(self.vehicles)          
            with lock:
                missions.append(m)

        threads = []
        for item in self.data["decoded_data"]:
            # Create a thread for each mission
            t = threading.Thread(target=create_mission, args=(item,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        Mission.missionID=0
        return missions  

    def init_vehicles(self):
        vehicles = []
        for i in range(self.data["n_vehicles"]):
            seg = self.generator.choice(self.data["segments"])
            v = Vehicle(0.5, seg.get_endpoints()[0], self.lmap, verbose=self.verbose, tau=self.data['tau'])
            vehicles.append(v)
        v.reset()
        return vehicles

    def reset(self, rfile=True, predict = False):
        
        import time
        start = time.perf_counter()        
        if (self.done and rfile and not eval) or predict:
            if self.verbose:
                print("---------> reset", self.done)
            self.data = write_config_not_fromfile(self.tg)
            self.current_step = 0
            
        file_reset = time.perf_counter()
        
        self.vehicles = self.init_vehicles()
        if 'missions' in self.data:
            self.missions = copy.deepcopy(self.data['missions'])
        else:
            self.missions = self.init_missions()
        self._agent_ids = {f"vehicle_{i}" for i in range(self.data["n_vehicles"])}
        # Return initial observations with an empty info dict
        self.action_memory = np.array([0]*self.data["n_missions"])
        self.solution =  ['None']*(mission_cfg['n_vehicle']*mission_cfg['n_miss_per_vec'])
        obs = self.get_observations()
        infos = {agent_id: {} for agent_id in self._agent_ids}  
        end = time.perf_counter()
        self.max_selection_turn = [self.data['n_miss_per_vec']]*self.data['n_vehicles']
        
        print("genfile {}, reset_object {}".format(file_reset-start, end-file_reset))
        return obs, infos 
    
    def reset_for_meta(self, rfile=True, predict = False):
        
        import time
        start = time.perf_counter()
        
        if (rfile ) or predict:
            if self.verbose:
                print("---------> reset", self.done)
            self.data = write_config_not_fromfile(self.tg)
            self.current_step = 0
            
        file_reset = time.perf_counter()
        
        self.vehicles = self.init_vehicles()
        if 'missions' in self.data:
            self.missions = copy.deepcopy(self.data['missions'])
        else:
            self.missions = self.init_missions()
        self._agent_ids = {f"vehicle_{i}" for i in range(self.data["n_vehicles"])}
        # Return initial observations with an empty info dict
        self.action_memory = np.array([0]*self.data["n_missions"])
        self.solution =  ['None']*(mission_cfg['n_vehicle']*mission_cfg['n_miss_per_vec'])
        obs = self.get_observations()
        infos = {agent_id: {} for agent_id in self._agent_ids}  
        end = time.perf_counter()
        self.max_selection_turn = [self.data['n_miss_per_vec']]*self.data['n_vehicles']
        
        print("genfile {}, reset_object {}".format(file_reset-start, end-file_reset))
        return obs, infos, self.data

    def get_observations(self):
        sobs = {}

        # Prepare segment and mission information once outside the loop
        segment_info = np.array([[seg.get_long(), seg.get_state()] for seg in self.data["segments"]], dtype=np.float32)
        mission_lengths = np.array([[mis.get_long()[0]] for mis in self.missions], dtype=np.float32)
        num_missions_depends_array = np.zeros((self.data['n_missions'], 10)) #maximum 10 depends
        for idx, mis in enumerate(self.missions):
            depends = mis.get_depends()
            num_missions_depends_array[idx][0:len(depends)] = depends
        num_missions_depends_array = np.array(num_missions_depends_array)  # Repeat for each mission

        max_vehicle_positions_size = len(self.missions) * 2   #depend on data input
        
        # Pad segment_info to match mission_lengths
        if len(self.missions) > segment_info.shape[0]:
            padding = np.zeros(((len(self.missions) - segment_info.shape[0]), 2))
            segment_info = np.concatenate([segment_info, padding])
        
        # Pad vehicle_positions to match mission_lengths
        vhicle_missions_depends_array = np.zeros((self.data['n_vehicles'], self.data['n_miss_per_vec'], 10)) #maximum 10 depends
        for i, v in enumerate(self.vehicles):
            waiting_mission = v.get_accepted_missions()
            for mis_idx, mis in enumerate(waiting_mission):
                vhicle_missions_depends_array[i][mis_idx][0:len(mis.get_depends())] = mis.get_depends()

        vhicle_missions_depends_array = vhicle_missions_depends_array.flatten()
        
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

            action_memory = np.array(self.action_memory).flatten()
            segment_info = np.reshape(segment_info, (-1))/5000
            mission_lengths = np.reshape(mission_lengths, (-1))/(5000*sqrt(2))
            vehicle_positions = np.reshape(vehicle_positions, (-1))/5000
            num_missions_depends_array = np.reshape(num_missions_depends_array, (-1))
            
            self.idx_dict_obs = {'segment_info': 0}
            self.idx_dict_obs['mission_lengths']= len(segment_info)
            self.idx_dict_obs['vehicle_positions'] = len(mission_lengths)
            self.idx_dict_obs['num_missions_depends_array'] = len(vehicle_positions)
            self.idx_dict_obs['action_memory'] = len(action_memory)
            self.idx_dict_obs['vhicle_missions_depends_array'] = len(num_missions_depends_array)
            max_len = max(len(segment_info), len(mission_lengths), len(vehicle_positions), 
              len(num_missions_depends_array), len(action_memory), len(vhicle_missions_depends_array))

            segment_info_padded = np.pad(segment_info, (0, max_len - len(segment_info)), mode='constant', constant_values=0)
            mission_lengths_padded = np.pad(mission_lengths, (0, max_len - len(mission_lengths)), mode='constant', constant_values=0)
            vehicle_positions_padded = np.pad(vehicle_positions, (0, max_len - len(vehicle_positions)), mode='constant', constant_values=0)
            num_missions_depends_array_padded = np.pad(num_missions_depends_array, (0, max_len - len(num_missions_depends_array)), mode='constant', constant_values=0)
            action_memory_padded = np.pad(action_memory, (0, max_len - len(action_memory)), mode='constant', constant_values=0)
            vhicle_missions_depends_array_padded = np.pad(vhicle_missions_depends_array, (0, max_len - len(vhicle_missions_depends_array)), mode='constant', constant_values=0)

            sobs[f"vehicle_{i}"] = np.concatenate([
                segment_info_padded,
                mission_lengths_padded,
                vehicle_positions_padded,
                num_missions_depends_array_padded,
                action_memory_padded,
                vhicle_missions_depends_array_padded
            ])            
        return sobs
    
    def get_ma_observations(self, depend_index = [], move_vehicle_pos=False):
        sobs = {}

        # Prepare segment and mission information once outside the loop
        segment_info = np.array([[seg.get_long(), seg.get_state()] for seg in self.data["segments"]], dtype=np.float32)
        mission_lengths = np.array([[mis.get_long()[0]] for mis in self.missions], dtype=np.float32)
        num_missions_depends_array = np.zeros((self.data['n_missions'], 10)) #maximum 10 depends
        for idx, mis in enumerate(self.missions):
            depends = mis.get_depends()
            for remove_idx in depend_index:
                if remove_idx in depends:
                    depends.remove(remove_idx)
            num_missions_depends_array[idx][0:len(depends)] = depends
        num_missions_depends_array = np.array(num_missions_depends_array)  # Repeat for each mission

        max_vehicle_positions_size = len(self.missions) * 2   #depend on data input
        
        # Pad segment_info to match mission_lengths
        if len(self.missions) > segment_info.shape[0]:
            padding = np.zeros(((len(self.missions) - segment_info.shape[0]), 2))
            segment_info = np.concatenate([segment_info, padding])
        
        # Pad vehicle_positions to match mission_lengths
        vhicle_missions_depends_array = np.zeros((self.data['n_vehicles'], self.data['n_miss_per_vec'], 10)) #maximum 10 depends
        for i, v in enumerate(self.vehicles):
            waiting_mission = v.get_accepted_missions()
            for remove_idx in depend_index:
                if remove_idx in waiting_mission:
                    waiting_mission.remove(remove_idx)
            for mis_idx, mis in enumerate(waiting_mission):
                vhicle_missions_depends_array[i][mis_idx][0:len(mis.get_depends())] = mis.get_depends()

        vhicle_missions_depends_array = vhicle_missions_depends_array.flatten()/self.data['n_missions']
        
        for i, v in enumerate(self.vehicles):
            # Vehicle positions (exclude self)
            vehicle_positions = []
            for other_v in self.vehicles:
                if other_v != v:
                    x, y = other_v.get_pos().get_point()
                    vehicle_positions.extend([x, y])
            # Padding vehicle_positions
            if not move_vehicle_pos:
                x, y = v.get_pos().get_point()
                vehicle_positions.extend([x, y])
                vehicle_positions.extend([0,0] * (max_vehicle_positions_size - len(vehicle_positions)))
                vehicle_positions = np.array(vehicle_positions, dtype=np.float32).reshape(-1, 2)
            else:
                #vehicle index = index of the current vehicle 
                x,y = self.missions[depend_index[-1]].get_mission_destination().get_point()
                vehicle_positions.extend([x, y])
                vehicle_positions.extend([0,0] * (max_vehicle_positions_size - len(vehicle_positions)))
                vehicle_positions = np.array(vehicle_positions, dtype=np.float32).reshape(-1, 2)

            
            action_memory = np.array(self.action_memory).flatten()
            segment_info = np.reshape(segment_info, (-1))/5000
            mission_lengths = np.reshape(mission_lengths, (-1))/(5000*sqrt(2))
            vehicle_positions = np.reshape(vehicle_positions, (-1))/5000
            num_missions_depends_array = np.reshape(num_missions_depends_array, (-1))/self.data['n_missions']
            
            self.idx_dict_obs = {'segment_info': 0}
            self.idx_dict_obs['mission_lengths']= len(segment_info)
            self.idx_dict_obs['vehicle_positions'] = len(mission_lengths)
            self.idx_dict_obs['num_missions_depends_array'] = len(vehicle_positions)
            self.idx_dict_obs['action_memory'] = len(num_missions_depends_array)
            self.idx_dict_obs['vhicle_missions_depends_array'] = len(action_memory)
            max_len = max(len(segment_info), len(mission_lengths), len(vehicle_positions),
                          len(num_missions_depends_array), len(action_memory), len(vhicle_missions_depends_array))

            segment_info_padded = np.pad(segment_info, (0, max_len - len(segment_info)), mode='constant', constant_values=0)
            mission_lengths_padded = np.pad(mission_lengths, (0, max_len - len(mission_lengths)), mode='constant', constant_values=0)
            vehicle_positions_padded = np.pad(vehicle_positions, (0, max_len - len(vehicle_positions)), mode='constant', constant_values=0)
            num_missions_depends_array_padded = np.pad(num_missions_depends_array, (0, max_len - len(num_missions_depends_array)), mode='constant', constant_values=0)
            action_memory_padded = np.pad(action_memory, (0, max_len - len(action_memory)), mode='constant', constant_values=0)
            vhicle_missions_depends_array_padded = np.pad(vhicle_missions_depends_array, (0, max_len - len(vhicle_missions_depends_array)), mode='constant', constant_values=0)

            sobs[f"vehicle_{i}"] = np.concatenate([
                segment_info_padded,
                mission_lengths_padded,
                vehicle_positions_padded,
                num_missions_depends_array_padded,
                action_memory_padded,
                vhicle_missions_depends_array_padded
            ])            
        return sobs
    
    
    def get_single_observation(self):
        obs = []

        # Prepare segment and mission information once outside the loop
        segment_info = np.array([[seg.get_long(), seg.get_state()] for seg in self.data["segments"]], dtype=np.float32)
        mission_lengths = np.array([[mis.get_long()[0]] for mis in self.missions], dtype=np.float32)
        num_missions_depends_array = np.zeros((self.data['n_missions'], 10)) #maximum 10 depends
        for idx, mis in enumerate(self.missions):
            depends = mis.get_depends()
            num_missions_depends_array[idx][0:len(depends)] = depends
        num_missions_depends_array = np.array(num_missions_depends_array)  # Repeat for each mission

        max_vehicle_positions_size = len(self.missions) * 2   #depend on data input
        
        # Pad segment_info to match mission_lengths
        if len(self.missions) > segment_info.shape[0]:
            padding = np.zeros(((len(self.missions) - segment_info.shape[0]), 2))
            segment_info = np.concatenate([segment_info, padding])
        
        # Pad vehicle_positions to match mission_lengths
        vhicle_missions_depends_array = np.zeros((self.data['n_vehicles'], self.data['n_miss_per_vec'], 10)) #maximum 10 depends
        for i, v in enumerate(self.vehicles):
            waiting_mission = v.get_accepted_missions()
            for mis_idx, mis in enumerate(waiting_mission):
                vhicle_missions_depends_array[i][mis_idx][0:len(mis.get_depends())] = mis.get_depends()

        vhicle_missions_depends_array = vhicle_missions_depends_array.flatten()/30
        
            # Vehicle positions (exclude self)
        vehicle_positions = []
        for other_v in self.vehicles:
            if other_v != v:
                x, y = other_v.get_pos().get_point()
                vehicle_positions.extend([x, y])
        # Padding vehicle_positions
        vehicle_positions.extend([0] * (max_vehicle_positions_size - len(vehicle_positions)))
        vehicle_positions = np.array(vehicle_positions, dtype=np.float32).reshape(-1, 2)

        action_memory = np.array(self.action_memory).flatten()
        segment_info = np.reshape(segment_info, (-1))/5000
        mission_lengths = np.reshape(mission_lengths, (-1))/(5000*sqrt(2))
        vehicle_positions = np.reshape(vehicle_positions, (-1))/5000
        num_missions_depends_array = np.reshape(num_missions_depends_array, (-1))/30
            
            
        obs = np.concatenate([
                segment_info,
                mission_lengths,
                vehicle_positions,
                num_missions_depends_array,
                action_memory,
                vhicle_missions_depends_array
            ])
            # print("vehicle_", i, action_memory)
        return obs
    
    def update_action(self, action):
        action = int(np.argmax(action[1]))
        self.action_memory[action]=1  
          
    def update_mem_obs(self, obs, v_id):
        action_memory = np.array(self.action_memory).flatten()
        obs[f"vehicle_{v_id}"][self.idx_dict_obs['action_memory']:self.idx_dict_obs['action_memory']+len(action_memory)]=action_memory
        return obs
    
    def get_solution(self):
        complete_tasks = 0
        stop = True
        for v in self.vehicles:
            complete_tasks += v.get_earn_completes()
            print(v.check_time(), v.get_ctrl_time())
            if v.check_time()==True:
                stop = False
        print("numbers of completed missions",complete_tasks)
        print("numbers of completed solution",self.solution)
        return stop
    
    def step(self, action_dict, agents = None, states = None):
        rewards = {}
        total_complet_tasks = 0
        total_benef_tasks = 0
        # completed_mission_ids = []
        terminateds = []  # Track episode termination
        truncateds = []   # Track episode truncation
        infos = []        # Per-agent

        wrong_action_penalty = {key: 0 for key in range(0, len(self.vehicles))}
        
        action_out = {}
        
        for idx, vid in enumerate(action_dict):
            if (self.action_memory == 1).all():
                for v in range(self.data['n_vehicles']):
                    if v not in (list(action_out.keys())):
                        action_out[v] = -1
                break
            elif self.max_selection_turn[idx] <1:
                action_out[idx] = -1
                continue
            v_action = action_dict[idx][1]
            v_action =  v_action.detach().numpy()
            # action = int(np.argmax(v_action))
            if agents!=None:
                agent = agents[idx]
                if type(agent).__name__ == "DDQNAgent" and agent.epsilon > self.generator.random():
                    action = self.generator.integers(0, agent.action_size)
                elif type(agent).__name__ == "PPOAgent":
                    action = v_action[0]
                else:
                    action = int(np.argmax(v_action))
            else:
                print("1 loi da xay ra")
                exit(1)
            if self.action_memory[action] == 1 and states!=None:
                agent.add_memory(states[list(states.keys())[idx]],action, [-0.01*avg_reward], states[list(states.keys())[idx]], 1)
                agent.add_global_memory(states[list(states.keys())[idx]],action, [-0.01*avg_reward], states[list(states.keys())[idx]], 1)
                wrong_action_penalty[idx] += -0.01*avg_reward
                action_out[idx] = action
                continue
            if self.action_memory[action] == False:
                self.vehicles[idx].set_mission(action, self.missions)
                self.action_memory[action] = 1
                self.solution[action] = idx
                self.max_selection_turn[idx] -= 1
            # else:
            #     wrong_action_penalty[idx] = -0.01*avg_reward
            action_out[idx] = action

        #clear total_reward for calcule again with new set of mission:
        for v in self.vehicles:
            v.clear_total_reward()  
        done_process_infors = []
        while (1):
            count_done = 0
            terminate = True
            for idx, v in enumerate(self.vehicles):
                done_process_infor = v.process_mission(self.missions)
                done_process_infors.append(done_process_infor)
                if done_process_infor==None:
                    count_done += 1
            for v in self.vehicles:
                if v.inprocess():
                    terminate = False
            if terminate:
                break
        total_system_profit = 0
        intime = False
        for idx, v in enumerate(self.vehicles):
            prof_sys = v.get_vhicle_prof()
            total_system_profit += prof_sys
            total_complet_tasks += v.get_earn_completes()
            if idx in rewards.keys():
                rewards[idx].append(prof_sys)
            elif idx not in rewards.keys():
                rewards[idx] = [prof_sys + wrong_action_penalty[idx]]
            # elif idx not in rewards.keys() and prof_sys <= 0:
            #     rewards[idx] = [-100]
            # elif idx in rewards.keys() and prof_sys <= 0:
            #     rewards[idx].append(-100)

            if v.check_time():
                intime = True
                
        terminateds = self.current_step >= self.max_steps or (np.array(self.action_memory) == 1).all() or intime == False
        truncateds = False  # Assuming no truncation in this example
        if terminateds or (count_done == self.data['n_vehicles']):
            print(self.action_memory)
            self.done = True
        else:
            self.done = False
        self.current_step += 1
        obs = self.get_observations()
        if self.verbose:
            print(self.action_memory)
            print("---> total_prof_sys {} total_complet_tasks {} total_benef_tasks {}".format(total_system_profit,total_complet_tasks,total_benef_tasks))
        return obs, rewards, self.done, truncateds, done_process_infors, action_out
    def step_ma(self, actions):
        total_complet_tasks = 0
        total_benef_tasks = 0
        # completed_mission_ids = []
        terminateds = []  # Track episode termination
        truncateds = []   # Track episode truncation
        infos = []        # Per-agent
        
        for v in self.vehicles:
            v.set_mission(actions, self.missions, mtuple = True)
        for v in self.vehicles:
            v.fit_order()
        
        import time
        start = time.perf_counter()
        while (1):
            terminate = True
            for idx, v in enumerate(self.vehicles):
                v.process_mission(self.missions)
            for v in self.vehicles:
                v.verify_ready()
            for v in self.vehicles:
                if v.inprocess():
                    terminate = False
                    break
            if terminate:
                break
        print("processing time: ",time.perf_counter() - start)
        
        intime = False
        rewards = [0]*self.data['n_vehicles']
        for idx, v in enumerate(self.vehicles):
            total_complet_tasks += v.get_earn_completes()
            total_benef_tasks += v.get_earn_profit()
            rewards[idx] = v.get_vhicle_prof()
            if v.check_time():
                intime = True
            # else:
            #     print("control time", idx, v.get_ctrl_time())
        # Can nhac viec vong lap cua viec thuc hien task va viec tinh toan loi nhuan bi trung lap  - BUGS
        if not intime:
            print("So luong task hoan thanh", total_complet_tasks, " - Loi nhuan tinh toan",  total_benef_tasks)
            print("Phan thuong tich luy nhan duoc", rewards)
            
        print(self.current_step, self.max_steps)
        terminateds =self.current_step >= self.max_steps or intime == False or sum(rewards)>avg_reward
        truncateds = False  # Assuming no truncation in this example
        infos = {}          # Empty info dict for this vehicle
        self.done = terminateds
        self.current_step += 1
        obs = self.get_observations()
        if self.verbose:
            print(self.action_memory)
            print("---> total_prof_sys {} total_complet_tasks {} total_benef_tasks {}".format(rewards,total_complet_tasks,total_benef_tasks))
            
        self.current_step +=  1
        return obs, rewards, terminateds, truncateds, infos
    
    
class SITSEnv(gym.Env): #single agent env
    def __init__(self, data, verbose = True, map__ = None, max_steps = 100):
        super().__init__()
        self.data = data
        self.verbose = verbose
        self.generator = np.random.default_rng(SEED)
        self.lmap = map__
        self.init_vehicles()
        self.init_missions()
        self.current_step = 0
        self.max_steps = max_steps
        self._agent_ids = set()
        self.action_space = Box(-np.inf, np.inf, shape=(data["n_missions"],), dtype='float32')
        
        self.observation_space = Box(-np.inf, np.inf, shape=(1077,1), dtype="float32")                    
        self.action_memory = [0]*data["n_missions"]
        self.solution =  ['None']*(mission_cfg['n_vehicle']*mission_cfg['n_miss_per_vec'])
        self.tg = TaskGenerator(15,self.lmap)
        self.done = True
        
    # def init_missions(self):
    #     missions = []
    #     for item in self.data["decoded_data"]:
    #         m = Mission(item['depart_p'], item['depart_s'], 1, graph=self.data["graph"], verbose = self.verbose)
    #         m.set_depends(item["depends"])
    #         m.register_observer(self.vehicles)
    #         missions.append(m)
    #     self.missions = missions
    #     m.reset()
    #     return missions
    
    def init_missions(self):
        missions = []
        lock = threading.Lock()  # For thread-safe access to the missions list
        def create_mission(item):
            m = Mission(item['depart_p'], item['depart_s'], 1, graph=self.data["graph"], verbose=self.verbose)
            m.set_depends(item["depends"])
            m.register_observer(self.vehicles)            
            with lock:
                missions.append(m)

        threads = []
        for item in self.data["decoded_data"]:
            # Create a thread for each mission
            t = threading.Thread(target=create_mission, args=(item,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        Mission.missionID=0
        return missions  

    def init_vehicles(self):
        vehicles = []
        for i in range(self.data["n_vehicles"]):
            seg = self.generator.choice(self.data["segments"])
            v = Vehicle(0.5, seg.get_endpoints()[0], self.lmap, verbose=self.verbose, tau=self.data['tau'])
            vehicles.append(v)
        v.reset()
        self.vehicles = vehicles
        return vehicles

    def reset(self, *, seed=None, options=None, current_datetime = "None", rfile=True, predict = False):
        
        import time
        start = time.perf_counter()
        
        if (self.done and rfile) or predict:
            self.data = write_config_not_fromfile(self.tg)
            self.current_step = 0
            
        file_reset = time.perf_counter()
        
        self.vehicles = self.init_vehicles()
        if 'missions' in self.data:
            self.missions = copy.deepcopy(self.data['missions'])
        else:
            self.missions = self.init_missions()
        self._agent_ids = {f"vehicle_{i}" for i in range(self.data["n_vehicles"])}
        # Return initial observations with an empty info dict
        self.action_memory = [0]*self.data["n_missions"]
        self.solution =  ['None']*(mission_cfg['n_vehicle']*mission_cfg['n_miss_per_vec'])
        obs = self.get_observations()
        #infos = {agent_id: {} for agent_id in self._agent_ids}  
        end = time.perf_counter()
        
        print("genfile {}, reset_object {}".format(file_reset-start, end-file_reset))
        return obs#, infos  

    def get_observations(self):
        # Prepare segment and mission information once outside the loop
        segment_info = np.array([[seg.get_long(), seg.get_state()] for seg in self.data["segments"]], dtype=np.float32)
        mission_lengths = np.array([[mis.get_long()[0]] for mis in self.missions], dtype=np.float32)
        num_missions_depends_array = np.zeros((self.data['n_missions'], 10)) #maximum 10 depends
        for idx, mis in enumerate(self.missions):
            depends = mis.get_depends()
            num_missions_depends_array[idx][0:len(depends)] = depends
        num_missions_depends_array = np.array(num_missions_depends_array)  # Repeat for each mission

        max_vehicle_positions_size = len(self.missions) * 2   #depend on data input
        
        # Pad segment_info to match mission_lengths
        if len(self.missions) > segment_info.shape[0]:
            padding = np.zeros(((len(self.missions) - segment_info.shape[0]), 2))
            segment_info = np.concatenate([segment_info, padding])
        
        # Pad vehicle_positions to match mission_lengths
            # Vehicle positions (exclude self)
        vehicle_positions = []
        for other_v in self.vehicles:
            x, y = other_v.get_pos().get_point()
            vehicle_positions.extend([x, y])
    
        # Padding vehicle_positions
        vehicle_positions.extend([0] * (max_vehicle_positions_size - len(vehicle_positions)))
        vehicle_positions = np.array(vehicle_positions, dtype=np.float32).reshape(-1, 2)

        segment_info = np.reshape(segment_info, (-1))/5000
        mission_lengths = np.reshape(mission_lengths, (-1))/(5000*sqrt(2))
        vehicle_positions = np.reshape(vehicle_positions, (-1))/5000
        num_missions_depends_array = np.reshape(num_missions_depends_array, (-1))/30
        sobs = np.concatenate([
            segment_info,
            mission_lengths,
            vehicle_positions,
            num_missions_depends_array
        ])
            # print("vehicle_", i, action_memory)
        return sobs
    def get_solution(self):
        complete_tasks = 0
        stop = True
        for v in self.vehicles:
            complete_tasks += v.get_earn_completes()
            print(v.check_time(), v.get_ctrl_time())
            if v.check_time()==True:
                stop = False
        print("numbers of completed missions",complete_tasks)
        print("numbers of completed solution",self.solution)
        return stop
    
    def step(self, action):
        rewards = 0
        total_complet_tasks = 0
        total_benef_tasks = 0
        # completed_mission_ids = []
        terminateds = []  # Track episode termination
        truncateds = []   # Track episode truncation
        infos = []        # Per-agent
        
        for v in self.vehicles:
            v.set_mission(action, self.missions, mtuple = True)
        for v in self.vehicles:
            v.fit_order()
        
        import time
        
        start = time.perf_counter()
            
        while (1):
            terminate = True
            for idx, v in enumerate(self.vehicles):
                v.process_mission(self.missions)
            for v in self.vehicles:
                v.verify_ready()
            for v in self.vehicles:
                if v.inprocess():
                    terminate = False
                    break
            if terminate:
                break
        print("processing time: ",time.perf_counter() - start)
        
        intime = False
        for idx, v in enumerate(self.vehicles):
            total_complet_tasks += v.get_earn_completes()
            total_benef_tasks += v.get_earn_profit()
            rewards += v.get_vhicle_prof()
            if v.check_time():
                intime = True
            # else:
            #     print("control time", idx, v.get_ctrl_time())
        # Can nhac viec vong lap cua viec thuc hien task va viec tinh toan loi nhuan bi trung lap  - BUGS
        if not intime:
            print("So luong task hoan thanh", total_complet_tasks, " - Loi nhuan tinh toan",  total_benef_tasks)
            print("Phan thuong tich luy nhan duoc", rewards)
            
        print(self.current_step, self.max_steps)
        self.done = terminateds = self.current_step >= self.max_steps or (np.array(self.action_memory) == 1).all() or intime == False
        
        # self.done = terminateds = intime == False
        # terminateds = True #fix
        truncateds = False  # Assuming no truncation in this example
        infos = {}          # Empty info dict for this vehicle

        self.current_step += 1
        obs = self.get_observations()
        if self.verbose:
            print(self.action_memory)
            print("---> total_prof_sys {} total_complet_tasks {} total_benef_tasks {}".format(rewards,total_complet_tasks,total_benef_tasks))
        return obs, rewards, terminateds, truncateds, infos

def main():
    # Load data
    with open('./task/task_information.json', 'r') as file:
        json_data = file.read()

    with open('./task/mission_information.json', 'r') as file:
        json_data = file.read()
    load = Load()
    mission_decoded_data, graph, map_ = load.data_load()
    # Prepare data
    data = {
        "n_missions": len(mission_decoded_data),
        "n_vehicles": 1,
        "n_miss_per_vec": 20,
        "decoded_data": mission_decoded_data,
        "segments": map_.get_segments(),
        "graph": graph
    }

    # Register the environment
    register_env("its_env", lambda data: ITSEnv(data))
