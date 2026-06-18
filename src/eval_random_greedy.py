"""Baseline evaluation: random and greedy scheduling.

This module implements simple baselines used for comparison with DRL and
meta-heuristic methods. Functions here are intended for offline evaluation and
write results to files under the evaluation output directory defined in
`configs.config.ParaConfig`.
"""

import numpy as np
from mealpy import Problem
from torch import dtype

from physic_definition.system_base.ITS_based import Mission, Vehicle, TaskGenerator
from configs.systemcfg import task_cfg, GLOBAL_SEED
from configs.config import ParaConfig
from collections import Counter
import copy, sys
from utils import write_config_not_fromfile, Load
generator = np.random.default_rng(GLOBAL_SEED)
def run_solution(sol, data, vehcls = []):
    if len(vehcls) == 0:
        vehcls = []
        for i in range(data["n_vehicles"]):
            seg = np.random.choice(data["segments"])  
            v = Vehicle(0.5, seg.get_endpoints()[0], data["map"], task_cfg['tau'])
            vehcls.append(v)        
    mission = []
    if not data["decoded_data"]:
        mission = copy.deepcopy(data['missions'])
        for item in mission:
            item.set_observers(vehcls)
    else:
        for item in data["decoded_data"]:
            m = Mission(item['depart_p'], item['depart_s'], 1, graph=data["graph"])
            m.set_depends(item["depends"])
            m.set_observers(vehcls)
            mission.append(m)
        m.reset()

    for v in vehcls:
        v.set_mission(sol, mission, mtuple=True)
        
    for v in vehcls:
        v.fit_order()
    while (1):
        terminate = True
        for idx, v in enumerate(vehcls):
            v.process_mission()
        for v in vehcls:
            v.verify_ready()
        for v in vehcls:
            if v.inprocess():
                terminate = False
                break
        if terminate:
            break
    total_prof_sys = 0
    total_complet_tasks = 0
    total_benef_tasks = 0
    for idx, v in enumerate(vehcls):
        total_prof_sys += v.get_vhicle_prof()
        total_complet_tasks += v.get_earn_completes()
        total_benef_tasks += v.get_earn_profit()
    v.reset()
    print(total_prof_sys, total_complet_tasks, total_benef_tasks)
    return total_prof_sys, total_complet_tasks, total_benef_tasks


def get_random_solution(n_vehicles, n_missions):
    action =[]
    for i in range(n_vehicles):
        num_m = n_missions//n_vehicles #num_mission per vehicle
        label = [i]*num_m
        order = list(range(num_m))
        generator.shuffle(order)
        action += list(zip(order,label))
    generator.shuffle(action)
    return action

def get_greedy_distance_solution(data):
    missions = data['missions']
    vehcls = []
    for i in range(data["n_vehicles"]):
        seg = np.random.choice(data["segments"])  
        v = Vehicle(0.5, seg.get_endpoints()[0], data["map"], task_cfg['tau'])
        vehcls.append(v) 
    #sort missiong by the distance btw two points
    for i in range(len(missions)):
        for j in range(len(missions)):
            if missions[i].get_long() < missions[j].get_long():
                temp = missions[i]
                missions[i] = missions[j]
                missions[j] = temp
    available_schedule = [data['n_miss_per_vec']]*data['n_vehicles']
    data['missions'] = missions
    
    order_schedule = [0]*data['n_vehicles']
    schedules = []
    for mis in missions:
        start_points = mis.get_dpart()
        min_length = float('inf')
        vid = 0
        for v in vehcls:
            v_pos = v.get_pos()
            data['graph'].dijkstra(start_points,v_pos)
            length = data['graph'].get_min_long()[0]
            if length < min_length and available_schedule[v.get_vid()]:
                min_length = length
                vid = v.get_vid()
        schedules.append((order_schedule[vid],vid))
        available_schedule[vid]-=1
        order_schedule[vid]+=1
    return schedules, vehcls

def random_distribution():
    load = Load()
    graph, map_information =  load.get_infor()
    task_generator = TaskGenerator(1, map_information)
    num_trails = 100
    original_stdout = sys.stdout
    with open(f"{ParaConfig.EVAL_PATH_SAVE}/random_result_.txt", "w") as file:
        sys.stdout = file
        for i in range(num_trails):
            print(i) 
            config= write_config_not_fromfile(task_generator)
            # print(config)
            action =get_random_solution(config['n_vehicles'], config['n_missions'])
            run_solution(action, config)
    sys.stdout = original_stdout

def greedy_distance():
    load = Load()
    graph, map_information =  load.get_infor()
    task_generator = TaskGenerator(1, map_information)
    num_trails = 100
    original_stdout = sys.stdout
    with open(f"{ParaConfig.EVAL_PATH_SAVE}/greedy_distance_result_.txt", "w") as file:
        sys.stdout = file
        for i in range(num_trails):
            print(i)
            config= write_config_not_fromfile(task_generator)
            # print(config)
            action, vehcls =get_greedy_distance_solution(config)
            run_solution(action, config, vehcls)
    sys.stdout = original_stdout

if __name__ == "__main__":
    greedy_distance()
    random_distribution()
    