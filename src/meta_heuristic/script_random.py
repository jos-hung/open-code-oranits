#!/usr/bin/env python
# Created by "Thieu" at 21:22, 05/05/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%
import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(parent_dir))
from physic_definition.system_base.ITS_based import Map, Task, Point, Mission, Vehicle
# from decas import *
# from graph import *
from src.utils import write_config
import json
import numpy as np
from mealpy import IntegerVar, WOA, Problem, SHADE, GA, PSO

np.random.seed(42)


def get_random_solution(n_vehicles, n_missions):
    action =[]
    for i in range(n_vehicles):
        num_m = n_missions//n_vehicles #num_mission per vehicle
        label = [i]*num_m
        order = list(range(num_m))
        np.random.shuffle(order)
        action += list(zip(order,label))
    np.random.shuffle(action)
    return action

data, graph, map_ = write_config()

# LB = (0, )*data["n_missions"]
# UB = (data["n_vehicles"]-1, )*data["n_missions"]

list_results = []
for idx in range(0, 50000):
    sol = get_random_solution(data["n_vehicles"], data["n_missions"])
    # print(sol)
    mission = []
    for item in data["decoded_data"]:
        m = Mission(item['depart_p'], item['depart_s'], 1, graph=data["graph"])
        m.set_depends(item["depends"])
        mission.append(m)
    m.reset()

    vehcls = []
    for i in range(data["n_vehicles"]):
        seg = np.random.choice(data["segments"])  # chọn randome 1 con đường rôif nó đặt 1 cái xe cái ô tô vào
        v = Vehicle(0.5, seg.get_endpoints()[0], map_)
        v.set_mission(sol, mission, mtuple = True)
        vehcls.append(v)
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
    for idx, v in enumerate(vehcls):
        total_prof_sys += v.get_vhicle_prof()

    v.reset()
    list_results.append(total_prof_sys)
    print("total_prof_sys ", total_prof_sys)

print(f"Max result = {np.max(list_results)}")
