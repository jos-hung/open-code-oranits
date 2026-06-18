#!/usr/bin/env python
# Created by "Thieu" at 19:31, 11/06/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')) 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')) 
from src.physic_definition.system_base.ITS_based import Task, generate, Mission
from src.physic_definition.map.decas import *
from src.physic_definition.map.graph import *
from configs.systemcfg import map_cfg, mission_cfg
import json
import numpy as np
from mealpy import IntegerVar, ParameterGrid, PermutationVar
from src.models.mha import dict_optimizer_classes
from src.models.problem import ItsProblem
from src.models.visualize import draw_model_figure
from configs.config import ParaConfig
import concurrent.futures as parallel
import time
from src.utils import Load, write_config
import multiprocessing
import copy
def run_trial(trial, algorithm, problem, mha_paras, seed, path_save):

    model = dict_optimizer_classes[algorithm["name"]](**mha_paras)
    model.solve(problem, seed=seed)    
    draw_model_figure(model, path_save=f"{path_save}/trial-{trial}/{algorithm['name']}")
    
    return model

def run_trials(algorithm, problem, mha_paras, n_trials, list_algorithm_seeds, path_save, start = 0):

    with multiprocessing.Pool(processes=ParaConfig.N_CPUS_RUN) as pool:
        tasks = [(trial, algorithm, problem, mha_paras, list_algorithm_seeds[trial-start], path_save) for trial in range(start,start+n_trials)]
        results = pool.starmap(run_trial, tasks)

    return results


def develop_model(algorithm, verbose):
    print(f"Start running: {algorithm['name']}")

    data, graph, map_ = write_config()
    data["map"] = map_
    print(f"verbose is {verbose}")
    mission = []
    for item in data["decoded_data"]:
        m = Mission(item['depart_p'], item['depart_s'], 1, graph=data["graph"])
        m.set_depends(item["depends"])
        mission.append(m)
    data['decoded_data'] = False
    data['missions'] = mission
    n_dim_vecs = data["n_vehicles"]*data["n_miss_per_vec"]
    bounds = [
        PermutationVar(valid_set=tuple(range(0, data["n_missions"])), name="mission_idx"),
        IntegerVar(lb=(0,)*n_dim_vecs,
                   ub=(data["n_vehicles"]-1, )*n_dim_vecs,
                   name="vehicle_idx"),
    ]
    problem = ItsProblem(bounds=bounds, minmax="max", data=data, log_to="console", obj_weights=(1.0, 0., 0.), seed=ParaConfig.SEED_GLOBAL, verbose=verbose)
    for paras_temp in list(ParameterGrid(algorithm["param_grid"])):
        mha_paras = dict((key, paras_temp[key]) for key in algorithm["param_grid"].keys())
        cnt = 0
        for group in ParaConfig.LIST_ALGORITHM_SEEDS:
            run_trials(algorithm, problem, mha_paras, len(group), group, ParaConfig.PATH_SAVE, start = cnt)
            cnt += len(group)
        
        # for trial in range(0, ParaConfig.N_TRIALS):
        #     model = dict_optimizer_classes[algorithm["name"]](**mha_paras)
        #     model.solve(problem, seed=ParaConfig.LIST_ALGORITHM_SEEDS[trial])
        #     draw_model_figure(model, path_save=f"{ParaConfig.PATH_SAVE}/trial-{trial}/{algorithm['name']}")


def eval_model(algorithm, data, cnt = 0):
    
    print(f"Start running: {algorithm['name']}")

    # data, graph, map_ = write_config()
    n_dim_vecs = data["n_vehicles"]*data["n_miss_per_vec"]
    bounds = [
        PermutationVar(valid_set=tuple(range(0, data["n_missions"])), name="mission_idx"),
        IntegerVar(lb=(0,)*n_dim_vecs,
                   ub=(data["n_vehicles"]-1, )*n_dim_vecs,
                   name="vehicle_idx"),
    ]
    problem = ItsProblem(bounds=bounds, minmax="max", data=data, log_to="console", obj_weights=(1.0, 0., 0.), seed=ParaConfig.SEED_GLOBAL, verbose=verbose)
    for paras_temp in list(ParameterGrid(algorithm["param_grid"])):
        mha_paras = dict((key, paras_temp[key]) for key in algorithm["param_grid"].keys())
        save_dir = f"{ParaConfig.EVAL_PATH_SAVE}/-data-{cnt}"
        cnt_ = 0
        flat_vector = np.array(ParaConfig.LIST_ALGORITHM_SEEDS).reshape(1, -1)
        for group in flat_vector:
            run_trials(algorithm, problem, mha_paras, len(group), group, save_dir, start = cnt_)
            cnt_ += len(group)
        # for trial in range(0, ParaConfig.N_TRIALS):
        #     model = dict_optimizer_classes[algorithm["name"]](**mha_paras)
        #     model.solve(problem, seed=ParaConfig.LIST_ALGORITHM_SEEDS[trial])
        #     draw_model_figure(model, path_save=f"{ParaConfig.PATH_SAVE}/trial-{trial}/{algorithm['name']}")

def dual_eval(algorithm, data, cnt = 0):
    time_start = time.perf_counter()
    # eval_model(algorithm[0], data, cnt)
    data_m = []
    cnt_m = []
    for i in range(len(algorithm)):
        data_m.append(copy.deepcopy(data))
        cnt_m.append(cnt)
    with parallel.ProcessPoolExecutor(ParaConfig.N_CPUS_RUN) as executor:
        results = executor.map(eval_model, algorithm, data_m, cnt_m)
    print(f"MHA-ITS Problem DONE: {time.perf_counter() - time_start} seconds")

def many_metaheuristics(**kwargs):
    verbose = kwargs.get('verbose', False)
    time_start = time.perf_counter()
    print(f"MHA-ITS Problem Start!!!")
    generate(mission_f_name="mission_information_metaheu.json")
    # develop_model(ParaConfig.models[0])
    with parallel.ProcessPoolExecutor(ParaConfig.N_CPUS_RUN) as executor:
        model_list = ParaConfig.models
        verbose_list = [verbose] * len(model_list)
        results = executor.map(develop_model, model_list, verbose_list)
    print(f"MHA-ITS Problem DONE: {time.perf_counter() - time_start} seconds")


if __name__ == '__main__':
    many_metaheuristics()
