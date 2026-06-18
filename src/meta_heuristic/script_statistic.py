#!/usr/bin/env python
# Created by "Thieu" at 15:57, 16/06/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
from configs.config import ParaConfig
import pickle
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')) 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')) 
from physic_definition.map import decas

def get_statistic_results():
    list_task = []
    list_benefit = []
    list_fitness = []
    list_trial = []
    list_model = []
    dict_results = {}
    for algorithm in ParaConfig.models:
        for trial in range(0, ParaConfig.N_TRIALS):
            # Load the pickled model from a file
            with open(f"{ParaConfig.PATH_SAVE}/trial-{trial}/{algorithm['name']}/model.pkl", 'rb') as file:
                print(file)
                model = pickle.load(file)

            list_task.append(model.g_best.target.objectives[1])
            list_benefit.append(model.g_best.target.objectives[2])
            list_fitness.append(model.g_best.target.fitness)
            list_trial.append(trial)
            list_model.append(algorithm["name"])
    dict_results["model"] = list_model
    dict_results["trial"] = list_trial
    dict_results["fitness"] = list_fitness
    dict_results["completed_tasks"] = list_task
    dict_results["total_benefits"] = list_benefit

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(dict_results)

    # Save the DataFrame to a CSV file
    df.to_csv(f"{ParaConfig.PATH_SAVE}/statistic.csv", index=False)


# get_statistic_results()
