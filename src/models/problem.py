#!/usr/bin/env python
# Created by "Thieu" at 19:38, 11/06/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy import Problem
from torch import dtype

from src.physic_definition.system_base.ITS_based import Mission, Vehicle
from configs.systemcfg import task_cfg
from collections import Counter
import copy

class ItsProblem(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, seed=None, **kwargs):
        self.data = data
        self.generator = np.random.default_rng(seed)
        super().__init__(bounds, minmax, **kwargs)
        print(kwargs)
        if "verbose" not in kwargs:
            self.verbose = True

    # def penalty_func(self, x, penalty_value=10):
    #     total_values = 0
    #
    #     # Mission appears more than 1 in solution
    #     unique_values, counts = np.unique(x, return_counts=True)
    #     # Print unique values and their counts
    #     for value, count in zip(unique_values, counts):
    #         if value != 0:
    #             if count > 1:
    #                 total_values += (count - 1) * penalty_value
    #     return total_values

    @staticmethod
    def round(x):
        frac = x - np.floor(x)
        t1 = np.floor(x)
        t2 = np.ceil(x)
        return np.where(frac < 0.5, t1, t2)

    def correct_solution(self, x: np.ndarray) -> np.ndarray:
        """
        Correct the solution to valid bounds

        Args:
            x (np.ndarray): The real-value solution

        Returns:
            The corrected solution
        """
        x = self.correct_solution_with_bounds(x, self.bounds)
        x = self.round(x)
        vec_idx = np.array(x[self.data["n_missions"]:], dtype=int).tolist()      # n_miss_per_vec

        def balance_array(x, target_count, abc=0):
            counts = Counter(x)
            if abc >= len(x):
                return x
            value = x[abc]
            if counts[value] == target_count:
                return balance_array(x, target_count, abc + 1)
            elif counts[value] > target_count:
                x.pop(abc)
                counts[value] -= 1
                return balance_array(x, target_count, abc)
            elif counts[value] < target_count:
                x.insert(abc, value)
                counts[value] += 1
                return balance_array(x, target_count, abc + 1)

        counts = Counter(vec_idx)
        keys = sorted(counts.keys())
        N = self.data["n_vehicles"]
        full_set = set(range(N))
        missing_numbers = full_set - set(keys)
        if len(missing_numbers) > 0:
            for val in missing_numbers:
                most_common_element = counts.most_common(1)[0][0]
                change_idx = vec_idx.index(most_common_element)
                vec_idx[change_idx] = val

        temp = np.array(balance_array(vec_idx, self.data["n_miss_per_vec"]))
        x[self.data["n_missions"]:] = temp
        return x


    # Calculate the fitness of an individual
    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        miss_idx = x_decoded["mission_idx"]
        vehicle_idx = x_decoded["vehicle_idx"]
        sol = list(zip(np.array(miss_idx)%self.data["n_vehicles"], vehicle_idx))

        vehcls = []
        for i in range(self.data["n_vehicles"]):
            seg = np.random.choice(self.data["segments"])  # chọn randome 1 con đường rôif nó đặt 1 cái xe cái ô tô vào
            v = Vehicle(0.5, seg.get_endpoints()[0], self.data["map"], task_cfg['tau'], verbose=self.verbose)
            vehcls.append(v)
            
        # print(sol)
        mission = []
        if not self.data["decoded_data"]:
            mission = copy.deepcopy(self.data['missions'])
            for item in mission:
                item.set_observers(vehcls)
        else:
            for item in self.data["decoded_data"]:
                m = Mission(item['depart_p'], item['depart_s'], 1, graph=self.data["graph"], verbose=self.verbose)
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
        # print("total_prof_sys {}, total_complet_tasks {}, total_benef_tasks {}".format(total_prof_sys, total_complet_tasks, total_benef_tasks))
        return total_prof_sys, total_complet_tasks, total_benef_tasks


def get_model_history(model):
    list_completed_tasks = []
    list_total_benefits = []
    list_global_completed_tasks = []
    list_global_total_benefits = []
    for idx in range(0, model.epoch):
        list_completed_tasks.append(model.history.list_current_best[idx].target.objectives[1])
        list_total_benefits.append(model.history.list_current_best[idx].target.objectives[2])

        list_global_completed_tasks.append(model.history.list_global_best[idx].target.objectives[1])
        list_global_total_benefits.append(model.history.list_global_best[idx].target.objectives[2])
    return list_completed_tasks, list_total_benefits, list_global_completed_tasks, list_global_total_benefits
