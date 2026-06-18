#!/usr/bin/env python
# Created by "Thieu" at 17:00, 05/05/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from physic_definition.system_base.ITS_based import Map, Task, Point, Mission, Vehicle
from decas import *
from graph import *
import json

import numpy as np
from mealpy import IntegerVar, WOA, Problem, SHADE, GA, PSO, TLO

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

data = {
    "n_missions": len(decoded_data),
    "n_vehicles": 20,
    "n_miss_per_vec": 10,
    "decoded_data": decoded_data,
    "segments": segments,
    "graph": graph
}


n_cplet_tasks=[]
earned_ben=[]
earn_profit = []
class ItsProblem(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, seed=None, **kwargs):
        self.data = data
        self.generator = np.random.default_rng(seed)
        super().__init__(bounds, minmax, **kwargs)

    def penalty_func(self, x, penalty_value=10):
        total_values = 0

        # Mission appears more than 1 in solution
        unique_values, counts = np.unique(x, return_counts=True)
        # Print unique values and their counts
        for value, count in zip(unique_values, counts):
            if value != 0:
                if count > 1:
                    total_values += (count - 1) * penalty_value
        return total_values

    # Calculate the fitness of an individual
    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        sol = x_decoded["myvar"]
        print(sol)
        # penalty_value = self.penalty_func(sol, penalty_value=300)
        # print(sol)


        vehcls = []
        for i in range(self.data["n_vehicles"]):
            seg = self.generator.choice(self.data["segments"])  # chọn randome 1 con đường rôif nó đặt 1 cái xe cái ô tô vào
            v = Vehicle(0.5, seg.get_endpoints()[0], map, verbose=True)
            vehcls.append(v)

        mission = []
        import copy
        vehc = copy.deepcopy(vehcls)
        for item in self.data["decoded_data"]:
            m = Mission(item['depart_p'], item['depart_s'], 1, graph=self.data["graph"], verbose=True)
            m.set_depends(item["depends"])
            m.set_observers(vehc)
            m.set_profit(item['profit'])
            mission.append(m)
        m.reset()
        for v in vehcls:
            v.set_mission(sol, mission)

        while (1):
            terminate = True
            for idx, v in enumerate(vehcls):
                v.process_mission()
                if v.inprocess():
                    terminate = False
            if terminate:
                break

        # for idx, v in enumerate(vehcls):
        #     v.process_mission()

        total_prof_sys = 0
        total_complet_tasks = 0
        total_benef_tasks = 0
        for idx, v in enumerate(vehcls):
            total_prof_sys += v.get_vhicle_prof()
            total_complet_tasks += v.get_earn_completes()
            total_benef_tasks += v.get_earn_profit()
        if (total_complet_tasks>100):
            print(sol)
        earn_profit.append(total_prof_sys)
        n_cplet_tasks.append(total_complet_tasks)
        earned_ben.append(total_benef_tasks) #remaining benifit
        v.reset()
        print("total_prof_sys {}, total_complet_tasks {}, total_benef_tasks {}".format(total_prof_sys, total_complet_tasks, total_benef_tasks))
        return total_prof_sys, total_complet_tasks, total_benef_tasks

ndim = data["n_missions"]
bounds = IntegerVar(lb=(0, )*ndim, ub=(data["n_vehicles"]-1, )*ndim, name="myvar")
problem = ItsProblem(bounds=bounds, minmax="max", data=data, log_to="console", obj_weights=(1.0, 0., 0.), seed=SEED)

epoch = 100

model = PSO.OriginalPSO(epoch=epoch, pop_size=50)
# model = SHADE.L_SHADE(epoch=epoch, pop_size=50)
# model = TLO.OriginalTLO(epoch=epoch, pop_size=50)

model.solve(problem, seed=SEED)

print(f"Best agent: {model.g_best}")                    # Encoded solution
print(f"Best solution: {model.g_best.solution}")        # Encoded solution
print(f"Best fitness: {model.g_best.target.fitness}")
print(f"Best real scheduling: {model.problem.decode_solution(model.g_best.solution)}")      # Decoded (Real) solution

plt.close()


list_completed_tasks = []
list_total_benefits = []
list_global_completed_tasks = []
list_global_total_benefits = []
for idx in range(0, epoch):
    list_completed_tasks.append(model.history.list_current_best[idx].target.objectives[1])
    list_total_benefits.append(model.history.list_current_best[idx].target.objectives[2])

    list_global_completed_tasks.append(model.history.list_global_best[idx].target.objectives[1])
    list_global_total_benefits.append(model.history.list_global_best[idx].target.objectives[2])

plt.plot(list_completed_tasks, label = "list_completed_tasks")
plt.legend()
plt.savefig("./hello/list_completed_tasks.png", dpi=300)
plt.close()

plt.plot(list_total_benefits, label = "list_total_benefits")
plt.legend()
plt.savefig("./hello/list_total_benefits.png", dpi=300)
plt.close()

plt.plot(list_global_completed_tasks, label = "list_global_completed_tasks")
plt.legend()
plt.savefig("./hello/list_global_completed_tasks.png", dpi=300)
plt.close()

plt.plot(list_global_total_benefits, label = "list_global_total_benefits")
plt.legend()
plt.savefig("./hello/list_global_total_benefits.png", dpi=300)
plt.close()

## You can access them all via object "history" like this:
model.history.save_global_best_fitness_chart(filename="hello/gbfc")
model.history.save_local_best_fitness_chart(filename="hello/lbfc")
model.history.save_runtime_chart(filename="hello/rtc")




