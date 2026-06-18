"""Utilities for loading map, tasks and missions.

This module provides JSON decoding helpers and the `Load` helper class used to
construct `Map` and `Graph` objects and to attach offloading tasks to segments.
It also provides convenience functions that return `config` dictionaries used
by simulation code.

Important: these are I/O and conversion helpers; comments added here do not
alter runtime logic.
"""

from physic_definition.system_base.ITS_based import Task
from physic_definition.map.map import Map
from physic_definition.map.decas import *
from physic_definition.map.graph import *
import json
from configs.systemcfg import map_cfg, mission_cfg, apply_thread, apply_detach, score_window_size, tau

import time

def decode_task(obj):
    if 'seg_tasksl' in obj:
        obj['seg_tasksl'] = [Task(*v) for k, v in obj['seg_tasksl'].items()]
    return obj
def decode_mission(obj):
    if 'depart_p' in obj:
        data = obj['depart_p']
        obj['depart_p'] = Point(data[0], data[1])
    else:
        raise ValueError("Wrong file: Mission architecture!!!")
    if 'depart_s' in obj:
        data = obj['depart_s']
        obj['depart_s'] = Point(data[0], data[1])
    else:
        raise ValueError("Wrong file: Mission architecture!!!")
    return obj

class Load:
    map = Map(map_cfg['n_lines'], busy=map_cfg['busy'], fromfile=map_cfg['fromfile'])
    def __init__(self, file_load = 'mission_information.json', graph = None):
        print("Load map and mission from file")
        self.map = Load.map
        #inital map from file
        self.map.draw_segments()
        self.map.draw_map()
        self.map.draw_segments()
        segments = self.map.get_segments()
        if graph == None:
            self.graph = Graph(segments)
        else:
            self.graph = graph
        # load offloading tasks and missions from file
        with open('./task/' + 'task_information.json', 'r') as file:
            json_data_task = file.read()
            task_decoded_data = json.loads(json_data_task, object_hook=decode_task)
        for item in task_decoded_data:
            seg_id = item['seg_id']
            idx = segments.index(seg_id)
            seg = segments[idx]
            seg.set_offloading_tasks(item['seg_tasksl'])
            
        self.read = True
        try:
            with open('./task/'+file_load, 'r') as file:
                json_data_mission = file.read()
                self.mission_decoded_data = json.loads(json_data_mission, object_hook=decode_mission)
        except:
            self.read = False
            raise ValueError("Cannot read file")
    def get_infor(self):
        return self.graph, Load.map
    def data_load(self):
        return self.mission_decoded_data, self.graph, Load.map
    
def write_config(file_read = "mission_information.json", graph=None, from_file = True):
    load = Load(file_read, graph)
    if not load.read:
        return False
    mission_decoded_data, graph, map =  load.data_load()
    # Prepare data
    config = {
        "n_missions": mission_cfg['n_mission'],
        "n_vehicles": mission_cfg['n_vehicle'],
        "n_miss_per_vec": mission_cfg['n_miss_per_vec'],
        "decoded_data": mission_decoded_data,
        "segments": map.get_segments(),
        "graph": graph,
        "thread": apply_thread,
        "detach_thread": apply_detach,
        "score_window_size": score_window_size,
        "tau": tau #min
    }
    print("Finish load data from file")
    return config, graph, map

def write_config_not_fromfile(generator_obj):
    missions, graph, map = generator_obj.gen_mission_non_file(mission_cfg['n_mission'])
    # Prepare data
    config = {
        "n_missions": mission_cfg['n_mission'],
        "n_vehicles": mission_cfg['n_vehicle'],
        "n_miss_per_vec": mission_cfg['n_miss_per_vec'],
        "decoded_data": '',
        "segments": map.get_segments(),
        "graph": graph,
        "missions": missions,
        "thread": apply_thread,
        "detach_thread": apply_detach,
        "score_window_size": score_window_size,
        "tau": tau, #min
        "map": map
    }
    return config