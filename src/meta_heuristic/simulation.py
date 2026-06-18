from physic_definition.system_base.ITS_based import Map, Task, Point, Mission, Vehicle
from decas import *
from graph import *
import json


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
map = Map(8, busy=0, fromfile=1)
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
seg.reset()
with open('./task/mission_information.json', 'r') as file:
    json_data = file.read()
decoded_data = json.loads(json_data, object_hook=decode_mission)




import time 
if __name__ == "__main__":
    while(1):
        start = time.time()
        mission = []
        for item in decoded_data:
            m = Mission(item['depart_p'], item['depart_s'], 1, graph=graph)
            m.set_depends(item["depends"])
            mission.append(m)
        m.reset()
        #1 soltion
        sol = [1,2,1,2,3,
               1,2,3,4,1,
               4,3,1,2,3,
               4,1,2,3,1,
               2,3,4,1,2,
               2,1,2,3,4,
               1,2,1,2,3,
               1,2,3,4,1,
               4,3,1,2,3,
               4,1,2,3,1]

        vehcls = []
        numvehcls = 6
        for i in range(numvehcls):
            seg = np.random.choice(segments) #chọn randome 1 con đường rôif nó đặt 1 cái xe cái ô tô vào
            v = Vehicle(0.5, seg.get_endpoints()[0], map)
            v.set_mission(sol, mission)
            vehcls.append(v)
        
        while(1):
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
        for idx, v in enumerate(vehcls):
            total_prof_sys += v.get_vhicle_prof()

        v.reset()
        end = time.time()
        print("total_prof_sys ", total_prof_sys, end-start)
        
        